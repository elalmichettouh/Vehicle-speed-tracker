"""
src/tracker.py
─────────────────────────────────────────────────────────────────────────────
Provides two classes:

  VehicleState
      Immutable-ish data container + online speed/direction estimator for a
      single ByteTrack ID.  Updated every frame the vehicle is visible.

  VehicleTracker
      Registry that owns all active VehicleState objects, creates new ones
      on first detection, and exposes a clean update interface to the pipeline.

Speed Estimation
────────────────
Each frame, the ground-contact point (bottom-centre of the bounding box) is
projected to real-world metres via the homography.  The instantaneous speed
is derived from:

    speed (m/s) = Δdistance_m / Δtime_s
    speed (km/h) = speed (m/s) × 3.6

A deque of fixed length acts as a sliding-window average to suppress
per-frame jitter caused by bounding-box wobble in the detector.

Direction Estimation
────────────────────
After the perspective warp, the Y-axis of the road plane corresponds to
depth from the camera.  The sign of the mean warped-Δy over several frames
determines direction:

    mean_Δy > 0  →  vehicle moving away from camera   → "OUT"
    mean_Δy < 0  →  vehicle moving toward camera       → "IN"

Author : Elalmi CHETTOUH  (Automation Engineer)
License: MIT 2026
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)


class VehicleState:
    """
    Holds all per-track runtime state for a single tracked vehicle.

    Parameters
    ----------
    track_id : int
        Unique integer assigned by ByteTrack.

    Attributes
    ----------
    track_id : int
    smooth_speed : Optional[float]
        Moving-average speed in km/h.  None until at least two measurements.
    direction : Optional[str]
        "IN", "OUT", or None if direction has not yet been committed.
    """

    def __init__(self, track_id: int) -> None:
        self.track_id: int = track_id

        # Real-world position and timestamp from the previous frame
        self._prev_pos:  Optional[Tuple[float, float]] = None
        self._prev_time: Optional[float]               = None

        # Sliding-window buffers
        self._speed_buf: deque[float] = deque(maxlen=config.SPEED_BUFFER_LEN)
        self._dy_buf:    deque[float] = deque(maxlen=config.SPEED_BUFFER_LEN)

        # Public read-outs
        self.smooth_speed: Optional[float] = None
        self.direction:    Optional[str]   = None

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def update(
        self,
        world_pos: Tuple[float, float],
        timestamp_sec: float,
    ) -> None:
        """
        Ingest a new real-world position and update speed / direction.

        Parameters
        ----------
        world_pos : Tuple[float, float]
            (world_x_m, world_y_m) from PerspectiveTransformer.pixel_to_world.

        timestamp_sec : float
            Video clock position in seconds for this frame, obtained via
            cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0.
        """
        if self._prev_pos is not None and self._prev_time is not None:
            dt: float = timestamp_sec - self._prev_time

            if dt > 0.0:
                dx: float = world_pos[0] - self._prev_pos[0]
                dy: float = world_pos[1] - self._prev_pos[1]

                distance_m: float  = float(np.hypot(dx, dy))
                speed_kmh:  float  = (distance_m / dt) * 3.6

                # Discard physically impossible artefacts (sudden track jumps)
                if speed_kmh < config.MAX_PLAUSIBLE_SPEED_KMH:
                    self._speed_buf.append(speed_kmh)

                # Accumulate direction evidence
                self._dy_buf.append(dy)
                self._update_direction()

        # Persist state for the next frame
        self._prev_pos  = world_pos
        self._prev_time = timestamp_sec

        # Recompute moving average
        if self._speed_buf:
            self.smooth_speed = sum(self._speed_buf) / len(self._speed_buf)

    def speed_label(self) -> str:
        """Human-readable speed string for HUD overlay."""
        if self.smooth_speed is not None:
            return f"{self.smooth_speed:.1f} km/h"
        return "-- km/h"

    def direction_label(self) -> str:
        """Human-readable direction string for HUD overlay."""
        return self.direction if self.direction is not None else "..."

    def id_label(self) -> str:
        """Human-readable ID + direction label for HUD overlay."""
        return f"ID:{self.track_id} | {self.direction_label()}"

    def reset_timing(self) -> None:
        """
        Clear temporal state.  Call this if the track is lost for several
        frames and re-acquired (track ID reuse may corrupt the Δt calculation).
        """
        self._prev_pos  = None
        self._prev_time = None
        self._speed_buf.clear()
        self._dy_buf.clear()
        self.smooth_speed = None
        self.direction    = None
        logger.debug("VehicleState %d timing reset.", self.track_id)

    def __repr__(self) -> str:
        return (
            f"VehicleState(id={self.track_id}, "
            f"speed={self.smooth_speed}, "
            f"direction={self.direction})"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _update_direction(self) -> None:
        """
        Commit direction once enough consistent dy samples are available.
        Uses the mean of the dy buffer to resist single-frame noise.
        """
        if len(self._dy_buf) < config.DIRECTION_COMMIT_FRAMES:
            return

        mean_dy: float = sum(self._dy_buf) / len(self._dy_buf)
        self.direction = "OUT" if mean_dy > 0.0 else "IN"


# ─────────────────────────────────────────────────────────────────────────────

class VehicleTracker:
    """
    Registry of all active VehicleState objects.

    The pipeline calls update() once per frame for every detected bounding
    box.  VehicleTracker handles creation of new states and looks up existing
    ones by track ID.

    Usage
    -----
    ::

        tracker = VehicleTracker()
        state   = tracker.update(track_id=5, world_pos=(3.2, 18.7), t=1.034)
        print(state.speed_label())

    """

    def __init__(self) -> None:
        self._registry: Dict[int, VehicleState] = {}
        logger.info("VehicleTracker initialised.")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def update(
        self,
        track_id:      int,
        world_pos:     Tuple[float, float],
        timestamp_sec: float,
    ) -> VehicleState:
        """
        Look up or create the VehicleState for track_id, then update it.

        Parameters
        ----------
        track_id : int
            ByteTrack unique ID for this detection.

        world_pos : Tuple[float, float]
            Real-world (x_m, y_m) from PerspectiveTransformer.

        timestamp_sec : float
            Video clock position in seconds.

        Returns
        -------
        VehicleState
            The updated state object (convenient for the pipeline to read
            labels immediately after calling update).
        """
        if track_id not in self._registry:
            self._registry[track_id] = VehicleState(track_id)
            logger.debug("New track registered: ID %d", track_id)

        state: VehicleState = self._registry[track_id]
        state.update(world_pos, timestamp_sec)
        return state

    def get(self, track_id: int) -> Optional[VehicleState]:
        """Return the VehicleState for track_id, or None if unknown."""
        return self._registry.get(track_id)

    def remove(self, track_id: int) -> None:
        """
        Remove a track from the registry (e.g. when the vehicle leaves the
        scene and the ID will not be reused).
        """
        if track_id in self._registry:
            del self._registry[track_id]
            logger.debug("Track %d removed from registry.", track_id)

    def active_count(self) -> int:
        """Total number of track IDs ever registered in this session."""
        return len(self._registry)

    def __len__(self) -> int:
        return self.active_count()

    def __repr__(self) -> str:
        return f"VehicleTracker(active_tracks={self.active_count()})"
