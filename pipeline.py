"""
src/pipeline.py
─────────────────────────────────────────────────────────────────────────────
Provides TrafficAnalyticsEngine — the central orchestration class that ties
together all components into a single, clean execution loop.

Responsibilities
────────────────
  1. Open and validate the video source.
  2. Instantiate YOLO with ByteTrack.
  3. For each frame:
       a. Resize to the processing resolution.
       b. Read the video timestamp.
       c. Run YOLO inference + ByteTrack.
       d. For every valid detection:
             • Project the ground-contact point through the homography.
             • Update the vehicle tracker.
             • Draw the bounding box and text labels.
       e. Draw the HUD strip.
       f. Display the frame.
       g. Handle keyboard interrupt.
  4. Release all resources cleanly regardless of how the loop exits.

Author : Elalmi CHETTOUH  (Automation Engineer)
License: MIT 2026
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

import config
from src.homography import PerspectiveTransformer
from src.tracker   import VehicleState, VehicleTracker

logger = logging.getLogger(__name__)


class TrafficAnalyticsEngine:
    """
    End-to-end traffic analytics pipeline.

    Parameters
    ----------
    video_path : str, optional
        Override the VIDEO_PATH from config.py (useful for CLI arguments).

    Example
    -------
    ::

        engine = TrafficAnalyticsEngine()
        engine.run()
    """

    def __init__(self, video_path: Optional[str] = None) -> None:
        self._video_path: str = video_path or config.VIDEO_PATH

        logger.info("Initialising TrafficAnalyticsEngine …")

        # ── Perspective transformer ───────────────────────────────────────────
        self._transformer = PerspectiveTransformer(
            src_points    = config.SRC_POINTS,
            real_width_m  = config.REAL_WIDTH_M,
            real_length_m = config.REAL_LENGTH_M,
            scale         = config.WARP_SCALE,
        )

        # ── Vehicle tracker registry ──────────────────────────────────────────
        self._tracker = VehicleTracker()

        # ── YOLO model ────────────────────────────────────────────────────────
        logger.info("Loading YOLO model: %s", config.MODEL_NAME)
        self._model = YOLO(config.MODEL_NAME)
        logger.info("Model loaded.")

        # ── Video capture handle (opened in run()) ────────────────────────────
        self._cap: Optional[cv2.VideoCapture] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Open the video source and start the main processing loop.

        The loop runs until the video ends or the user presses Q.
        Resources are released in a finally block, so they are freed even
        if an exception is raised mid-loop.
        """
        self._open_video()

        frame_count: int = 0

        logger.info(
            "Starting loop on '%s'  —  press Q to quit.",
            self._video_path,
        )

        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    logger.info("End of video stream reached.")
                    break

                frame_count += 1

                # Resize 4 K → processing resolution
                frame = cv2.resize(
                    frame,
                    (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT),
                )

                # Video clock position in seconds (more accurate than wall clock)
                timestamp_sec: float = (
                    self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                )

                # YOLO inference + ByteTrack association
                results = self._model.track(
                    frame,
                    persist         = True,
                    classes         = config.VEHICLE_CLASSES,
                    conf            = config.CONF_THRESHOLD,
                    tracker         = config.TRACKER_CONFIG,
                    verbose         = False,
                )

                # Process detections and draw overlays
                self._process_detections(frame, results, timestamp_sec)

                # Bottom HUD strip
                self._draw_hud(frame, frame_count)

                cv2.imshow(config.WINDOW_TITLE, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User requested quit (Q key).")
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by keyboard (Ctrl+C).")

        except Exception as exc:
            logger.exception("Fatal error in processing loop: %s", exc)
            raise

        finally:
            self._release()
            logger.info(
                "Session complete.  Total frames processed: %d", frame_count
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Private — frame-level processing
    # ─────────────────────────────────────────────────────────────────────────

    def _process_detections(
        self,
        frame: np.ndarray,
        results,
        timestamp_sec: float,
    ) -> None:
        """
        Iterate over every YOLO/ByteTrack detection in the current frame,
        update the tracker, and draw visual output.

        Parameters
        ----------
        frame : np.ndarray
            The 1280×720 BGR frame (mutated in-place by drawing calls).

        results :
            Output of model.track() — a list of ultralytics Results objects.

        timestamp_sec : float
            Current video clock position in seconds.
        """
        if not results or results[0].boxes is None:
            return

        boxes_data = results[0].boxes

        # boxes_data.id is None for the first frame before ByteTrack warms up
        if boxes_data.id is None:
            return

        for i in range(len(boxes_data)):
            track_id: int   = int(boxes_data.id[i].item())
            conf:     float = float(boxes_data.conf[i].item())

            if conf < config.CONF_THRESHOLD:
                continue

            # Bounding box in pixel coordinates (already at 1280×720)
            xyxy = boxes_data.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)

            # Ground-contact point — bottom-centre of the bounding box.
            # This approximates the tyre–road contact and lies on the road
            # plane, which is the plane the homography was calibrated for.
            foot_x: int = (x1 + x2) // 2
            foot_y: int = y2

            # Project to real-world metres
            try:
                world_pos = self._transformer.pixel_to_world(foot_x, foot_y)
            except RuntimeError as exc:
                logger.debug(
                    "Skipping track %d: homography projection failed — %s",
                    track_id,
                    exc,
                )
                continue

            # Update tracker and get the enriched state back
            state: VehicleState = self._tracker.update(
                track_id      = track_id,
                world_pos     = world_pos,
                timestamp_sec = timestamp_sec,
            )

            # ── Draw green bounding box ───────────────────────────────────────
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                config.BOX_COLOR,
                config.BOX_THICKNESS,
            )

            # ── Draw speed label (closer to box) ─────────────────────────────
            h_speed = self._put_label(
                img        = frame,
                text       = state.speed_label(),
                origin     = (x1, y1 - 4),
                text_color = config.SPEED_COLOR,
            )

            # ── Draw ID + direction label (above speed label) ─────────────────
            self._put_label(
                img        = frame,
                text       = state.id_label(),
                origin     = (x1, y1 - 4 - h_speed - 2),
                text_color = config.ID_COLOR,
            )

    def _draw_hud(self, frame: np.ndarray, frame_count: int) -> None:
        """
        Render the bottom status strip showing frame index and total vehicle
        count for the session.

        Parameters
        ----------
        frame : np.ndarray
            The 1280×720 BGR frame (mutated in-place).

        frame_count : int
            Current frame index (1-based).
        """
        hud_text: str = (
            f"Frame {frame_count:05d}  |  "
            f"Vehicles tracked: {self._tracker.active_count():3d}  |  "
            f"Q = quit"
        )
        self._put_label(
            img        = frame,
            text       = hud_text,
            origin     = (8, config.DISPLAY_HEIGHT - 10),
            font_scale = config.HUD_FONT_SCALE,
            text_color = config.HUD_COLOR,
            bg_color   = (0, 0, 0),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Private — video lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def _open_video(self) -> None:
        """
        Open the video capture.  Raises FileNotFoundError if the source
        cannot be opened (covers both missing files and broken RTSP streams).
        """
        self._cap = cv2.VideoCapture(self._video_path)

        if not self._cap.isOpened():
            raise FileNotFoundError(
                f"Cannot open video source: '{self._video_path}'\n"
                "Check that the file exists and the path is correct."
            )

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        w   = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Video opened — source resolution %d×%d @ %.2f FPS", w, h, fps)

    def _release(self) -> None:
        """Release the VideoCapture and destroy all OpenCV windows."""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            logger.debug("VideoCapture released.")
        cv2.destroyAllWindows()
        logger.debug("OpenCV windows destroyed.")

    # ─────────────────────────────────────────────────────────────────────────
    # Private — drawing utility
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _put_label(
        img:        np.ndarray,
        text:       str,
        origin:     tuple[int, int],
        font_scale: float                = config.LABEL_FONT_SCALE,
        thickness:  int                  = 2,
        text_color: tuple[int, int, int] = config.BOX_COLOR,
        bg_color:   tuple[int, int, int] = config.LABEL_BG_COLOR,
    ) -> int:
        """
        Render a single line of text with a filled background rectangle.

        Parameters
        ----------
        img        : np.ndarray  — frame to draw on (mutated in-place).
        text       : str         — string to render.
        origin     : (ox, oy)    — bottom-left of the text baseline.
        font_scale : float
        thickness  : int
        text_color : BGR tuple
        bg_color   : BGR tuple

        Returns
        -------
        int
            Total pixel height consumed (text height + baseline + padding).
            Useful for stacking multiple labels above a bounding box.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        ox, oy = origin

        # Background rectangle
        cv2.rectangle(
            img,
            (ox - 2,          oy - text_h - baseline - 3),
            (ox + text_w + 2, oy + baseline + 1),
            bg_color,
            -1,
        )

        # Text
        cv2.putText(
            img, text, (ox, oy),
            font, font_scale, text_color, thickness, cv2.LINE_AA,
        )

        return text_h + baseline + 3
