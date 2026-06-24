"""
src/homography.py
─────────────────────────────────────────────────────────────────────────────
Provides PerspectiveTransformer — a self-contained class that:
  • Builds the 3×3 homography matrix once from the calibration quad.
  • Converts any (pixel_x, pixel_y) point on the camera frame to a
    real-world (x_metres, y_metres) coordinate on the flat road plane.
  • Exposes diagnostic helpers for unit-testing and calibration tooling.

Theory
──────
A traffic camera mounted at an angle creates perspective compression:
objects near the top of the frame (further away) appear much smaller than
objects near the bottom (closer).  A naive distance-in-pixels calculation
therefore grossly overestimates the speed of distant vehicles.

cv2.getPerspectiveTransform() solves a system of 8 equations to find the
unique 3×3 projective matrix M such that:

    [u']     [m00 m01 m02] [u]
    [v']  ~  [m10 m11 m12] [v]
    [w']     [m20 m21 m22] [1]

    world_x = u' / w'
    world_y = v' / w'

After mapping, equal pixel differences anywhere in the warped space
correspond to equal physical distances, eliminating the perspective error.

Author : Elalmi CHETTOUH  (Automation Engineer)
License: MIT 2026
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PerspectiveTransformer:
    """
    Encapsulates a single perspective homography for projecting 2-D camera
    pixels onto a metrically correct top-down road plane.

    Parameters
    ----------
    src_points : np.ndarray, shape (4, 2), dtype float32
        Four pixel coordinates on the 1280×720 camera frame that correspond
        to a known rectangle on the road surface.
        Order: TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT.

    real_width_m : float
        Physical width of the road rectangle (TL ↔ TR), in metres.

    real_length_m : float
        Physical length of the road rectangle (TL ↔ BL), in metres.

    scale : float, optional
        Internal unit scale (default 100 → 1 warped unit = 1 cm).
        Higher values improve floating-point precision for small distances.

    Raises
    ------
    ValueError
        If src_points does not have shape (4, 2) or if real dimensions are
        non-positive.
    cv2.error
        If OpenCV cannot compute the homography (e.g. degenerate quad).
    """

    def __init__(
        self,
        src_points: np.ndarray,
        real_width_m: float,
        real_length_m: float,
        scale: float = 100.0,
    ) -> None:
        self._validate_inputs(src_points, real_width_m, real_length_m, scale)

        self._src_points:   np.ndarray = src_points.astype(np.float32)
        self._real_width_m: float      = real_width_m
        self._real_length_m: float     = real_length_m
        self._scale:        float      = scale

        self._matrix: np.ndarray = self._build_matrix()
        logger.info(
            "PerspectiveTransformer initialised  "
            "(road patch %.1f m × %.1f m, scale %.0f)",
            real_width_m,
            real_length_m,
            scale,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def pixel_to_world(
        self,
        pixel_x: float,
        pixel_y: float,
    ) -> Tuple[float, float]:
        """
        Project a single pixel coordinate to real-world metres.

        Parameters
        ----------
        pixel_x, pixel_y : float
            Pixel coordinate on the 1280×720 camera frame.

        Returns
        -------
        world_x_m, world_y_m : Tuple[float, float]
            Real-world position in metres.
            • world_x_m increases from left to right of the road patch.
            • world_y_m increases from the near edge (bottom of frame) toward
              the far edge (top of frame) — i.e. increasing world_y means the
              vehicle is moving *away* from the camera.

        Raises
        ------
        RuntimeError
            If the perspective division produces a near-zero denominator
            (the input point is at or behind the camera's optical centre).
        """
        point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)

        try:
            warped = cv2.perspectiveTransform(point, self._matrix)
        except cv2.error as exc:
            raise RuntimeError(
                f"OpenCV perspectiveTransform failed for point "
                f"({pixel_x:.1f}, {pixel_y:.1f}): {exc}"
            ) from exc

        raw_x = float(warped[0, 0, 0])
        raw_y = float(warped[0, 0, 1])

        world_x = raw_x / self._scale
        world_y = raw_y / self._scale
        return world_x, world_y

    def batch_pixel_to_world(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorised version of pixel_to_world for multiple points.

        Parameters
        ----------
        points : np.ndarray, shape (N, 2), dtype float32
            Array of (pixel_x, pixel_y) pairs.

        Returns
        -------
        np.ndarray, shape (N, 2)
            Array of (world_x_m, world_y_m) pairs.
        """
        pts = points.astype(np.float32).reshape(1, -1, 2)

        try:
            warped = cv2.perspectiveTransform(pts, self._matrix)
        except cv2.error as exc:
            raise RuntimeError(
                f"Batch perspectiveTransform failed: {exc}"
            ) from exc

        return warped.reshape(-1, 2) / self._scale

    @property
    def matrix(self) -> np.ndarray:
        """The raw 3×3 homography matrix (read-only copy)."""
        return self._matrix.copy()

    @property
    def real_width_m(self) -> float:
        return self._real_width_m

    @property
    def real_length_m(self) -> float:
        return self._real_length_m

    def __repr__(self) -> str:
        return (
            f"PerspectiveTransformer("
            f"width={self._real_width_m}m, "
            f"length={self._real_length_m}m, "
            f"scale={self._scale})"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_matrix(self) -> np.ndarray:
        """
        Construct the destination rectangle and compute M via
        cv2.getPerspectiveTransform.

        The destination quad is a perfect rectangle aligned with the axes,
        scaled so that each unit represents (1 / scale) metres.

            DST:  TL=(0, 0)   TR=(W, 0)   BR=(W, H)   BL=(0, H)
            where W = real_width_m  × scale
                  H = real_length_m × scale
        """
        W = self._real_width_m  * self._scale
        H = self._real_length_m * self._scale

        dst_points = np.float32([
            [0.0, 0.0],   # TL
            [W,   0.0],   # TR
            [W,   H  ],   # BR
            [0.0, H  ],   # BL
        ])

        matrix = cv2.getPerspectiveTransform(self._src_points, dst_points)

        if matrix is None:
            raise RuntimeError(
                "cv2.getPerspectiveTransform returned None.  "
                "Check that SRC_POINTS form a valid non-degenerate quadrilateral."
            )

        return matrix

    @staticmethod
    def _validate_inputs(
        src_points: np.ndarray,
        real_width_m: float,
        real_length_m: float,
        scale: float,
    ) -> None:
        if not isinstance(src_points, np.ndarray):
            raise ValueError("src_points must be a numpy ndarray.")
        if src_points.shape != (4, 2):
            raise ValueError(
                f"src_points must have shape (4, 2), got {src_points.shape}."
            )
        if real_width_m <= 0:
            raise ValueError(
                f"real_width_m must be positive, got {real_width_m}."
            )
        if real_length_m <= 0:
            raise ValueError(
                f"real_length_m must be positive, got {real_length_m}."
            )
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}.")
