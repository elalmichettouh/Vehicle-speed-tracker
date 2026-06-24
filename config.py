"""
config.py
─────────────────────────────────────────────────────────────────────────────
Central configuration for the Real-Time Traffic Analytics Engine.

All tuneable parameters live here. No magic numbers should appear anywhere
else in the codebase — import from this module instead.

Author : Elalmi CHETTOUH  (Automation Engineer)
License: MIT 2026
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# VIDEO SOURCE
# ─────────────────────────────────────────────────────────────────────────────

# Absolute path to the source video file.
# Supports local files and RTSP streams (e.g. "rtsp://192.168.1.1/stream").
VIDEO_PATH: str = r"C:\Users\CHETTOUH\Downloads\4K Video of Highway Traffic!.mp4"

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY / PROCESSING RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

# The 4K source is downscaled to this resolution before inference.
# Reducing further (e.g. 960×540) trades accuracy for speed on slow hardware.
DISPLAY_WIDTH:  int = 1280
DISPLAY_HEIGHT: int = 720

# ─────────────────────────────────────────────────────────────────────────────
# PERSPECTIVE CALIBRATION  ← most important section
# ─────────────────────────────────────────────────────────────────────────────
#
# SRC_POINTS defines a quadrilateral on the ROAD SURFACE as it appears in the
# 1280×720 camera frame.  The four corners must correspond to a real-world
# rectangle whose physical dimensions you know.
#
# Calibration workflow
# ────────────────────
#  1. Run the helper script:  python tools/extract_frame.py
#     → saves  assets/frame0.png  (1280×720)
#
#  2. Open frame0.png in any image editor that displays pixel coordinates
#     (MS Paint, GIMP, Photoshop).
#
#  3. Identify four points on the road that form a rectangle when viewed from
#     above — e.g. pairs of lane-marking dashes at a known spacing.
#     Click each corner and note its (x, y) pixel coordinate.
#     Record them in TOP-LEFT → TOP-RIGHT → BOTTOM-RIGHT → BOTTOM-LEFT order
#     (as they appear on screen, not on the ground).
#
#  4. Paste the pixel coordinates into SRC_POINTS below.
#
#  5. Measure (or look up on Google Maps satellite view) the physical width
#     and length of that same rectangle in metres, and fill in
#     REAL_WIDTH_M and REAL_LENGTH_M.
#
#  6. Re-run the engine and validate: a vehicle travelling at a known speed
#     (e.g. a highway cruising speed of ~110 km/h) should read close to that
#     value.  If readings are systematically too high, reduce REAL_LENGTH_M;
#     if too low, increase it.
#
# Visual guide:
#
#   Camera frame (1280×720)
#   ┌─────────────────────────────────┐
#   │          vanishing point        │
#   │        TL ──────── TR           │  ← far edge of road patch
#   │       /                \        │
#   │      /    road patch    \       │
#   │     BL ──────────────── BR      │  ← near edge of road patch
#   └─────────────────────────────────┘
#
#   After the homography warp these become a flat rectangle:
#   TL ──── TR
#   │        │   REAL_WIDTH_M wide
#   BL ──── BR   REAL_LENGTH_M tall
#
SRC_POINTS: np.ndarray = np.float32([
    [490, 310],   # TL — top-left  of road patch  ← ADJUST
    [790, 310],   # TR — top-right of road patch  ← ADJUST
    [980, 600],   # BR — bottom-right              ← ADJUST
    [300, 600],   # BL — bottom-left               ← ADJUST
])

# Physical width of the road patch  (TL ↔ TR on the ground), in metres.
REAL_WIDTH_M:  float = 10.0   # ← ADJUST

# Physical length of the road patch  (TL ↔ BL on the ground), in metres.
REAL_LENGTH_M: float = 40.0   # ← ADJUST

# Internal scale factor: 1 metre → WARP_SCALE units in the warped coordinate
# space.  100 means 1 warped unit = 1 cm.  Do not change unless you have a
# specific reason — it only affects internal floating-point precision.
WARP_SCALE: float = 100.0

# ─────────────────────────────────────────────────────────────────────────────
# YOLO MODEL
# ─────────────────────────────────────────────────────────────────────────────

# Model weights filename.  Ultralytics will download automatically on first run.
#   Speed  (fastest → slowest): yolov8n  yolov8s  yolov8m  yolov8l  yolov8x
#   Accuracy (lowest → highest): same order
MODEL_NAME: str = "yolov8n.pt"

# ByteTrack configuration file bundled with ultralytics.
TRACKER_CONFIG: str = "bytetrack.yaml"

# Minimum detection confidence.  Lower values detect more objects but increase
# false positives.  Range: [0.0, 1.0].
CONF_THRESHOLD: float = 0.35

# COCO class indices to track.
#   2 = car   3 = motorcycle   5 = bus   7 = truck
# Add or remove indices to change which vehicle categories are tracked.
VEHICLE_CLASSES: list[int] = [2, 3, 5, 7]

# ─────────────────────────────────────────────────────────────────────────────
# SPEED ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

# Number of frames over which the instantaneous speed is averaged.
# Higher values → smoother display, slower reaction to real speed changes.
# Recommended range: 5 – 15.
SPEED_BUFFER_LEN: int = 8

# Hard upper bound (km/h).  Instantaneous speed samples above this value are
# discarded as physically impossible tracking artefacts.
MAX_PLAUSIBLE_SPEED_KMH: float = 250.0

# Minimum number of consecutive direction-consistent samples before the
# direction label ("IN" / "OUT") is committed and shown.
DIRECTION_COMMIT_FRAMES: int = 3

# ─────────────────────────────────────────────────────────────────────────────
# VISUAL APPEARANCE
# ─────────────────────────────────────────────────────────────────────────────

# Bounding-box and primary label colour (BGR).
BOX_COLOR:   tuple[int, int, int] = (0, 255, 0)    # green
SPEED_COLOR: tuple[int, int, int] = (0, 255, 0)    # green (speed text)
ID_COLOR:    tuple[int, int, int] = (0, 255, 0)    # green (ID / direction text)
HUD_COLOR:   tuple[int, int, int] = (200, 200, 200)# light grey (HUD strip)

# Label background colour (BGR).
LABEL_BG_COLOR: tuple[int, int, int] = (10, 10, 10)

# Bounding-box line thickness in pixels.
BOX_THICKNESS: int = 2

# Font scale for vehicle labels.  Reduce if labels feel too large.
LABEL_FONT_SCALE: float = 0.55

# Font scale for the bottom HUD strip.
HUD_FONT_SCALE: float = 0.42

# OpenCV window title.
WINDOW_TITLE: str = "Real-Time Traffic Analytics Engine"
