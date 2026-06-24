"""
tools/extract_frame.py
─────────────────────────────────────────────────────────────────────────────
Calibration helper: extracts a single frame from the source video, saves it
to assets/frame0.png at the processing resolution (1280×720), and optionally
opens an interactive point-picker window so you can click the four calibration
corners and print their pixel coordinates directly to the terminal.

Usage
─────
    # Save frame only:
    python tools/extract_frame.py

    # Save frame AND open interactive point picker:
    python tools/extract_frame.py --pick

Author : Elalmi CHETTOUH  (Automation Engineer)
License: MIT 2026
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


ASSETS_DIR  = os.path.join(os.path.dirname(__file__), "..", "assets")
OUTPUT_PATH = os.path.join(ASSETS_DIR, "frame0.png")

_clicked_points: list[tuple[int, int]] = []


def _mouse_callback(event: int, x: int, y: int, flags: int, param) -> None:
    """Record left-click coordinates and draw a marker on the display frame."""
    if event == cv2.EVENT_LBUTTONDOWN:
        _clicked_points.append((x, y))
        order_labels = ["TL", "TR", "BR", "BL"]
        idx = len(_clicked_points) - 1
        label = order_labels[idx] if idx < 4 else str(idx)

        print(f"  [{label}]  pixel ({x:4d}, {y:4d})")

        frame: np.ndarray = param
        cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(
            frame, label, (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
        )
        cv2.imshow("Point Picker — click TL, TR, BR, BL then press Enter", frame)

        if len(_clicked_points) == 4:
            print("\n── Paste this into config.py ──────────────────────────────")
            print("SRC_POINTS = np.float32([")
            for lbl, pt in zip(order_labels, _clicked_points):
                print(f"    [{pt[0]:4d}, {pt[1]:4d}],   # {lbl}")
            print("])")
            print("────────────────────────────────────────────────────────────\n")


def extract_frame(frame_index: int = 0) -> np.ndarray:
    """
    Open the source video, seek to frame_index, and return the resized frame.
    """
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(
            f"Cannot open video: '{config.VIDEO_PATH}'"
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read frame {frame_index} from video.")

    return cv2.resize(frame, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))


def save_frame(frame: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, frame)
    print(f"[INFO] Frame saved → {os.path.abspath(path)}")


def run_point_picker(frame: np.ndarray) -> None:
    """
    Open an interactive OpenCV window.
    Click the four corners in TL → TR → BR → BL order.
    Press Enter or Q to close.
    """
    display = frame.copy()
    window  = "Point Picker — click TL, TR, BR, BL then press Enter"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, _mouse_callback, display)

    print("\n[INFO] Click the four road corners in this order:")
    print("         1. TOP-LEFT   (TL)")
    print("         2. TOP-RIGHT  (TR)")
    print("         3. BOTTOM-RIGHT (BR)")
    print("         4. BOTTOM-LEFT  (BL)")
    print("       Press Enter or Q when done.\n")

    while True:
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, ord("q")):   # Enter or Q
            break

    cv2.destroyAllWindows()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a calibration frame from the source video."
    )
    parser.add_argument(
        "--frame",
        type    = int,
        default = 0,
        help    = "Zero-based frame index to extract (default: 0).",
    )
    parser.add_argument(
        "--pick",
        action  = "store_true",
        help    = "Open the interactive point-picker after saving the frame.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"[INFO] Extracting frame {args.frame} from:")
    print(f"       {config.VIDEO_PATH}")

    frame = extract_frame(args.frame)
    save_frame(frame, OUTPUT_PATH)

    if args.pick:
        run_point_picker(frame)


if __name__ == "__main__":
    main()
