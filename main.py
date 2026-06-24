"""
main.py
─────────────────────────────────────────────────────────────────────────────
Entry point for the Real-Time Traffic Analytics Engine.

Usage
─────
    # Default video path from config.py:
    python main.py

    # Override video path at runtime:
    python main.py --video "C:/path/to/your/video.mp4"

Author : Elalmi CHETTOUH  (Automation Engineer)
License: MIT 2026
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.pipeline import TrafficAnalyticsEngine


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def _configure_logging(level: str = "INFO") -> None:
    """
    Configure the root logger with a clean, timestamped format.

    Parameters
    ----------
    level : str
        Standard Python logging level name, e.g. "DEBUG", "INFO", "WARNING".
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level  = numeric_level,
        format = "%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
        datefmt= "%H:%M:%S",
        stream = sys.stdout,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog        = "traffic_analytics",
        description = "Real-Time Traffic Analytics Engine — "
                      "Continuous Speed Estimation via Homography Matrix Projection.",
    )
    parser.add_argument(
        "--video",
        type    = str,
        default = None,
        help    = "Path to the source video file or RTSP stream URL.  "
                  "Overrides VIDEO_PATH in config.py.",
    )
    parser.add_argument(
        "--log-level",
        type    = str,
        default = "INFO",
        choices = ["DEBUG", "INFO", "WARNING", "ERROR"],
        help    = "Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("=" * 66)
    logger.info("  Real-Time Traffic Analytics Engine")
    logger.info("  Author: Elalmi CHETTOUH  (Automation Engineer)")
    logger.info("=" * 66)

    try:
        engine = TrafficAnalyticsEngine(video_path=args.video)
        engine.run()

    except FileNotFoundError as exc:
        logger.error("Video source error: %s", exc)
        sys.exit(1)

    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        sys.exit(2)


if __name__ == "__main__":
    main()
