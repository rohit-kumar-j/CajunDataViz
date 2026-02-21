"""
test_modular.py

Entry point for the Go2 DataViz viewer.

Usage:
    python test_modular.py
    python test_modular.py --config config.json
    python test_modular.py --config config.json --urdf go2/urdf/go2.urdf --data go2/data/run1
    python test_modular.py --help

All unhandled exceptions are caught, logged, and printed before exit so that
no error is silently swallowed.
"""

import argparse
import sys
import traceback

# ---------------------------------------------------------------------------
# Logger — configure BEFORE importing any dataviz modules so that
# all module-level log calls use this setup.
# The format includes {file}:{line} so every log line shows its source.
# ---------------------------------------------------------------------------
from loguru import logger

logger.remove()   # remove default handler
logger.add(
    sys.stderr,
    level="DEBUG",
    format=(
        "<green>{time:HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{file}</cyan>:<cyan>{line}</cyan> | "
        "{message}"
    ),
    colorize=True,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Go2 DataViz — robot motion replay viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_modular.py
  python test_modular.py --config my_config.json
  python test_modular.py --urdf robots/go2/urdf/go2.urdf --data robots/go2/data/run1
        """,
    )
    parser.add_argument(
        "--config",
        default="config.json",
        metavar="PATH",
        help="Path to config.json  (default: config.json)",
    )
    parser.add_argument(
        "--urdf",
        default=None,
        metavar="PATH",
        help="Optional: direct path to URDF file to load on startup",
    )
    parser.add_argument(
        "--data",
        default=None,
        metavar="DIR",
        help="Optional: direct path to data directory to load on startup",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()

    logger.info(f"test_modular.py starting")
    logger.info(f"  --config : {args.config}")
    logger.info(f"  --urdf   : {args.urdf  or '(none)'}")
    logger.info(f"  --data   : {args.data  or '(none)'}")

    # ── Load config first — all other modules depend on CFG ───────────────
    try:
        from dataviz.config import init_config
        cfg = init_config(args.config)
        logger.info("Config loaded OK")
    except FileNotFoundError as exc:
        logger.critical(f"Config file not found: {exc}")
        sys.exit(1)
    except ValueError as exc:
        logger.critical(f"Config file invalid: {exc}")
        sys.exit(1)

    # ── Validate --urdf / --data pairing ──────────────────────────────────
    if bool(args.urdf) != bool(args.data):
        logger.critical(
            "You must provide BOTH --urdf and --data, or neither.\n"
            f"  --urdf: {args.urdf}\n"
            f"  --data: {args.data}"
        )
        sys.exit(1)

    # ── Launch viewer ─────────────────────────────────────────────────────
    try:
        from dataviz import run_viewer
        run_viewer(
            config_path  = args.config,
            initial_urdf = args.urdf,
            initial_data = args.data,
        )
    except FileNotFoundError as exc:
        logger.critical(f"File not found: {exc}")
        sys.exit(1)
    except RuntimeError as exc:
        logger.critical(f"Runtime error: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl-C)")
        sys.exit(0)
    except Exception as exc:
        logger.critical(
            f"Unhandled exception: {type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc()}"
        )
        sys.exit(1)

    logger.info("Exited cleanly")


if __name__ == "__main__":
    main()
