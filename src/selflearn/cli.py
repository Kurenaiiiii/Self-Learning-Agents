"""CLI skeleton. Full commands land in later phases."""

from __future__ import annotations

import argparse
import sys

from selflearn import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    p = argparse.ArgumentParser(prog="selflearn", description="Self-Learning Agents")
    p.add_argument("--version", action="store_true", help="print version")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("doctor", help="check config and connectivity")
    sub.add_parser("chat", help="chat with the memory agent")
    sub.add_parser("memories", help="manage memories")
    sub.add_parser("eval", help="run the benchmark")
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    print("selflearn: skeleton CLI (full commands arrive in later phases).")
    print("Run `selflearn --help` for usage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
