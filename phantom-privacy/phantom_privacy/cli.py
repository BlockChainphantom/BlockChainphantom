"""Command-line interface for Phantom Privacy.

Usage:
    python -m phantom_privacy phone
    python -m phantom_privacy footprint
    python -m phantom_privacy scan [TARGET] [--ports 1-1024] [--allow-public]
"""

from __future__ import annotations

import argparse
import sys

from . import __version__, footprint, network_scan, phone_privacy


def _ask_yes_no(prompt: str) -> bool | None:
    """Ask a yes/no question. Returns True/False, or None if the user skips."""

    while True:
        raw = input(f"{prompt} [y/n/skip] ").strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        if raw in ("s", "skip", ""):
            return None
        print("  Please answer y, n or skip.")


def cmd_phone(_args: argparse.Namespace) -> int:
    print(phone_privacy.render())
    return 0


def cmd_voip(_args: argparse.Namespace) -> int:
    print("VoIP / masking-number comparison\n")
    print(phone_privacy.render_voip_table())
    return 0


def cmd_footprint(_args: argparse.Namespace) -> int:
    print("Answer a few questions to assess your privacy footprint.\n")
    answers: dict[str, bool] = {}
    for q in footprint.QUESTIONS:
        ans = _ask_yes_no(q.text)
        if ans is not None:
            answers[q.key] = ans

    result = footprint.assess(answers)
    print()
    print(f"Privacy posture: {result.rating} ({result.percent}% – "
          f"{result.score}/{result.max_score} points)")
    if result.recommendations:
        print("\nTop recommendations:")
        for rec in result.recommendations:
            print(f"  * {rec}")
    else:
        print("\nNo gaps found in the questions you answered. Nice.")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    print(network_scan.AUTHORIZATION_NOTICE)
    print()
    try:
        result = network_scan.scan(
            target=args.target,
            ports=args.ports,
            allow_public=args.allow_public,
        )
    except network_scan.ScanError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(network_scan.render(result))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phantom_privacy",
        description="Personal privacy and self-audit toolkit.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_phone = sub.add_parser("phone", help="Phone-number privacy guidance.")
    p_phone.set_defaults(func=cmd_phone)

    p_voip = sub.add_parser("voip", help="Compare VoIP / masking-number services.")
    p_voip.set_defaults(func=cmd_voip)

    p_fp = sub.add_parser("footprint", help="Assess your digital-footprint exposure.")
    p_fp.set_defaults(func=cmd_footprint)

    p_scan = sub.add_parser("scan", help="Audit your own network with nmap.")
    p_scan.add_argument(
        "target",
        nargs="?",
        default="127.0.0.1",
        help="Host or CIDR to scan (default: 127.0.0.1). Must be private/loopback "
             "unless --allow-public is given.",
    )
    p_scan.add_argument("--ports", help="Port range, e.g. '1-1024'. Default: top 100.")
    p_scan.add_argument(
        "--allow-public",
        action="store_true",
        help="Permit scanning a non-private target (only for hosts you are "
             "authorised to test).",
    )
    p_scan.set_defaults(func=cmd_scan)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
