"""JSON-in / JSON-out commands; validation does not need credentials."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import __version__
from .client import DecisionClient, DecisionError, validate_request


def _read_request(path: str) -> dict:
    try:
        text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise ValueError("Cannot read request file as UTF-8") from None
    try:
        data = json.loads(text)
    except (ValueError, RecursionError):
        raise ValueError("Request must be valid JSON") from None
    if not isinstance(data, dict) or set(data) - {"model", "state", "questions"}:
        raise ValueError("Request must be an object containing state, questions and optional model")
    if "model" in data and (not isinstance(data["model"], str) or not data["model"].strip()):
        raise ValueError("model must be a nonempty string")
    validate_request(state=data.get("state"), questions=data.get("questions"))
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jev-lab", description="Typed decision workflows with an explicit backend.")
    parser.add_argument("--version", action="version", version=f"jev-lab {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate a request offline; no key or network needed")
    validate.add_argument("request", help="JSON request file, or - for stdin")
    run = commands.add_parser("run", help="Send one request to the selected backend")
    run.add_argument("request", help="JSON request file, or - for stdin")
    run.add_argument("--backend", choices=("typesafe", "local"), required=True)
    run.add_argument("--model", help="Override the request/default model")
    run.add_argument("--base-url", help="Local server API base URL, including /v1")
    run.add_argument("--timeout", type=float, default=30, help="Socket timeout in seconds (default: 30)")
    run.add_argument("--output", type=Path, help="Write JSON to this file instead of stdout (replaces existing file)")
    args = parser.parse_args(argv)
    try:
        data = _read_request(args.request)
        if args.command == "validate":
            print(json.dumps({"valid": True, "questions": len(data["questions"]), "network_used": False}))
            return 0
        client = DecisionClient(args.backend, model=args.model or data.get("model"),
                                base_url=args.base_url, timeout=args.timeout)
        result = client.decide(state=data["state"], questions=data["questions"])
        output = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        else:
            sys.stdout.write(output)
        return 0
    except DecisionError as exc:
        print(f"jev-lab: {exc}", file=sys.stderr)
        return 3
    except ValueError as exc:
        print(f"jev-lab: {exc}", file=sys.stderr)
        return 2
    except OSError:
        print("jev-lab: cannot write output; check the destination directory and permissions", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
