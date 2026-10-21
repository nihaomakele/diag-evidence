"""Command-line entry points; errors never echo input or credential values."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from . import __version__
from .agent import DiagnosisAgent
from .demo import SCENARIOS, evaluate
from .haconfig.settings import Settings


def _reject_constant(value):
    raise ValueError("JSON contains non-finite number")


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        """Do not echo rejected arguments, which can contain credential values."""
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: invalid arguments; use --help for accepted options.\n")


def main(argv=None) -> int:
    parser = _SafeArgumentParser(prog="diag-evidence", description="Diagnostic evidence workflow research toolkit")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run synthetic offline contract scenarios")
    demo.add_argument("--scenario", choices=["all"] + [s.name for s in SCENARIOS], default="consistent")
    evaluation = commands.add_parser("evaluate", help="Run the full synthetic failure suite")
    run = commands.add_parser("run", help="Read input JSON and call configured external HTTP services")
    run.add_argument("input", type=Path)
    run.add_argument("--skip-health", action="store_true")
    for command in (demo, evaluation, run):
        command.add_argument("--output-dir", type=Path, default=Path("artifacts"))
        command.add_argument("--no-write", action="store_true", help="Disable all artifact and log writes")
    args = parser.parse_args(argv)
    try:
        if args.command in {"demo", "evaluate"}:
            names = None if args.command == "evaluate" or args.scenario == "all" else [args.scenario]
            result = evaluate(output_dir=args.output_dir, write=not args.no_write, names=names)
            exit_code = 0 if result["passed"] else 1
        else:
            # Standard-library parser accepts NaN/Infinity by default; disable it.
            payload = json.loads(args.input.read_text(encoding="utf-8"), parse_constant=_reject_constant)
            if not isinstance(payload, dict) or "signal" not in payload:
                raise ValueError("Input JSON must be an object containing signal")
            allowed = {"signal", "sample_info", "input_is_synthetic", "llm_prompt_text", "llm_expected"}
            if set(payload) - allowed:
                raise ValueError("Unknown input fields")
            for field in ("input_is_synthetic", "llm_expected"):
                if field in payload and type(payload[field]) is not bool:
                    raise ValueError("Boolean input field has invalid type")
            cfg = replace(Settings.from_env(), logs_dir=str(args.output_dir / "logs"),
                          reports_dir=str(args.output_dir / "reports"))
            result = DiagnosisAgent(config=cfg, persist_trace=not args.no_write).run(
                **payload, skip_services_check=args.skip_health)
            exit_code = {"COMPLETED": 0, "DEGRADED": 2, "ASKING_USER": 3, "FAILED": 4}[result["terminal_state"]]
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return exit_code
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"error": type(exc).__name__,
                          "message": "配置、输入或输出路径无效；请检查文档与字段格式。"}, ensure_ascii=False), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
