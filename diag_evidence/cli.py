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
from .reference_demo import evaluate_references
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
    references = commands.add_parser("demo-references", help="Run synthetic historical-reference applicability scenarios")
    run = commands.add_parser("run", help="Read input JSON and call configured external HTTP services")
    run.add_argument("input", type=Path)
    run.add_argument("--skip-health", action="store_true")
    benchmark = commands.add_parser("benchmark-hust", help="Real four-class HUST benchmark; requires [benchmark] extra")
    benchmark.add_argument("--cache-dir", type=Path, required=True)
    benchmark.add_argument("--output-dir", type=Path, required=True, help="New directory, must not exist")
    benchmark.add_argument("--download", action="store_true", help="Explicitly download 325 MB of CC BY 4.0 data")
    study = commands.add_parser("study-hust-health", help="Bounded validation-selected health study; requires [study]")
    study.add_argument("--cache-dir", type=Path, required=True)
    study.add_argument("--holdout-cache-dir", type=Path, required=True)
    study.add_argument("--output-dir", type=Path, required=True)
    study.add_argument("--download", action="store_true", help="Download pinned 6204 holdout after selection is frozen")
    grouped = commands.add_parser("study-hust-groups", help="Nested leave-type-out validation; requires [study]")
    grouped.add_argument("--cache-dir", type=Path, required=True)
    grouped.add_argument("--output-dir", type=Path, required=True)
    external = commands.add_parser("evaluate-ottawa", help="Frozen model transfer to independent Ottawa data")
    external.add_argument("--selection-dir", type=Path, required=True)
    external.add_argument("--cache-dir", type=Path, required=True)
    external.add_argument("--output-dir", type=Path, required=True)
    external.add_argument("--download", action="store_true", help="Download pinned CC BY 4.0 Ottawa profile A")
    order = commands.add_parser("study-envelope-order", help="Offline source-selected envelope/order experiment")
    order.add_argument("--cache-dir", type=Path, required=True)
    order.add_argument("--output-dir", type=Path, required=True)
    order.add_argument("--legacy-selection-dir", type=Path)
    order_eval = commands.add_parser("evaluate-envelope-order", help="Frozen envelope/order comparison on A or B")
    order_eval.add_argument("--selection-dir", type=Path, required=True)
    order_eval.add_argument("--cache-dir", type=Path, required=True)
    order_eval.add_argument("--output-dir", type=Path, required=True)
    order_eval.add_argument("--profile", choices=["A", "B"], required=True)
    order_eval.add_argument("--download", action="store_true")
    for command in (demo, evaluation, references, run):
        command.add_argument("--output-dir", type=Path, default=Path("artifacts"))
        command.add_argument("--no-write", action="store_true", help="Disable all artifact and log writes")
    args = parser.parse_args(argv)
    try:
        if args.command == "study-envelope-order":
            from .benchmarks.order_study import run_order_development
            summary = run_order_development(args.cache_dir, args.output_dir, args.legacy_selection_dir)
            result = {"selection_path": str(args.output_dir / "frozen_selection.json"),
                      "selected": summary["selected"], "quality_screen_passed": summary["quality_screen_passed"]}
            exit_code = 0
        elif args.command == "evaluate-envelope-order":
            from .benchmarks.order_study import evaluate_order_profile
            summary = evaluate_order_profile(args.selection_dir, args.cache_dir, args.output_dir, args.profile, download=args.download)
            result = {"metrics_path": str(args.output_dir / "metrics.json"), "profile": summary["profile"],
                      "outcomes": summary["outcomes"], "default_replaced": False}
            exit_code = 0
        elif args.command == "evaluate-ottawa":
            from .benchmarks.ottawa_external import run_external_ottawa
            summary = run_external_ottawa(args.selection_dir, args.cache_dir, args.output_dir, download=args.download)
            result = {"metrics_path": str(args.output_dir / "metrics.json"), "all_outputs": summary["all_outputs"],
                      "quality_guarded": summary["quality_guarded"], "default_replaced": False}
            exit_code = 0
        elif args.command == "study-hust-groups":
            from .benchmarks.grouped_validation import run_grouped_validation
            summary = run_grouped_validation(args.cache_dir, args.output_dir)
            result = {"metrics_path": str(args.output_dir / "metrics.json"),
                      "aggregate": summary["aggregate"], "external_model_candidate": summary["external_model_candidate"],
                      "default_baseline_replaced": False}
            exit_code = 0
        elif args.command == "study-hust-health":
            from .benchmarks.hust_health import run_health_study
            summary = run_health_study(args.cache_dir, args.holdout_cache_dir, args.output_dir, download=args.download)
            result = {"metrics_path": str(args.output_dir / "metrics.json"), "selected": summary["selected"],
                      "results": summary["results"], "default_baseline_replaced": False}
            exit_code = 0
        elif args.command == "benchmark-hust":
            from .benchmarks.hust import run_benchmark
            summary = run_benchmark(args.cache_dir, args.output_dir, download=args.download)
            result = {"metrics_path": str(args.output_dir / "metrics.json"), "scope": summary["scope"],
                      "test_classifier": summary["metrics"]["test"]["classifier"]}
            exit_code = 0
        elif args.command == "demo-references":
            result = evaluate_references(output_dir=args.output_dir, write=not args.no_write)
            exit_code = 0 if result["passed"] else 1
        elif args.command in {"demo", "evaluate"}:
            names = None if args.command == "evaluate" or args.scenario == "all" else [args.scenario]
            result = evaluate(output_dir=args.output_dir, write=not args.no_write, names=names)
            exit_code = 0 if result["passed"] else 1
        else:
            # Standard-library parser accepts NaN/Infinity by default; disable it.
            payload = json.loads(args.input.read_text(encoding="utf-8"), parse_constant=_reject_constant)
            if not isinstance(payload, dict) or "signal" not in payload:
                raise ValueError("Input JSON must be an object containing signal")
            allowed = {"signal", "sample_info", "input_is_synthetic", "llm_prompt_text", "llm_expected",
                       "evidence_context"}
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
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
        print(json.dumps({"error": type(exc).__name__,
                          "message": "配置、输入、可选依赖或输出路径无效；请按文档检查。"}, ensure_ascii=False), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
