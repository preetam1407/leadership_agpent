from __future__ import annotations

import argparse
import json
from pathlib import Path

from leadership_agent.answering import LeadershipAgent, render_report, save_report
from leadership_agent.config import AppConfig
from leadership_agent.evaluate import generate_eval_plots, run_evaluation
from leadership_agent.utils import slugify


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lean leadership document-intelligence agent")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Parse raw documents into structured artifacts")
    ingest.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    ingest.add_argument("--input_dir", default=None, help="Override raw input directory")

    build = sub.add_parser("build-index", help="Build BM25, dense, and metadata indexes")
    build.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    build.add_argument("--ingest-first", action="store_true", help="Run ingest before building indexes")
    build.add_argument("--input_dir", default=None, help="Override raw input directory when --ingest-first is used")

    ask = sub.add_parser("ask", help="Answer one leadership question")
    ask.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    ask.add_argument("--question", required=True, help="Natural-language leadership question")
    ask.add_argument(
        "--report_path",
        default=None,
        help="Optional report filename/path. Output is always written under <output_dir>/sample_answers.",
    )
    ask.add_argument("--output_dir", default=None, help="Where plots/sample artifacts should be written")

    evaluate = sub.add_parser("eval", help="Run the eval set end to end")
    evaluate.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    evaluate.add_argument("--questions", default=None, help="Override eval questions JSON path")
    evaluate.add_argument("--results_path", default=None, help="Where to write eval results JSON")

    plots = sub.add_parser("plot", help="Generate plots from eval results")
    plots.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    plots.add_argument("--results_path", default=None, help="Eval results JSON path")
    plots.add_argument("--output_dir", default="outputs/sample_plots", help="Directory for plot PNG files")

    return parser


def _resolve_report_path(config: AppConfig, question: str, requested_path: str | None) -> Path:
    answers_dir = config.output_dir / "sample_answers"
    if requested_path:
        filename = Path(requested_path).name
    else:
        filename = f"{slugify(question) or 'latest_answer'}.json"
    if not filename:
        filename = "latest_answer.json"
    if not filename.lower().endswith(".json"):
        filename = f"{filename}.json"
    return answers_dir / filename


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = AppConfig.load(args.config)
    agent = LeadershipAgent(config)

    if args.command == "ingest":
        stats = agent.ingest(input_dir=args.input_dir)
        print(json.dumps(stats, indent=2))
        return

    if args.command == "build-index":
        if args.ingest_first:
            ingest_stats = agent.ingest(input_dir=args.input_dir)
            print(json.dumps({"ingest": ingest_stats}, indent=2))
        stats = agent.build_index()
        print(json.dumps(stats, indent=2))
        return

    if args.command == "ask":
        report_path = _resolve_report_path(config, args.question, args.report_path)
        report = agent.ask(args.question, output_dir=args.output_dir)
        save_report(report, report_path)
        print(render_report(report))
        print(f"\nJSON saved to {report_path}")
        return

    if args.command == "eval":
        payload = run_evaluation(config, questions_path=args.questions, results_path=args.results_path)
        print(json.dumps(payload["summary"], indent=2))
        return

    if args.command == "plot":
        results_path = args.results_path or (config.eval_questions_path.parent / "results.json")
        created = generate_eval_plots(results_path, args.output_dir)
        print(json.dumps({"plots": created}, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
