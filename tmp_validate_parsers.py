from __future__ import annotations

import json
from pathlib import Path

from leadership_agent.answering import LeadershipAgent, save_report
from leadership_agent.config import AppConfig
from leadership_agent.utils import slugify


QUESTIONS = [
    "What is our current revenue trend?",
    "Which departments are underperforming?",
    "What were the key risks highlighted in the last quarter?",
]

PARSERS = [
    ("docling", False),
    ("marker", False),
    ("native", True),
]


def run() -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    output_dir = Path("outputs") / "sample_answers"
    output_dir.mkdir(parents=True, exist_ok=True)

    for parser_name, allow_native in PARSERS:
        print(f"\n=== Running parser: {parser_name} ===", flush=True)
        config = AppConfig.load("config.yaml")
        config.parser.primary = parser_name
        config.parser.fallback = parser_name
        config.parser.allow_native_fallback = allow_native
        if not config.runtime.openai_api_key:
            config.runtime.llm_provider = "extractive"

        agent = LeadershipAgent(config)
        ingest_stats = agent.ingest()
        build_stats = agent.build_index()

        dense_dir = config.index_dir / "dense"
        faiss_files = sorted(path.name for path in dense_dir.glob("*.faiss"))

        parser_result: dict[str, object] = {
            "parser": parser_name,
            "allow_native_fallback": allow_native,
            "ingest": ingest_stats,
            "build_index": build_stats,
            "faiss_files": faiss_files,
            "answers": [],
        }

        answers: list[dict[str, object]] = []
        for question in QUESTIONS:
            report = agent.ask(question, output_dir=config.output_dir)
            filename = f"{parser_name}_{slugify(question)}.json"
            report_path = output_dir / filename
            save_report(report, report_path)
            answer_row = {
                "question": question,
                "report_path": str(report_path),
                "direct_answer": report.direct_answer,
                "caveats": report.caveats,
                "sources": [source.doc_name for source in report.sources],
                "insufficient_evidence": report.analytics.get("insufficient_evidence", False),
                "computed": report.analytics.get("computed", {}),
            }
            answers.append(answer_row)
            print(f"Question: {question}", flush=True)
            print(f"Answer: {report.direct_answer}", flush=True)
            print(f"Sources: {answer_row['sources']}", flush=True)
            print(f"Insufficient evidence: {answer_row['insufficient_evidence']}", flush=True)
            print("", flush=True)

        parser_result["answers"] = answers
        summary.append(parser_result)

    summary_path = output_dir / "parser_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved summary to {summary_path}", flush=True)
    return summary


if __name__ == "__main__":
    run()
