from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from leadership_agent.answering import LeadershipAgent
from leadership_agent.config import AppConfig
from leadership_agent.utils import ensure_dir, write_json


def load_eval_questions(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Evaluation questions file must contain a JSON list.")
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(payload, start=1):
        if not isinstance(row, dict) or not row.get("question"):
            raise ValueError(f"Invalid eval row at index {idx}")
        rows.append(row)
    return rows


def run_evaluation(config: AppConfig, questions_path: str | Path | None = None, results_path: str | Path | None = None) -> dict[str, Any]:
    agent = LeadershipAgent(config)
    questions = load_eval_questions(questions_path or config.eval_questions_path)
    results: list[dict[str, Any]] = []

    for case in questions:
        started = time.perf_counter()
        report = agent.ask(case["question"], output_dir=config.output_dir)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        source_docs = [source.doc_name for source in report.sources]
        expected_docs = case.get("expected_docs", [])
        hit = 0.0
        if expected_docs:
            hit = sum(1 for item in expected_docs if any(item.lower() in doc.lower() for doc in source_docs)) / len(expected_docs)
        grounded = bool(report.sources) and not report.analytics.get("insufficient_evidence", False)
        results.append(
            {
                "question": case["question"],
                "question_class": report.question_class,
                "direct_answer": report.direct_answer,
                "source_docs": source_docs,
                "retrieval_hit": round(hit, 3),
                "citation_count": len(report.sources),
                "grounded": grounded,
                "latency_ms": latency_ms,
                "answer_chars": len(report.direct_answer),
            }
        )

    retrieval_hit_rate = statistics.mean(item["retrieval_hit"] for item in results) if results else 0.0
    citation_coverage = statistics.mean(1.0 if item["citation_count"] > 0 else 0.0 for item in results) if results else 0.0
    groundedness_pass_rate = statistics.mean(1.0 if item["grounded"] else 0.0 for item in results) if results else 0.0
    answer_lengths = [item["answer_chars"] for item in results]
    latencies = [item["latency_ms"] for item in results]
    summary = {
        "cases": len(results),
        "retrieval_hit_rate": round(retrieval_hit_rate, 3),
        "citation_coverage": round(citation_coverage, 3),
        "groundedness_pass_rate": round(groundedness_pass_rate, 3),
        "answer_length_stats": {
            "mean_chars": round(statistics.mean(answer_lengths), 1) if answer_lengths else 0.0,
            "max_chars": max(answer_lengths) if answer_lengths else 0,
            "min_chars": min(answer_lengths) if answer_lengths else 0,
        },
        "latency_stats_ms": {
            "mean": round(statistics.mean(latencies), 2) if latencies else 0.0,
            "p95": round(_percentile(latencies, 95), 2) if latencies else 0.0,
            "max": max(latencies) if latencies else 0.0,
        },
    }
    payload = {"summary": summary, "results": results}
    output_path = Path(results_path or (config.eval_questions_path.parent / "results.json"))
    write_json(output_path, payload)
    return payload


def generate_eval_plots(results_path: str | Path, output_dir: str | Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    payload = json.loads(Path(results_path).read_text(encoding="utf-8"))
    results = payload.get("results", [])
    if not results:
        return []

    ensure_dir(output_dir)
    created: list[str] = []

    type_counts: dict[str, int] = {}
    for row in results:
        question_class = row.get("question_class", "unknown")
        type_counts[question_class] = type_counts.get(question_class, 0) + 1
    path = Path(output_dir) / "question_type_distribution.png"
    plt.figure(figsize=(8, 4))
    plt.bar(type_counts.keys(), type_counts.values())
    plt.xticks(rotation=20, ha="right")
    plt.title("Question Type Distribution")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    created.append(str(path))

    latencies = [row.get("latency_ms", 0.0) for row in results]
    path = Path(output_dir) / "latency_distribution.png"
    plt.figure(figsize=(8, 4))
    plt.hist(latencies, bins=min(10, max(3, len(latencies))), edgecolor="black")
    plt.title("Latency Distribution (ms)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    created.append(str(path))

    retrieval_hits = [row.get("retrieval_hit", 0.0) for row in results]
    path = Path(output_dir) / "retrieval_hit_distribution.png"
    plt.figure(figsize=(8, 4))
    plt.hist(retrieval_hits, bins=min(10, max(3, len(retrieval_hits))), edgecolor="black")
    plt.title("Retrieval Hit Distribution")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    created.append(str(path))

    return created


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
