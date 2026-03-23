from pathlib import Path

from leadership_agent.config import AppConfig, ParserSettings, RuntimeSettings
from leadership_agent.indexing import build_indexes
from leadership_agent.ingest import ingest_corpus
from leadership_agent.models import QuestionPlan
from leadership_agent.retrieval import HybridRetriever, QuestionClassifier


def _make_config(tmp_path: Path) -> AppConfig:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return AppConfig(
        raw_data_dir=raw_dir,
        parsed_data_dir=tmp_path / "data" / "parsed",
        index_dir=tmp_path / "data" / "indexes",
        output_dir=tmp_path / "outputs",
        eval_questions_path=tmp_path / "data" / "eval" / "questions.json",
        parser=ParserSettings(primary="native", fallback="native", allow_native_fallback=True),
        runtime=RuntimeSettings(llm_provider="extractive", openai_api_key=""),
    )


def test_table_first_retrieval_for_metric_question(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    (config.raw_data_dir / "metrics.txt").write_text(
        """
[PAGE 1]
Financial Overview
Description Q4FY24 Q1FY25 Q2FY25
Revenue 5.61 5.71 5.87
Digital Media 4.15 4.23 4.35
Digital Experience 1.40 1.41 1.46
""".strip(),
        encoding="utf-8",
    )
    (config.raw_data_dir / "risks.txt").write_text(
        """
[PAGE 1]
Risk Factors
Cybersecurity threats, privacy obligations, and platform resilience remain material risks.
""".strip(),
        encoding="utf-8",
    )

    ingest_corpus(config)
    build_indexes(config)
    retriever = HybridRetriever(config)
    plan, candidates, _ = retriever.retrieve("What is our current revenue trend?")
    assert plan.prioritize_tables is True
    assert candidates
    assert any(candidate.object_type == "table" for candidate in candidates[:2])


def test_high_confidence_underperforming_prompt_overrides_conflicting_llm_class(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    classifier = QuestionClassifier(config)
    heuristic = classifier._heuristic_plan("Which departments are underperforming?")
    conflicting = QuestionPlan(
        question=heuristic.question,
        question_class="risk_extraction",
        lexical_query="risk query",
        semantic_query="risk query",
        prioritize_tables=False,
        prioritize_latest=True,
        prioritize_report_types=["annual_report"],
        target_metrics=[],
        target_entities=[],
        time_focus=None,
        widen_if_sparse=True,
    )

    resolved = classifier._apply_high_confidence_overrides(heuristic.question, heuristic, conflicting)

    assert resolved.question_class == "comparison"
    assert resolved.prioritize_tables is True
