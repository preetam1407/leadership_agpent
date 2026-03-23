from pathlib import Path

from leadership_agent.answering import LeadershipAgent
from leadership_agent.config import AppConfig, AnswerSettings, ParserSettings, RuntimeSettings


def _make_config(tmp_path: Path, threshold: float = 0.2) -> AppConfig:
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
        answer=AnswerSettings(max_context_items=8, require_citations=True, insufficient_evidence_threshold=threshold),
    )


def test_answer_includes_sources_and_schema(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    (config.raw_data_dir / "report.txt").write_text(
        """
[PAGE 1]
Financial Overview
Description Q4FY24 Q1FY25 Q2FY25
Revenue 5.61 5.71 5.87
""".strip(),
        encoding="utf-8",
    )
    agent = LeadershipAgent(config)
    agent.ingest()
    agent.build_index()
    report = agent.ask("What is our current revenue trend?")
    payload = report.to_dict()
    assert payload["sources"]
    assert "markdown_report" in payload
    assert "## Sources" in report.markdown_report


def test_insufficient_evidence_path(tmp_path: Path) -> None:
    config = _make_config(tmp_path, threshold=0.95)
    (config.raw_data_dir / "report.txt").write_text(
        """
[PAGE 1]
General Overview
This document is brief and does not contain the requested fact.
""".strip(),
        encoding="utf-8",
    )
    agent = LeadershipAgent(config)
    agent.ingest()
    agent.build_index()
    report = agent.ask("What is the exact market share in Japan?")
    assert report.analytics["insufficient_evidence"] is True
    assert report.caveats


def test_underperforming_departments_answer(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    (config.raw_data_dir / "segments.txt").write_text(
        """
[PAGE 1]
Business Segments
Description Q4FY24 Q1FY25 Q2FY25
Digital Media 4.15 4.23 4.36
Digital Experience 1.40 1.41 1.45
""".strip(),
        encoding="utf-8",
    )
    agent = LeadershipAgent(config)
    agent.ingest()
    agent.build_index()
    report = agent.ask("Which departments are underperforming?")
    lowered = report.direct_answer.lower()
    assert "underperforming" in lowered
    assert "digital experience" in lowered


def test_underperforming_departments_ignores_immaterial_residual_row(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    (config.raw_data_dir / "segments.txt").write_text(
        """
[PAGE 1]
Business Segments
Description Q4FY24 Q1FY25 Q2FY25
Digital Media 4.15 4.23 4.36
Digital Experience 1.40 1.41 1.46
Publishing and Advertising 0.07 0.07 0.07
""".strip(),
        encoding="utf-8",
    )
    agent = LeadershipAgent(config)
    agent.ingest()
    agent.build_index()
    report = agent.ask("Which departments are underperforming?")
    lowered = report.direct_answer.lower()
    assert "digital experience" in lowered
    assert "publishing and advertising" not in lowered


def test_trend_answer_from_markdown_table(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    (config.raw_data_dir / "datasheet.md").write_text(
        """
[PAGE 1]
# Adobe Investor Relations Data Sheet

| Metric | Q1FY24 | Q2FY24 | Q3FY24 | Q4FY24 | Q1FY25 | Q2FY25 |
| --- | --- | --- | --- | --- | --- | --- |
| Total revenue | 5.18 | 5.31 | 5.41 | 5.61 | 5.71 | 5.87 |
| Digital Media | 3.90 | 4.02 | 4.11 | 4.27 | 4.31 | 4.36 |
""".strip(),
        encoding="utf-8",
    )

    agent = LeadershipAgent(config)
    agent.ingest()
    agent.build_index()
    report = agent.ask("What is our current revenue trend?")
    computed = report.analytics["computed"]
    assert computed["metric_series"]
    assert computed["trend_summary"] is not None
    assert "upward trend" in computed["trend_summary"].lower()
