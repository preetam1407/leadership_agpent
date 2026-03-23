from pathlib import Path

from leadership_agent.config import AppConfig, ParserSettings, RuntimeSettings
from leadership_agent.ingest import ingest_corpus, load_parsed_artifacts


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


def test_markdown_table_normalization_populates_rows(tmp_path: Path) -> None:
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

    ingest_corpus(config)
    _, _, _, tables = load_parsed_artifacts(config)
    assert tables

    normalized_tables = [table for table in tables if table.metadata.get("table_source") == "markdown_table_normalized"]
    assert normalized_tables
    table = normalized_tables[0]
    assert table.normalized_rows
    revenue_rows = [row for row in table.normalized_rows if "revenue" in str(row.get("label", "")).lower()]
    assert revenue_rows
    revenue = revenue_rows[0]
    assert revenue["values"]["Q1FY24"] == "5.18"
    assert revenue["values"]["Q2FY25"] == "5.87"
    assert "Total revenue" in table.table_text_summary
