from leadership_agent.config import AppConfig, ChunkingSettings
from leadership_agent.ingest import build_chunks, split_sections
from leadership_agent.models import DocumentRecord


def test_heading_aware_sections_and_chunks() -> None:
    markdown = """
[PAGE 1]
Management Discussion
Revenue Trend
Revenue increased meaningfully over the quarter.

[PAGE 2]
Risk Factors
Cybersecurity remains a key risk area.
""".strip()
    document = DocumentRecord(
        doc_id="doc1",
        doc_name="sample.txt",
        source_path="sample.txt",
        parser_used="native",
        markdown_path="sample.md",
        structured_path="sample.json",
        report_type="quarterly_report",
        inferred_period="Q2 FY2025",
        inferred_year=2025,
        inferred_quarter=2,
    )
    sections = split_sections(document, markdown)
    assert len(sections) >= 2
    assert sections[0].heading in {"Management Discussion", "Revenue Trend"}
    assert sections[-1].section_type == "risk"

    config = AppConfig(chunking=ChunkingSettings(max_tokens=120, overlap_tokens=20))
    chunks = build_chunks(document, sections, config)
    assert chunks
    assert all(chunk.section_id for chunk in chunks)
    assert any(chunk.metadata["section_type"] == "risk" for chunk in chunks)
