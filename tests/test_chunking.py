import json

from leadership_agent.config import AppConfig, ChunkingSettings
from leadership_agent.ingest import _assign_section_pages, build_chunks, split_sections
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


def test_docling_section_pages_inferred_from_parser_output(tmp_path) -> None:
    markdown = """
Revenue
Revenue increased from Q1 to Q2.

Risk Factors
Cybersecurity remains a key risk area.
""".strip()
    structured_path = tmp_path / "sample.json"
    structured_payload = {
        "parser_output": {
            "texts": [
                {"text": "Revenue", "prov": [{"page_no": 1}]},
                {"text": "Revenue increased from Q1 to Q2.", "prov": [{"page_no": 1}]},
                {"text": "Risk Factors", "prov": [{"page_no": 2}]},
                {"text": "Cybersecurity remains a key risk area.", "prov": [{"page_no": 2}]},
            ]
        }
    }
    structured_path.write_text(json.dumps(structured_payload), encoding="utf-8")

    document = DocumentRecord(
        doc_id="doc1",
        doc_name="sample.pdf",
        source_path="sample.pdf",
        parser_used="docling",
        markdown_path="sample.md",
        structured_path=str(structured_path),
        report_type="quarterly_report",
        inferred_period="Q2 FY2025",
        inferred_year=2025,
        inferred_quarter=2,
        page_count=2,
    )

    sections = split_sections(document, markdown)
    assert all(section.page_start is None for section in sections)

    _assign_section_pages(document, markdown, sections)

    assert sections[0].page_start == 1
    assert sections[-1].page_start == 2
    assert sections[-1].page_end == 2
