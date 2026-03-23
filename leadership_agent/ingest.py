from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from leadership_agent.config import AppConfig
from leadership_agent.models import (
    ChunkRecord,
    DocumentRecord,
    ParsedDocumentArtifact,
    ParserQuality,
    SectionRecord,
    TableRecord,
)
from leadership_agent.utils import (
    LOGGER,
    chunk_paragraphs,
    ensure_dir,
    normalize_whitespace,
    stable_id,
    token_count,
    write_json,
    write_jsonl,
)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
VALUE_RE = re.compile(r"\(?-?\$?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?%?\)?")
ITEM_HEADING_RE = re.compile(r"^Item\s+\d+[A-Z]?\.\s+.+", flags=re.IGNORECASE)
MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
PERIOD_RE = re.compile(r"Q[1-4]\s*FY\d{2,4}|FY\d{2,4}\s*YTD|FY\d{2,4}", flags=re.IGNORECASE)
PAGE_MARKER_RE = re.compile(r"^\[PAGE\s+(\d+)\]$")
MARKDOWN_TABLE_SEPARATOR_CELL_RE = re.compile(r"^:?-{2,}:?$")


def _safe_page_count(*values: Any) -> int:
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            if value >= 0:
                return int(value)
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                return int(stripped)
            continue
        if isinstance(value, list):
            return len(value)
    return 0


class BaseParser:
    name = "base"

    def parse(self, path: Path) -> tuple[str, dict[str, Any], int]:
        raise NotImplementedError


class DoclingParser(BaseParser):
    name = "docling"

    def parse(self, path: Path) -> tuple[str, dict[str, Any], int]:
        try:
            from docling.document_converter import DocumentConverter
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Docling is not installed.") from exc

        converter = DocumentConverter()
        result = converter.convert(str(path))
        markdown = normalize_whitespace(result.document.export_to_markdown())
        structured: dict[str, Any] = {}
        if hasattr(result.document, "export_to_dict"):
            try:
                structured = result.document.export_to_dict()
            except Exception:
                structured = {}
        page_count = _safe_page_count(
            getattr(result, "pages", None),
            structured.get("page_count"),
            structured.get("pages"),
            structured.get("children"),
        )
        return markdown, structured, page_count


class MarkerParser(BaseParser):
    name = "marker"

    def parse(self, path: Path) -> tuple[str, dict[str, Any], int]:
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Marker is not installed.") from exc

        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(str(path))
        markdown, _, _ = text_from_rendered(rendered)
        structured: dict[str, Any] = {}
        if hasattr(rendered, "model_dump"):
            try:
                structured = rendered.model_dump()
            except Exception:
                structured = {}
        page_count = len(structured.get("children", [])) if isinstance(structured.get("children"), list) else 0
        return normalize_whitespace(markdown), structured, page_count


class NativeParser(BaseParser):
    name = "native"

    def parse(self, path: Path) -> tuple[str, dict[str, Any], int]:
        ext = path.suffix.lower()
        if ext in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            return normalize_whitespace(text), {}, 1
        if ext == ".docx":
            try:
                from docx import Document
            except Exception as exc:  # pragma: no cover - dependency issue
                raise RuntimeError("python-docx is required for DOCX parsing.") from exc
            document = Document(str(path))
            text = "\n".join(paragraph.text for paragraph in document.paragraphs)
            return normalize_whitespace(text), {}, 1
        if ext == ".pdf":
            try:
                from pypdf import PdfReader
            except Exception as exc:  # pragma: no cover - dependency issue
                raise RuntimeError("pypdf is required for PDF parsing.") from exc
            reader = PdfReader(str(path))
            pages: list[str] = []
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = normalize_whitespace(page.extract_text() or "")
                pages.append(f"[PAGE {page_number}]\n{page_text}")
            return "\n\n".join(pages).strip(), {"page_count": len(reader.pages)}, len(reader.pages)
        raise ValueError(f"Unsupported extension: {ext}")


PARSER_REGISTRY = {
    "docling": DoclingParser,
    "marker": MarkerParser,
    "native": NativeParser,
}


def list_input_files(input_dir: str | Path) -> list[Path]:
    base = Path(input_dir)
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {base}")
    paths = sorted(path for path in base.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not paths:
        raise ValueError(f"No supported input files found in {base}")
    return paths


def assess_markdown_quality(markdown: str, file_size_bytes: int) -> ParserQuality:
    lines = [line.rstrip() for line in markdown.splitlines()]
    nonempty = [line for line in lines if line.strip()]
    heading_count = sum(1 for line in nonempty if detect_heading(line) is not None)
    broken_lines = sum(
        1
        for current, nxt in zip(nonempty, nonempty[1:])
        if _looks_like_broken_line(current, nxt)
    )
    broken_line_ratio = broken_lines / max(1, len(nonempty))
    empty_line_ratio = (len(lines) - len(nonempty)) / max(1, len(lines))

    issues: list[str] = []
    score = 1.0
    minimum_length = max(500, int(file_size_bytes * 0.01))
    if len(markdown) < minimum_length:
        score -= 0.45
        issues.append("markdown too short relative to file size")
    if broken_line_ratio > 0.35:
        score -= 0.20
        issues.append("too many broken lines")
    if empty_line_ratio > 0.55:
        score -= 0.10
        issues.append("too many empty lines")
    if heading_count == 0:
        score -= 0.15
        issues.append("missing heading structure")
    if re.search(r"[A-Za-z]", markdown) is None:
        score -= 0.50
        issues.append("unreadable extraction")

    return ParserQuality(
        score=max(0.0, round(score, 3)),
        issues=issues,
        heading_count=heading_count,
        broken_line_ratio=round(broken_line_ratio, 3),
        empty_line_ratio=round(empty_line_ratio, 3),
    )


def clean_markdown(markdown: str) -> str:
    markdown = normalize_whitespace(markdown)
    pages = _split_pages(markdown)
    repeated_lines = _detect_repeated_headers_and_footers(pages)

    cleaned_pages: list[str] = []
    for page_number, lines in pages:
        filtered = [line for line in lines if line.strip() not in repeated_lines]
        rejoined = _rejoin_broken_lines(filtered)
        body = "\n".join(rejoined).strip()
        if page_number is not None:
            cleaned_pages.append(f"[PAGE {page_number}]\n{body}".strip())
        else:
            cleaned_pages.append(body)

    return "\n\n".join(part for part in cleaned_pages if part).strip()


def parse_document(path: Path, config: AppConfig) -> ParsedDocumentArtifact:
    file_size = path.stat().st_size
    last_error: Exception | None = None
    selected_markdown = ""
    selected_structured: Any = {}
    selected_page_count = 0
    selected_parser = "native"
    parser_attempts: list[dict[str, Any]] = []

    for parser_name in config.parser_chain():
        parser_cls = PARSER_REGISTRY.get(parser_name)
        if parser_cls is None:
            continue
        parser = parser_cls()
        try:
            markdown, structured, page_count = parser.parse(path)
        except Exception as exc:
            last_error = exc
            parser_attempts.append({"parser": parser_name, "status": "error", "error": str(exc)})
            LOGGER.warning("Parser %s failed for %s: %s", parser_name, path.name, exc)
            continue

        quality = assess_markdown_quality(markdown, file_size)
        selected_markdown = markdown
        selected_structured = structured
        selected_page_count = page_count
        selected_parser = parser_name
        parser_attempts.append(
            {
                "parser": parser_name,
                "status": "ok",
                "quality": quality.to_dict(),
                "page_count": page_count,
            }
        )

        if quality.score >= config.parser.quality_min_score or parser_name == config.parser_chain()[-1]:
            break
        LOGGER.info(
            "Parser %s produced low quality score %.3f for %s, trying next parser",
            parser_name,
            quality.score,
            path.name,
        )

    if not selected_markdown:
        raise RuntimeError(f"Unable to parse {path.name}: {last_error}")

    cleaned_markdown = clean_markdown(selected_markdown)
    quality = assess_markdown_quality(cleaned_markdown, file_size)

    doc_id = stable_id(path.resolve())
    markdown_path = config.parsed_data_dir / "markdown" / f"{doc_id}.md"
    structured_path = config.parsed_data_dir / "documents" / f"{doc_id}.json"
    if config.parser.persist_markdown:
        ensure_dir(markdown_path.parent)
        markdown_path.write_text(cleaned_markdown, encoding="utf-8")

    artifact = ParsedDocumentArtifact(
        doc_id=doc_id,
        doc_name=path.name,
        source_path=str(path),
        parser_used=selected_parser,
        markdown_path=str(markdown_path),
        structured_path=str(structured_path),
        file_size_bytes=file_size,
        page_count=selected_page_count,
        quality=quality,
        raw_markdown=cleaned_markdown,
    )
    structured_payload = {
        "artifact": artifact.to_dict(),
        "parser_attempts": parser_attempts,
        "parser_output": _to_json_safe(selected_structured),
    }
    write_json(structured_path, structured_payload)
    return artifact


def infer_document_metadata(path: Path, artifact: ParsedDocumentArtifact) -> DocumentRecord:
    filename = path.stem.lower()
    text = artifact.raw_markdown[:8000].lower()

    report_type = "other"
    if any(token in filename for token in ["annual_report", "10-k", "annual"]):
        report_type = "annual_report"
    elif any(token in filename for token in ["earnings_release", "quarterly", "10-q"]):
        report_type = "quarterly_report"
    elif "datasheet" in filename:
        report_type = "investor_datasheet"

    quarter = None
    year = None
    file_quarter = re.search(r"q([1-4])[_ -]?fy[_ -]?(\d{2,4})", filename)
    if file_quarter:
        quarter = int(file_quarter.group(1))
        year = _normalize_year(file_quarter.group(2))
    if quarter is None:
        word_quarter = re.search(r"(first|second|third|fourth) quarter fiscal year (\d{4})", text)
        if word_quarter:
            quarter = {"first": 1, "second": 2, "third": 3, "fourth": 4}[word_quarter.group(1)]
            year = int(word_quarter.group(2))
    if year is None:
        year_match = re.search(r"\b(20\d{2})\b", filename + "\n" + text)
        if year_match:
            year = int(year_match.group(1))

    inferred_period = None
    if quarter and year:
        inferred_period = f"Q{quarter} FY{year}"
    elif year:
        inferred_period = f"FY{year}"

    business_areas = []
    for keyword in [
        "digital media",
        "digital experience",
        "acrobat",
        "creative cloud",
        "firefly",
        "ai",
        "cybersecurity",
        "document cloud",
    ]:
        if keyword in text and keyword not in business_areas:
            business_areas.append(keyword)

    latest_sort_key = (year or 0) * 10 + (quarter or 0)
    return DocumentRecord(
        doc_id=artifact.doc_id,
        doc_name=artifact.doc_name,
        source_path=artifact.source_path,
        parser_used=artifact.parser_used,
        markdown_path=artifact.markdown_path,
        structured_path=artifact.structured_path,
        report_type=report_type,
        inferred_period=inferred_period,
        inferred_year=year,
        inferred_quarter=quarter,
        likely_business_areas=business_areas,
        latest_sort_key=latest_sort_key,
        page_count=artifact.page_count,
        parser_quality_score=artifact.quality.score,
        metadata={"parser_quality": artifact.quality.to_dict()},
    )


def detect_heading(line: str) -> tuple[int, str] | None:
    stripped = line.strip()
    if not stripped or PAGE_MARKER_RE.match(stripped):
        return None

    markdown_match = MARKDOWN_HEADING_RE.match(stripped)
    if markdown_match:
        level = len(markdown_match.group(1))
        return level, markdown_match.group(2).strip()

    if ITEM_HEADING_RE.match(stripped):
        return 1, stripped.rstrip(".")

    if _looks_like_heading(stripped):
        return 2, stripped.rstrip(":")

    return None


def split_sections(document: DocumentRecord, markdown: str) -> list[SectionRecord]:
    lines = markdown.splitlines()
    sections: list[SectionRecord] = []
    current_heading = "Document Overview"
    current_heading_path = [current_heading]
    current_lines: list[str] = []
    current_page = 1 if "[PAGE " in markdown else None
    current_page_start = current_page
    stack: list[tuple[int, str]] = []
    previous_was_heading = False

    def flush(page_end: int | None) -> None:
        nonlocal current_lines
        text_lines = [line for line in current_lines if line.strip() and not PAGE_MARKER_RE.match(line.strip())]
        text = "\n".join(text_lines).strip()
        if not text:
            current_lines = []
            return
        section_id = stable_id(document.doc_id, current_heading, page_end, len(sections))
        sections.append(
            SectionRecord(
                section_id=section_id,
                doc_id=document.doc_id,
                doc_name=document.doc_name,
                heading=current_heading,
                heading_path=list(current_heading_path),
                page_start=current_page_start,
                page_end=page_end,
                section_type=infer_section_type(current_heading_path, text),
                text=text,
                metadata={"report_type": document.report_type},
            )
        )
        current_lines = []

    for raw_line in lines:
        line = raw_line.rstrip()
        page_match = PAGE_MARKER_RE.match(line.strip())
        if page_match:
            current_page = int(page_match.group(1))
            if current_page_start is None:
                current_page_start = current_page
            continue

        heading = detect_heading(line)
        if heading is not None:
            flush(current_page)
            level, heading_text = heading
            if previous_was_heading and stack:
                level = stack[-1][0] + 1
            stack = [item for item in stack if item[0] < level]
            stack.append((level, heading_text))
            current_heading = heading_text
            current_heading_path = [item[1] for item in stack]
            current_page_start = current_page
            previous_was_heading = True
            continue

        current_lines.append(line)
        if line.strip():
            previous_was_heading = False

    flush(current_page)
    if not sections:
        sections.append(
            SectionRecord(
                section_id=stable_id(document.doc_id, "overview"),
                doc_id=document.doc_id,
                doc_name=document.doc_name,
                heading="Document Overview",
                heading_path=["Document Overview"],
                page_start=1 if document.page_count else None,
                page_end=document.page_count or None,
                section_type=infer_section_type(["Document Overview"], markdown),
                text=markdown,
                metadata={"report_type": document.report_type},
            )
        )
    return sections


def infer_section_type(heading_path: list[str], text: str) -> str:
    joined = " ".join(heading_path).lower() + "\n" + text[:1200].lower()
    if any(_contains_term(joined, token) for token in ["risk", "cybersecurity", "forward-looking"]):
        return "risk"
    if any(_contains_term(joined, token) for token in ["revenue", "margin", "income", "arr", "earnings", "financial"]):
        return "financials"
    if any(_contains_term(joined, token) for token in ["operations", "employee", "capacity", "supply", "productivity"]):
        return "operations"
    if any(_contains_term(joined, token) for token in ["strategy", "innovation", "ai", "firefly", "acrobat ai assistant"]):
        return "strategy"
    return "other"


def build_chunks(document: DocumentRecord, sections: list[SectionRecord], config: AppConfig) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for section in sections:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", section.text) if part.strip()]
        if not paragraphs:
            continue
        chunk_texts = chunk_paragraphs(
            paragraphs,
            max_tokens=config.chunking.max_tokens,
            overlap_tokens=config.chunking.overlap_tokens,
        )
        for chunk_index, chunk_text in enumerate(chunk_texts):
            chunk_type = infer_chunk_type(chunk_text, section.section_type)
            chunk_id = stable_id(section.section_id, chunk_index)
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    section_id=section.section_id,
                    doc_id=document.doc_id,
                    doc_name=document.doc_name,
                    heading_path=list(section.heading_path),
                    page_start=section.page_start,
                    page_end=section.page_end,
                    chunk_type=chunk_type,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    metadata={"section_type": section.section_type, "report_type": document.report_type},
                )
            )
    return chunks


def infer_chunk_type(text: str, section_type: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullet_ratio = sum(1 for line in lines if line.startswith(("-", "*", "•"))) / max(1, len(lines))
    if bullet_ratio > 0.4:
        return "bullets"
    if section_type == "financials" and sum(ch.isdigit() for ch in text) > 30:
        return "table_context"
    if token_count(text) <= 150:
        return "summary"
    return "paragraph"


def extract_tables(document: DocumentRecord, sections: list[SectionRecord]) -> list[TableRecord]:
    tables: list[TableRecord] = []
    for section in sections:
        tables.extend(_extract_numeric_tables_from_section(document, section, len(tables)))
        tables.extend(_extract_markdown_tables_from_section(document, section, len(tables)))
    return tables


def _extract_numeric_tables_from_section(document: DocumentRecord, section: SectionRecord, offset: int) -> list[TableRecord]:
    lines = section.text.splitlines()
    current_page = section.page_start
    header_tokens: list[str] = []
    row_buffer: list[tuple[int | None, str]] = []
    title_hint = section.heading
    extracted: list[TableRecord] = []

    def flush() -> None:
        nonlocal header_tokens, row_buffer
        if len(header_tokens) < 3 or len(row_buffer) < 2:
            header_tokens = []
            row_buffer = []
            return

        normalized_rows: list[dict[str, Any]] = []
        markdown_lines = ["| Label | " + " | ".join(header_tokens) + " |", "| --- | " + " | ".join(["---"] * len(header_tokens)) + " |"]
        metric_tags: set[str] = set()
        page_values = [page for page, _ in row_buffer if page is not None]
        for page_number, row_line in row_buffer:
            parsed = _parse_numeric_row(row_line, header_tokens)
            if parsed is None:
                continue
            label, values = parsed
            normalized_rows.append({"label": label, "values": values, "page": page_number})
            markdown_lines.append("| " + label + " | " + " | ".join(values.get(token, "") for token in header_tokens) + " |")
            metric_tags.update(_infer_metric_tags(label))

        if len(normalized_rows) < 2:
            header_tokens = []
            row_buffer = []
            return

        summary_rows = "; ".join(
            f"{row['label']}: " + ", ".join(f"{period}={value}" for period, value in list(row['values'].items())[:3])
            for row in normalized_rows[:3]
        )
        table_id = stable_id(document.doc_id, section.section_id, title_hint, offset + len(extracted))
        extracted.append(
            TableRecord(
                table_id=table_id,
                doc_id=document.doc_id,
                section_id=section.section_id,
                doc_name=document.doc_name,
                page_start=min(page_values) if page_values else section.page_start,
                page_end=max(page_values) if page_values else section.page_end,
                title=title_hint,
                heading_path=list(section.heading_path),
                table_markdown="\n".join(markdown_lines),
                table_text_summary=f"{title_hint}: {summary_rows}",
                normalized_rows=normalized_rows,
                metric_tags=sorted(metric_tags),
                metadata={"table_source": "numeric_block"},
            )
        )
        header_tokens = []
        row_buffer = []

    for raw_line in lines:
        stripped = raw_line.strip()
        page_match = PAGE_MARKER_RE.match(stripped)
        if page_match:
            current_page = int(page_match.group(1))
            continue
        if not stripped:
            flush()
            continue

        periods = _extract_period_headers(stripped)
        if len(periods) >= 2:
            flush()
            header_tokens = periods
            previous_title = _previous_title_hint(lines, raw_line)
            title_hint = previous_title or section.heading
            continue

        if header_tokens:
            if _numeric_token_count(stripped) >= max(3, min(len(header_tokens), 4)):
                row_buffer.append((current_page, stripped))
                continue
            if _looks_like_heading(stripped) and not row_buffer:
                title_hint = stripped
                continue
            flush()

    flush()
    return extracted


def _extract_markdown_tables_from_section(document: DocumentRecord, section: SectionRecord, offset: int) -> list[TableRecord]:
    lines = section.text.splitlines()
    tables: list[TableRecord] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if len(current) < 2:
            current = []
            return
        title = section.heading
        normalized_rows, row_metric_tags, summary = _normalize_markdown_table(current, page_hint=section.page_start)
        metric_tags = sorted(set(_infer_metric_tags(" ".join(current[:3]))) | set(row_metric_tags))
        table_id = stable_id(document.doc_id, section.section_id, title, offset + len(tables))
        table_source = "markdown_table_normalized" if normalized_rows else "markdown_table"
        summary_text = f"{title}: {summary}" if summary else f"{title}: markdown table extracted from section."
        tables.append(
            TableRecord(
                table_id=table_id,
                doc_id=document.doc_id,
                section_id=section.section_id,
                doc_name=document.doc_name,
                page_start=section.page_start,
                page_end=section.page_end,
                title=title,
                heading_path=list(section.heading_path),
                table_markdown="\n".join(current),
                table_text_summary=summary_text,
                normalized_rows=normalized_rows,
                metric_tags=metric_tags,
                metadata={"table_source": table_source, "normalized_row_count": len(normalized_rows)},
            )
        )
        current = []

    for line in lines:
        stripped = line.strip()
        if _is_markdown_table_line(stripped):
            current.append(stripped)
        else:
            flush()
    flush()
    return tables


def ingest_corpus(config: AppConfig, input_dir: str | Path | None = None) -> dict[str, Any]:
    source_dir = Path(input_dir) if input_dir else config.raw_data_dir
    ensure_dir(config.parsed_data_dir / "markdown")
    ensure_dir(config.parsed_data_dir / "documents")
    ensure_dir(config.parsed_data_dir / "sections")
    ensure_dir(config.parsed_data_dir / "chunks")
    ensure_dir(config.parsed_data_dir / "tables")
    _reset_parsed_artifacts(config.parsed_data_dir)

    documents: list[DocumentRecord] = []
    sections: list[SectionRecord] = []
    chunks: list[ChunkRecord] = []
    tables: list[TableRecord] = []

    for path in list_input_files(source_dir):
        artifact = parse_document(path, config)
        document = infer_document_metadata(path, artifact)
        doc_sections = split_sections(document, artifact.raw_markdown)
        doc_chunks = build_chunks(document, doc_sections, config)
        doc_tables = extract_tables(document, doc_sections)

        documents.append(document)
        sections.extend(doc_sections)
        chunks.extend(doc_chunks)
        tables.extend(doc_tables)

        structured_path = Path(document.structured_path)
        structured_payload = json.loads(structured_path.read_text(encoding="utf-8"))
        structured_payload["document"] = document.to_dict()
        structured_payload["section_count"] = len(doc_sections)
        structured_payload["chunk_count"] = len(doc_chunks)
        structured_payload["table_count"] = len(doc_tables)
        structured_path.write_text(json.dumps(structured_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    write_jsonl(config.parsed_data_dir / "sections" / "sections.jsonl", [row.to_dict() for row in sections])
    write_jsonl(config.parsed_data_dir / "chunks" / "chunks.jsonl", [row.to_dict() for row in chunks])
    write_jsonl(config.parsed_data_dir / "tables" / "tables.jsonl", [row.to_dict() for row in tables])

    return {
        "documents": len(documents),
        "sections": len(sections),
        "chunks": len(chunks),
        "tables": len(tables),
        "source_dir": str(source_dir),
        "parser_chain": config.parser_chain(),
    }


def load_parsed_artifacts(config: AppConfig) -> tuple[list[DocumentRecord], list[SectionRecord], list[ChunkRecord], list[TableRecord]]:
    document_rows: list[DocumentRecord] = []
    for path in sorted((config.parsed_data_dir / "documents").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        doc = payload.get("document")
        if doc:
            document_rows.append(DocumentRecord(**doc))

    sections = [SectionRecord(**row) for row in _read_jsonl_required(config.parsed_data_dir / "sections" / "sections.jsonl")]
    chunks = [ChunkRecord(**row) for row in _read_jsonl_required(config.parsed_data_dir / "chunks" / "chunks.jsonl")]
    tables = [TableRecord(**row) for row in _read_jsonl_required(config.parsed_data_dir / "tables" / "tables.jsonl")]
    return document_rows, sections, chunks, tables


def _read_jsonl_required(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing parsed artifact: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _reset_parsed_artifacts(parsed_data_dir: Path) -> None:
    for directory_name in ["markdown", "documents"]:
        directory = parsed_data_dir / directory_name
        if not directory.exists():
            continue
        for path in directory.glob("*"):
            if path.is_file():
                path.unlink()

    for artifact in [
        parsed_data_dir / "sections" / "sections.jsonl",
        parsed_data_dir / "chunks" / "chunks.jsonl",
        parsed_data_dir / "tables" / "tables.jsonl",
    ]:
        if artifact.exists():
            artifact.unlink()


def _split_pages(markdown: str) -> list[tuple[int | None, list[str]]]:
    pages: list[tuple[int | None, list[str]]] = []
    current_page: int | None = None
    current_lines: list[str] = []
    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        page_match = PAGE_MARKER_RE.match(line.strip())
        if page_match:
            if current_lines or current_page is not None:
                pages.append((current_page, current_lines))
            current_page = int(page_match.group(1))
            current_lines = []
            continue
        current_lines.append(line)
    pages.append((current_page, current_lines))
    return pages


def _detect_repeated_headers_and_footers(pages: list[tuple[int | None, list[str]]]) -> set[str]:
    counts: dict[str, int] = {}
    for _, lines in pages:
        nonempty = [line.strip() for line in lines if line.strip()]
        candidates = nonempty[:2] + nonempty[-2:]
        for candidate in candidates:
            counts[candidate] = counts.get(candidate, 0) + 1
    threshold = max(2, len(pages) // 2)
    return {line for line, count in counts.items() if count >= threshold and len(line.split()) <= 8}


def _looks_like_broken_line(current: str, nxt: str) -> bool:
    current = current.strip()
    nxt = nxt.strip()
    if not current or not nxt:
        return False
    if any(pattern.match(current) for pattern in [PAGE_MARKER_RE, ITEM_HEADING_RE, MARKDOWN_HEADING_RE]):
        return False
    if _looks_like_heading(current) or _looks_like_heading(nxt):
        return False
    if current.endswith(('.', ':', ';', '?', '!', '|')):
        return False
    if nxt.startswith(('-', '*', '•')):
        return False
    return nxt[:1].islower() or nxt[:1].isdigit() or nxt.startswith('(')


def _rejoin_broken_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if merged and merged[-1] != "":
                merged.append("")
            continue
        if not merged:
            merged.append(stripped)
            continue
        if _looks_like_broken_line(merged[-1], stripped):
            if merged[-1].endswith('-'):
                merged[-1] = merged[-1][:-1] + stripped.lstrip()
            else:
                merged[-1] = merged[-1] + " " + stripped
        else:
            merged.append(stripped)
    return merged


def _looks_like_heading(line: str) -> bool:
    if not line or len(line) > 120 or line.endswith('.'):
        return False
    if VALUE_RE.search(line):
        return False
    words = line.split()
    if not 1 <= len(words) <= 12:
        return False
    alpha_words = [word for word in words if re.search(r"[A-Za-z]", word)]
    if len(alpha_words) < 2:
        return False
    titleish = sum(1 for word in alpha_words if word[:1].isupper() or word.isupper()) / len(alpha_words)
    return titleish >= 0.7


def _normalize_year(raw_year: str) -> int:
    year = int(raw_year)
    if year < 100:
        year += 2000
    return year


def _contains_term(text: str, term: str) -> bool:
    if " " in term or "-" in term:
        return term in text
    return re.search(rf"\b{re.escape(term)}\b", text) is not None


def _extract_period_headers(line: str) -> list[str]:
    tokens = [match.group(0).replace(" ", "") for match in PERIOD_RE.finditer(line)]
    return tokens


def _normalize_period_token(value: str) -> str | None:
    tokens = _extract_period_headers(value)
    return tokens[0] if tokens else None


def _numeric_token_count(line: str) -> int:
    return len(VALUE_RE.findall(line))


def _parse_numeric_row(line: str, headers: list[str]) -> tuple[str, dict[str, str]] | None:
    match = VALUE_RE.search(line)
    if match is None:
        return None
    label = line[: match.start()].strip(" :-") or "Row"
    values = VALUE_RE.findall(line[match.start() :])
    if len(values) < min(3, len(headers)):
        return None
    mapped = {header: value.replace("$", "") for header, value in zip(headers, values)}
    return label, mapped


def _infer_metric_tags(text: str) -> list[str]:
    lowered = text.lower()
    tags: list[str] = []
    for tag, keywords in {
        "revenue": ["revenue", "subscription"],
        "arr": ["arr", "recurring"],
        "margin": ["margin"],
        "cost": ["cost", "expense"],
        "income": ["income", "eps", "earnings"],
        "employee": ["employee", "shares outstanding"],
        "segment": ["digital media", "digital experience", "publishing and advertising"],
    }.items():
        if any(keyword in lowered for keyword in keywords):
            tags.append(tag)
    return tags


def _split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped:
        return []
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_markdown_table_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("|") and stripped.endswith("|"):
        return True
    return stripped.count("|") >= 2


def _is_markdown_separator_row(cells: list[str]) -> bool:
    present_cells = [cell.strip() for cell in cells if cell.strip()]
    if not present_cells:
        return False
    return all(MARKDOWN_TABLE_SEPARATOR_CELL_RE.match(cell) is not None for cell in present_cells)


def _select_table_label_index(headers: list[str], period_columns: dict[int, str]) -> int:
    candidates = [idx for idx in range(len(headers)) if idx not in period_columns]
    if not candidates:
        return 0
    for idx in candidates:
        lowered = headers[idx].lower()
        if any(token in lowered for token in ["description", "metric", "label", "segment", "item"]):
            return idx
    for idx in candidates:
        if headers[idx].strip():
            return idx
    return candidates[0]


def _extract_cell_numeric_value(cell: str) -> str | None:
    cleaned = re.sub(r"<[^>]+>", " ", cell)
    cleaned = normalize_whitespace(cleaned)
    match = VALUE_RE.search(cleaned)
    if match is None:
        return None
    return match.group(0).replace("$", "")


def _summarize_normalized_rows(rows: list[dict[str, Any]]) -> str:
    summary_rows = "; ".join(
        f"{row['label']}: " + ", ".join(f"{period}={value}" for period, value in list(row["values"].items())[:3])
        for row in rows[:3]
    )
    return summary_rows


def _normalize_markdown_table(lines: list[str], page_hint: int | None) -> tuple[list[dict[str, Any]], list[str], str | None]:
    parsed_rows = [_split_markdown_row(line) for line in lines]
    parsed_rows = [row for row in parsed_rows if any(cell.strip() for cell in row)]
    if len(parsed_rows) < 2:
        return [], [], None

    width = max(len(row) for row in parsed_rows)
    if width < 2:
        return [], [], None
    padded_rows = [row + [""] * (width - len(row)) for row in parsed_rows]

    header = padded_rows[0]
    data_start = 1
    if len(padded_rows) > 1 and _is_markdown_separator_row(padded_rows[1]):
        data_start = 2
    elif _is_markdown_separator_row(padded_rows[0]):
        if len(padded_rows) < 3:
            return [], [], None
        header = padded_rows[1]
        data_start = 2
    if data_start >= len(padded_rows):
        return [], [], None

    period_columns: dict[int, str] = {}
    for idx, cell in enumerate(header):
        token = _normalize_period_token(cell)
        if token:
            period_columns[idx] = token
    if len(period_columns) < 2:
        return [], [], None

    label_idx = _select_table_label_index(header, period_columns)
    normalized_rows: list[dict[str, Any]] = []
    metric_tags: set[str] = set()

    for row in padded_rows[data_start:]:
        if _is_markdown_separator_row(row):
            continue
        label = row[label_idx].strip() if label_idx < len(row) else ""
        label = normalize_whitespace(label).strip(":-")
        if not label:
            continue

        values: dict[str, str] = {}
        for col_idx, period in period_columns.items():
            if col_idx >= len(row):
                continue
            raw_value = _extract_cell_numeric_value(row[col_idx])
            if raw_value is None:
                continue
            values[period] = raw_value

        if len(values) < 2:
            continue

        normalized_rows.append({"label": label, "values": values, "page": page_hint})
        metric_tags.update(_infer_metric_tags(label))

    if not normalized_rows:
        return [], sorted(metric_tags), None
    summary = _summarize_normalized_rows(normalized_rows)
    return normalized_rows, sorted(metric_tags), summary


def _previous_title_hint(lines: list[str], raw_line: str) -> str | None:
    try:
        index = lines.index(raw_line)
    except ValueError:
        return None
    for candidate in reversed(lines[:index]):
        stripped = candidate.strip()
        if not stripped or PAGE_MARKER_RE.match(stripped):
            continue
        if _looks_like_heading(stripped):
            return stripped
    return None


def _to_json_safe(value: Any, depth: int = 0, max_depth: int = 10) -> Any:
    if depth > max_depth:
        return str(type(value).__name__)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item, depth + 1, max_depth) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(item, depth + 1, max_depth) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _to_json_safe(value.model_dump(), depth + 1, max_depth)
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _to_json_safe(value.tolist(), depth + 1, max_depth)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _to_json_safe(vars(value), depth + 1, max_depth)
        except Exception:
            pass
    return str(value)
