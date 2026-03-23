from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ParserQuality:
    score: float
    issues: list[str] = field(default_factory=list)
    heading_count: int = 0
    broken_line_ratio: float = 0.0
    empty_line_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedDocumentArtifact:
    doc_id: str
    doc_name: str
    source_path: str
    parser_used: str
    markdown_path: str
    structured_path: str
    file_size_bytes: int
    page_count: int
    quality: ParserQuality
    raw_markdown: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentRecord:
    doc_id: str
    doc_name: str
    source_path: str
    parser_used: str
    markdown_path: str
    structured_path: str
    report_type: str
    inferred_period: str | None
    inferred_year: int | None
    inferred_quarter: int | None
    likely_business_areas: list[str] = field(default_factory=list)
    latest_sort_key: int = 0
    page_count: int = 0
    parser_quality_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SectionRecord:
    section_id: str
    doc_id: str
    doc_name: str
    heading: str
    heading_path: list[str]
    page_start: int | None
    page_end: int | None
    section_type: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChunkRecord:
    chunk_id: str
    section_id: str
    doc_id: str
    doc_name: str
    heading_path: list[str]
    page_start: int | None
    page_end: int | None
    chunk_type: str
    text: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TableRecord:
    table_id: str
    doc_id: str
    section_id: str | None
    doc_name: str
    page_start: int | None
    page_end: int | None
    title: str | None
    heading_path: list[str]
    table_markdown: str
    table_text_summary: str
    normalized_rows: list[dict[str, Any]] = field(default_factory=list)
    metric_tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QuestionPlan:
    question: str
    question_class: str
    lexical_query: str
    semantic_query: str
    prioritize_tables: bool
    prioritize_latest: bool
    prioritize_report_types: list[str] = field(default_factory=list)
    target_metrics: list[str] = field(default_factory=list)
    target_entities: list[str] = field(default_factory=list)
    time_focus: str | None = None
    widen_if_sparse: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalCandidate:
    object_id: str
    object_type: str
    doc_id: str
    doc_name: str
    source_path: str
    heading_path: list[str]
    page_start: int | None
    page_end: int | None
    text: str
    title: str | None
    bm25_score: float
    dense_score: float
    fused_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceCitation:
    doc_name: str
    heading_path: list[str]
    page_start: int | None
    page_end: int | None
    object_type: str
    object_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerReport:
    question: str
    question_class: str
    direct_answer: str
    key_evidence: list[str]
    sources: list[SourceCitation]
    caveats: list[str]
    evidence_pack: list[dict[str, Any]]
    analytics: dict[str, Any]
    markdown_report: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sources"] = [source.to_dict() for source in self.sources]
        return data
