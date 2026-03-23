from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from leadership_agent.utils import dedupe_preserve_order


@dataclass
class ParserSettings:
    primary: str = "docling"
    fallback: str = "marker"
    persist_markdown: bool = True
    allow_native_fallback: bool = True
    quality_min_score: float = 0.55


@dataclass
class ModelSettings:
    nano: str = "gpt-5.4-nano"
    mini: str = "gpt-5.4-mini"
    embedding: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32


@dataclass
class ChunkingSettings:
    max_tokens: int = 650
    overlap_tokens: int = 80


@dataclass
class RetrievalSettings:
    top_sections: int = 8
    top_chunks: int = 8
    top_tables: int = 5
    table_weight: float = 1.25
    bm25_weight: float = 0.45
    dense_weight: float = 0.55
    latest_doc_bonus: float = 0.12
    table_priority_bonus: float = 0.15
    search_multiplier: int = 6


@dataclass
class AnswerSettings:
    max_context_items: int = 10
    require_citations: bool = True
    insufficient_evidence_threshold: float = 0.20


@dataclass
class RuntimeSettings:
    llm_provider: str = "openai"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_timeout_sec: int = 60


@dataclass
class AppConfig:
    raw_data_dir: Path = Path("data/raw")
    parsed_data_dir: Path = Path("data/parsed")
    index_dir: Path = Path("data/indexes")
    output_dir: Path = Path("outputs")
    eval_questions_path: Path = Path("data/eval/questions.json")
    parser: ParserSettings = field(default_factory=ParserSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    answer: AnswerSettings = field(default_factory=AnswerSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)

    @classmethod
    def load(cls, config_path: str | Path = "config.yaml") -> "AppConfig":
        config = cls()
        path = Path(config_path)
        if path.exists():
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            config = _merge_config(config, payload)
        return _apply_env_overrides(config)

    def parser_chain(self) -> list[str]:
        chain = [self.parser.primary, self.parser.fallback]
        if self.parser.allow_native_fallback:
            chain.append("native")
        return dedupe_preserve_order([item.strip().lower() for item in chain if item and item.strip()])

    def nano_model_candidates(self) -> list[str]:
        configured = self.models.nano.strip()
        candidates = [configured]
        if configured == "gpt-5.4-nano":
            candidates.append("gpt-5-nano")
        return dedupe_preserve_order(candidates)

    def mini_model_candidates(self) -> list[str]:
        configured = self.models.mini.strip()
        candidates = [configured]
        if configured == "gpt-5.4-mini":
            candidates.append("gpt-5-mini")
        return dedupe_preserve_order(candidates)


def _merge_dataclass(instance: Any, updates: dict[str, Any]) -> Any:
    for key, value in updates.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        elif isinstance(current, Path):
            setattr(instance, key, Path(value))
        else:
            setattr(instance, key, value)
    return instance


def _merge_config(config: AppConfig, payload: dict[str, Any]) -> AppConfig:
    return _merge_dataclass(config, payload)


def _parse_bool(raw: str, default: bool) -> bool:
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def _apply_env_overrides(config: AppConfig) -> AppConfig:
    env = os.environ
    if env.get("RAW_DATA_DIR"):
        config.raw_data_dir = Path(env["RAW_DATA_DIR"])
    if env.get("PARSED_DATA_DIR"):
        config.parsed_data_dir = Path(env["PARSED_DATA_DIR"])
    if env.get("INDEX_DIR"):
        config.index_dir = Path(env["INDEX_DIR"])
    if env.get("OUTPUT_DIR"):
        config.output_dir = Path(env["OUTPUT_DIR"])
    if env.get("EVAL_QUESTIONS_PATH"):
        config.eval_questions_path = Path(env["EVAL_QUESTIONS_PATH"])

    if env.get("PARSER_PRIMARY"):
        config.parser.primary = env["PARSER_PRIMARY"]
    if env.get("PARSER_FALLBACK"):
        config.parser.fallback = env["PARSER_FALLBACK"]
    config.parser.persist_markdown = _parse_bool(env.get("PERSIST_PARSED_MARKDOWN"), config.parser.persist_markdown)
    config.parser.allow_native_fallback = _parse_bool(env.get("ALLOW_NATIVE_FALLBACK"), config.parser.allow_native_fallback)
    if env.get("PARSER_QUALITY_MIN_SCORE"):
        config.parser.quality_min_score = float(env["PARSER_QUALITY_MIN_SCORE"])

    if env.get("OPENAI_NANO_MODEL"):
        config.models.nano = env["OPENAI_NANO_MODEL"]
    if env.get("OPENAI_MINI_MODEL"):
        config.models.mini = env["OPENAI_MINI_MODEL"]
    if env.get("EMBEDDING_MODEL"):
        config.models.embedding = env["EMBEDDING_MODEL"]
    if env.get("EMBEDDING_BATCH_SIZE"):
        config.models.embedding_batch_size = int(env["EMBEDDING_BATCH_SIZE"])

    if env.get("CHUNK_MAX_TOKENS"):
        config.chunking.max_tokens = int(env["CHUNK_MAX_TOKENS"])
    if env.get("CHUNK_OVERLAP_TOKENS"):
        config.chunking.overlap_tokens = int(env["CHUNK_OVERLAP_TOKENS"])

    if env.get("RETRIEVAL_TOP_SECTIONS"):
        config.retrieval.top_sections = int(env["RETRIEVAL_TOP_SECTIONS"])
    if env.get("RETRIEVAL_TOP_CHUNKS"):
        config.retrieval.top_chunks = int(env["RETRIEVAL_TOP_CHUNKS"])
    if env.get("RETRIEVAL_TOP_TABLES"):
        config.retrieval.top_tables = int(env["RETRIEVAL_TOP_TABLES"])
    if env.get("RETRIEVAL_TABLE_WEIGHT"):
        config.retrieval.table_weight = float(env["RETRIEVAL_TABLE_WEIGHT"])
    if env.get("RETRIEVAL_BM25_WEIGHT"):
        config.retrieval.bm25_weight = float(env["RETRIEVAL_BM25_WEIGHT"])
    if env.get("RETRIEVAL_DENSE_WEIGHT"):
        config.retrieval.dense_weight = float(env["RETRIEVAL_DENSE_WEIGHT"])

    if env.get("ANSWER_MAX_CONTEXT_ITEMS"):
        config.answer.max_context_items = int(env["ANSWER_MAX_CONTEXT_ITEMS"])
    config.answer.require_citations = _parse_bool(env.get("ANSWER_REQUIRE_CITATIONS"), config.answer.require_citations)
    if env.get("INSUFFICIENT_EVIDENCE_THRESHOLD"):
        config.answer.insufficient_evidence_threshold = float(env["INSUFFICIENT_EVIDENCE_THRESHOLD"])

    if env.get("LLM_PROVIDER"):
        config.runtime.llm_provider = env["LLM_PROVIDER"].strip().lower()
    if env.get("OPENAI_API_KEY"):
        config.runtime.openai_api_key = env["OPENAI_API_KEY"].strip()
    if env.get("OPENAI_BASE_URL"):
        config.runtime.openai_base_url = env["OPENAI_BASE_URL"].strip()
    if env.get("LLM_TIMEOUT_SEC"):
        config.runtime.llm_timeout_sec = int(env["LLM_TIMEOUT_SEC"])

    return config
