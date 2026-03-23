from __future__ import annotations

import re
from typing import Any

from leadership_agent.config import AppConfig
from leadership_agent.indexing import DenseArtifact, MetadataStore, bm25_scores
from leadership_agent.ingest import load_parsed_artifacts
from leadership_agent.models import ChunkRecord, QuestionPlan, RetrievalCandidate
from leadership_agent.utils import min_max_normalize, openai_json_completion

QUESTION_CLASSES = {
    "metric_lookup",
    "trend_analysis",
    "comparison",
    "risk_extraction",
    "strategic_summary",
    "general_factoid",
}


class QuestionClassifier:
    def __init__(self, config: AppConfig):
        self.config = config

    def classify(self, question: str) -> tuple[QuestionPlan, dict[str, Any]]:
        heuristic = self._heuristic_plan(question)
        if self.config.runtime.llm_provider != "openai" or not self.config.runtime.openai_api_key:
            return heuristic, {"model": None, "errors": []}

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "question_class": {"type": "string", "enum": sorted(QUESTION_CLASSES)},
                "lexical_query": {"type": "string"},
                "semantic_query": {"type": "string"},
                "prioritize_tables": {"type": "boolean"},
                "prioritize_latest": {"type": "boolean"},
                "prioritize_report_types": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
                "target_metrics": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
                "target_entities": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
                "time_focus": {"type": ["string", "null"]},
                "widen_if_sparse": {"type": "boolean"},
            },
            "required": [
                "question_class",
                "lexical_query",
                "semantic_query",
                "prioritize_tables",
                "prioritize_latest",
                "prioritize_report_types",
                "target_metrics",
                "target_entities",
                "time_focus",
                "widen_if_sparse",
            ],
        }
        system_prompt = (
            "You are a question router for a grounded document-intelligence system. "
            "Classify the leadership question, produce focused retrieval queries, and set retrieval priorities."
        )
        user_prompt = (
            f"Question: {question}\n\n"
            "Use only these classes: metric_lookup, trend_analysis, comparison, risk_extraction, strategic_summary, general_factoid. "
            "Set prioritize_tables=true for revenue/KPI/trend questions. Prefer annual_report for filing/risk questions."
        )
        parsed, usage = openai_json_completion(
            self.config.runtime,
            self.config.nano_model_candidates(),
            system_prompt,
            user_prompt,
            "question_plan",
            schema,
        )
        if parsed is None:
            return heuristic, usage

        plan = QuestionPlan(
            question=question,
            question_class=parsed.get("question_class", heuristic.question_class),
            lexical_query=parsed.get("lexical_query", heuristic.lexical_query),
            semantic_query=parsed.get("semantic_query", heuristic.semantic_query),
            prioritize_tables=bool(parsed.get("prioritize_tables", heuristic.prioritize_tables)),
            prioritize_latest=bool(parsed.get("prioritize_latest", heuristic.prioritize_latest)),
            prioritize_report_types=list(parsed.get("prioritize_report_types", heuristic.prioritize_report_types)),
            target_metrics=list(parsed.get("target_metrics", heuristic.target_metrics)),
            target_entities=list(parsed.get("target_entities", heuristic.target_entities)),
            time_focus=parsed.get("time_focus", heuristic.time_focus),
            widen_if_sparse=bool(parsed.get("widen_if_sparse", heuristic.widen_if_sparse)),
        )
        plan = self._apply_high_confidence_overrides(question, heuristic, plan)
        return plan, usage

    def _apply_high_confidence_overrides(self, question: str, heuristic: QuestionPlan, llm_plan: QuestionPlan) -> QuestionPlan:
        q = question.lower()
        high_confidence_class = None
        if any(token in q for token in ["underperform", "underperforming", "lagging", "weakest", "versus", "compare"]):
            high_confidence_class = "comparison"
        elif any(token in q for token in ["trend", "trajectory", "over time", "yoy", "qoq"]):
            high_confidence_class = "trend_analysis"
        elif any(token in q for token in ["risk", "risks", "headwind", "highlighted", "cybersecurity"]):
            high_confidence_class = "risk_extraction"

        if high_confidence_class and llm_plan.question_class != high_confidence_class:
            return heuristic
        return llm_plan

    def _heuristic_plan(self, question: str) -> QuestionPlan:
        q = question.lower()
        underperforming_prompt = any(token in q for token in ["underperform", "lagging", "weakest"])
        last_quarter_prompt = "last quarter" in q or "recent quarter" in q or "latest quarter" in q
        detected_entities = [
            entity
            for entity in ["digital media", "digital experience", "acrobat", "firefly", "cybersecurity"]
            if entity in q
        ]
        question_class = "general_factoid"
        if any(token in q for token in ["trend", "trajectory", "over time", "yoy", "qoq"]):
            question_class = "trend_analysis"
        elif any(token in q for token in ["underperform", "compare", "which", "versus", "larger segment"]) or (
            "perform" in q and len(detected_entities) >= 2
        ):
            question_class = "comparison"
        elif any(token in q for token in ["risk", "risks", "headwind", "cybersecurity", "highlighted"]):
            question_class = "risk_extraction"
        elif any(token in q for token in ["strategy", "emphasize", "focus", "ai", "firefly", "acrobat ai assistant"]):
            question_class = "strategic_summary"
        elif any(token in q for token in ["revenue", "arr", "margin", "employees", "eps", "cost", "expense"]):
            question_class = "metric_lookup"

        target_metrics = [metric for metric in ["revenue", "arr", "margin", "cost", "income", "employees"] if metric in q]
        if question_class == "comparison" and underperforming_prompt and "revenue" not in target_metrics:
            target_metrics.append("revenue")
        target_entities = detected_entities
        prioritize_tables = question_class in {"metric_lookup", "trend_analysis", "comparison"}
        prioritize_latest = any(token in q for token in ["current", "recent", "latest", "last quarter"]) or question_class in {
            "metric_lookup",
            "trend_analysis",
            "strategic_summary",
        }
        report_types: list[str] = []
        if question_class in {"metric_lookup", "trend_analysis", "comparison"}:
            report_types = ["investor_datasheet", "quarterly_report"]
        elif question_class == "risk_extraction":
            report_types = ["quarterly_report", "annual_report"] if last_quarter_prompt else ["annual_report", "quarterly_report"]
        elif question_class == "strategic_summary":
            report_types = ["quarterly_report", "annual_report"]

        time_focus = None
        period_match = re.search(r"(q[1-4]\s*fy\s*\d{4}|fy\s*\d{4}|20\d{2})", q)
        if period_match:
            time_focus = re.sub(r"\s+", "", period_match.group(1).upper()).replace("FY", " FY")
            time_focus = time_focus.replace("Q", "Q").strip()

        lexical_query = question
        if question_class == "trend_analysis":
            lexical_query += " revenue trend quarterly table investor datasheet"
        elif question_class == "comparison":
            lexical_query += " segment performance compare table"
            if underperforming_prompt:
                lexical_query += " department division business unit segment year-over-year growth revenue by segment"
        elif question_class == "risk_extraction":
            lexical_query += " risk factors cybersecurity management commentary"
            if last_quarter_prompt:
                lexical_query += " forward-looking statements factors that might cause earnings release quarter"
        elif question_class == "strategic_summary":
            lexical_query += " strategy priorities product innovation roadmap ai initiatives"

        semantic_query = lexical_query
        return QuestionPlan(
            question=question,
            question_class=question_class,
            lexical_query=lexical_query,
            semantic_query=semantic_query,
            prioritize_tables=prioritize_tables,
            prioritize_latest=prioritize_latest,
            prioritize_report_types=report_types,
            target_metrics=target_metrics,
            target_entities=target_entities,
            time_focus=time_focus,
            widen_if_sparse=True,
        )


class HybridRetriever:
    def __init__(self, config: AppConfig):
        self.config = config
        documents, sections, chunks, tables = load_parsed_artifacts(config)
        self.documents = {row.doc_id: row for row in documents}
        self.sections = {row.section_id: row for row in sections}
        self.chunks = {row.chunk_id: row for row in chunks}
        self.tables = {row.table_id: row for row in tables}
        self.chunks_by_section: dict[str, list[ChunkRecord]] = {}
        for chunk in chunks:
            self.chunks_by_section.setdefault(chunk.section_id, []).append(chunk)
        for section_chunks in self.chunks_by_section.values():
            section_chunks.sort(key=lambda item: item.chunk_index)

        self.metadata = MetadataStore(config.index_dir / "metadata" / "leadership.duckdb")
        self.section_dense = DenseArtifact("sections", config, config.index_dir)
        self.chunk_dense = DenseArtifact("chunks", config, config.index_dir)
        self.table_dense = DenseArtifact("tables", config, config.index_dir)
        self.section_dense.load()
        self.chunk_dense.load()
        if self.tables:
            self.table_dense.load()
        self.classifier = QuestionClassifier(config)

    def retrieve(self, question: str) -> tuple[QuestionPlan, list[RetrievalCandidate], dict[str, Any]]:
        plan, classifier_usage = self.classifier.classify(question)
        doc_ids = self.metadata.filter_doc_ids(
            prioritize_latest=plan.prioritize_latest,
            report_types=plan.prioritize_report_types,
            time_focus=plan.time_focus,
        )
        if not doc_ids:
            doc_ids = sorted(self.documents.keys(), key=lambda item: self.documents[item].latest_sort_key, reverse=True)

        allowed_section_ids = set(self.metadata.section_ids_for_docs(doc_ids))
        section_hits = self._search_sections(plan, allowed_section_ids)
        shortlisted_sections = [candidate.object_id for candidate in section_hits[: self.config.retrieval.top_sections]]

        allowed_chunk_ids = set(self.metadata.chunk_ids_for_sections(shortlisted_sections)) if shortlisted_sections else set()
        if not allowed_chunk_ids and plan.widen_if_sparse:
            allowed_chunk_ids = {chunk_id for chunk_id, chunk in self.chunks.items() if chunk.doc_id in doc_ids}
        chunk_hits = self._search_chunks(plan, allowed_chunk_ids or None)

        allowed_table_ids = set(self.metadata.table_ids_for_docs(doc_ids)) if self.tables else set()
        table_hits = self._search_tables(plan, allowed_table_ids or None)
        if plan.widen_if_sparse and plan.prioritize_tables and len(table_hits) < 2:
            table_hits = self._search_tables(plan, None)

        final_candidates = self._build_evidence_pack(plan, section_hits, chunk_hits, table_hits)
        analytics = {
            "question_plan": plan.to_dict(),
            "classifier_usage": classifier_usage,
            "doc_filter_ids": doc_ids,
            "section_hits": len(section_hits),
            "chunk_hits": len(chunk_hits),
            "table_hits": len(table_hits),
            "returned_candidates": len(final_candidates),
        }
        return plan, final_candidates, analytics

    def _search_sections(self, plan: QuestionPlan, allowed_ids: set[str] | None) -> list[RetrievalCandidate]:
        bm25 = bm25_scores(
            self.config.index_dir / "bm25" / "sections.pkl",
            plan.lexical_query,
            allowed_ids=allowed_ids,
            top_k=self.config.retrieval.top_sections * self.config.retrieval.search_multiplier,
        )
        dense = self.section_dense.query(
            plan.semantic_query,
            top_k=self.config.retrieval.top_sections * self.config.retrieval.search_multiplier,
            allowed_ids=allowed_ids,
        )
        return self._fuse_candidates("section", bm25, dense, plan, self.sections)

    def _search_chunks(self, plan: QuestionPlan, allowed_ids: set[str] | None) -> list[RetrievalCandidate]:
        bm25 = bm25_scores(
            self.config.index_dir / "bm25" / "chunks.pkl",
            plan.lexical_query,
            allowed_ids=allowed_ids,
            top_k=self.config.retrieval.top_chunks * self.config.retrieval.search_multiplier,
        )
        dense = self.chunk_dense.query(
            plan.semantic_query,
            top_k=self.config.retrieval.top_chunks * self.config.retrieval.search_multiplier,
            allowed_ids=allowed_ids,
        )
        return self._fuse_candidates("chunk", bm25, dense, plan, self.chunks)

    def _search_tables(self, plan: QuestionPlan, allowed_ids: set[str] | None) -> list[RetrievalCandidate]:
        if not self.tables:
            return []
        bm25 = bm25_scores(
            self.config.index_dir / "bm25" / "tables.pkl",
            plan.lexical_query,
            allowed_ids=allowed_ids,
            top_k=self.config.retrieval.top_tables * self.config.retrieval.search_multiplier,
        )
        dense = self.table_dense.query(
            plan.semantic_query,
            top_k=self.config.retrieval.top_tables * self.config.retrieval.search_multiplier,
            allowed_ids=allowed_ids,
        )
        return self._fuse_candidates("table", bm25, dense, plan, self.tables)

    def _fuse_candidates(
        self,
        object_type: str,
        bm25_scores_map: dict[str, float],
        dense_scores_map: dict[str, float],
        plan: QuestionPlan,
        rows: dict[str, Any],
    ) -> list[RetrievalCandidate]:
        bm25_norm = min_max_normalize(bm25_scores_map)
        dense_norm = min_max_normalize(dense_scores_map)
        candidate_ids = set(bm25_scores_map) | set(dense_scores_map)
        results: list[RetrievalCandidate] = []
        for object_id in candidate_ids:
            row = rows[object_id]
            fused = (
                self.config.retrieval.bm25_weight * bm25_norm.get(object_id, 0.0)
                + self.config.retrieval.dense_weight * dense_norm.get(object_id, 0.0)
                + self._heuristic_bonus(object_type, row, plan)
            )
            if object_type == "table":
                summary = (row.table_text_summary or "").strip()
                if not summary or summary.endswith("markdown table extracted from section."):
                    text = row.table_markdown or summary
                else:
                    text = summary
                title = row.title
            elif object_type == "section":
                text = row.text
                title = row.heading
            else:
                text = row.text
                title = row.heading_path[-1] if row.heading_path else None
            results.append(
                RetrievalCandidate(
                    object_id=object_id,
                    object_type=object_type,
                    doc_id=row.doc_id,
                    doc_name=row.doc_name,
                    source_path=self.documents[row.doc_id].source_path,
                    heading_path=list(row.heading_path),
                    page_start=row.page_start,
                    page_end=row.page_end,
                    text=text,
                    title=title,
                    bm25_score=float(bm25_scores_map.get(object_id, 0.0)),
                    dense_score=float(dense_scores_map.get(object_id, 0.0)),
                    fused_score=float(fused),
                    metadata=row.metadata if hasattr(row, "metadata") else {},
                )
            )
        results.sort(key=lambda item: item.fused_score, reverse=True)
        return results

    def _heuristic_bonus(self, object_type: str, row: Any, plan: QuestionPlan) -> float:
        bonus = 0.0
        doc = self.documents[row.doc_id]
        question_lower = plan.question.lower()
        underperforming_prompt = any(token in question_lower for token in ["underperform", "lagging", "weakest"])
        last_quarter_prompt = "last quarter" in question_lower or "recent quarter" in question_lower or "latest quarter" in question_lower
        if plan.prioritize_latest:
            latest_key = max(document.latest_sort_key for document in self.documents.values())
            if doc.latest_sort_key == latest_key:
                bonus += self.config.retrieval.latest_doc_bonus
        if plan.question_class == "risk_extraction" and last_quarter_prompt:
            if doc.report_type == "quarterly_report":
                bonus += 0.12
            elif doc.report_type == "annual_report":
                bonus -= 0.06
        if object_type == "table":
            bonus += self.config.retrieval.table_weight * 0.05
            if plan.prioritize_tables:
                bonus += self.config.retrieval.table_priority_bonus
            normalized_rows = getattr(row, "normalized_rows", []) or []
            table_source = (getattr(row, "metadata", {}) or {}).get("table_source", "")
            if normalized_rows:
                bonus += 0.08 + min(0.08, len(normalized_rows) * 0.01)
            elif plan.question_class in {"metric_lookup", "trend_analysis", "comparison"}:
                bonus -= 0.06
            if table_source in {"numeric_block", "markdown_table_normalized"}:
                bonus += 0.06
            metric_tags = {tag.lower() for tag in getattr(row, "metric_tags", [])}
            if metric_tags & set(plan.target_metrics):
                bonus += 0.10
            if plan.question_class == "comparison" and underperforming_prompt:
                if "segment" in metric_tags:
                    bonus += 0.15
                elif metric_tags & {"income", "cost", "margin"}:
                    bonus -= 0.10
        if object_type == "section":
            if getattr(row, "section_type", "other") == "risk" and plan.question_class == "risk_extraction":
                bonus += 0.12
            if getattr(row, "section_type", "other") == "strategy" and plan.question_class == "strategic_summary":
                bonus += 0.10
            if getattr(row, "section_type", "other") == "financials" and plan.question_class in {"metric_lookup", "trend_analysis", "comparison"}:
                bonus += 0.08
        if object_type == "chunk":
            section_type = row.metadata.get("section_type", "other")
            if section_type == "financials" and plan.prioritize_tables:
                bonus += 0.04
            if section_type == "risk" and plan.question_class == "risk_extraction":
                bonus += 0.08
        haystack = (getattr(row, "title", None) or "") + " " + getattr(row, "text", "")[:600]
        haystack = haystack.lower()
        if plan.question_class == "risk_extraction" and "forward-looking statement" in haystack:
            bonus += 0.15
        if plan.question_class == "comparison" and underperforming_prompt:
            if any(token in haystack for token in ["segment", "department", "division", "business unit"]):
                bonus += 0.10
            if "non-gaap results" in haystack:
                bonus -= 0.08
        if any(entity in haystack for entity in plan.target_entities):
            bonus += 0.08
        if any(metric in haystack for metric in plan.target_metrics):
            bonus += 0.06
        return bonus

    def _build_evidence_pack(
        self,
        plan: QuestionPlan,
        section_hits: list[RetrievalCandidate],
        chunk_hits: list[RetrievalCandidate],
        table_hits: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        evidence: list[RetrievalCandidate] = []
        seen: set[tuple[str, str]] = set()

        def add(candidate: RetrievalCandidate) -> None:
            key = (candidate.object_type, candidate.object_id)
            if key in seen or len(evidence) >= self.config.answer.max_context_items:
                return
            seen.add(key)
            evidence.append(candidate)

        if plan.prioritize_tables:
            for candidate in table_hits[: self.config.retrieval.top_tables]:
                add(candidate)

        for candidate in chunk_hits[: self.config.retrieval.top_chunks]:
            add(candidate)
            if len(evidence) >= self.config.answer.max_context_items:
                break
            if candidate.object_type == "chunk":
                for sibling in self._sibling_chunks(candidate.object_id):
                    add(sibling)
                    if len(evidence) >= self.config.answer.max_context_items:
                        break

        for candidate in section_hits[: self.config.retrieval.top_sections]:
            add(candidate)
            if len(evidence) >= self.config.answer.max_context_items:
                break

        if not evidence:
            combined = table_hits + chunk_hits + section_hits
            for candidate in combined[: self.config.answer.max_context_items]:
                add(candidate)

        evidence.sort(key=lambda item: item.fused_score, reverse=True)
        return evidence

    def _sibling_chunks(self, chunk_id: str) -> list[RetrievalCandidate]:
        chunk = self.chunks.get(chunk_id)
        if chunk is None:
            return []
        siblings = self.chunks_by_section.get(chunk.section_id, [])
        results: list[RetrievalCandidate] = []
        for sibling in siblings:
            if abs(sibling.chunk_index - chunk.chunk_index) != 1:
                continue
            results.append(
                RetrievalCandidate(
                    object_id=sibling.chunk_id,
                    object_type="chunk",
                    doc_id=sibling.doc_id,
                    doc_name=sibling.doc_name,
                    source_path=self.documents[sibling.doc_id].source_path,
                    heading_path=list(sibling.heading_path),
                    page_start=sibling.page_start,
                    page_end=sibling.page_end,
                    text=sibling.text,
                    title=sibling.heading_path[-1] if sibling.heading_path else None,
                    bm25_score=0.0,
                    dense_score=0.0,
                    fused_score=0.01,
                    metadata=sibling.metadata,
                )
            )
        return results
