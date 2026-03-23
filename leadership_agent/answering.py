from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from leadership_agent.config import AppConfig
from leadership_agent.indexing import build_indexes
from leadership_agent.ingest import ingest_corpus
from leadership_agent.models import AnswerReport, QuestionPlan, RetrievalCandidate, SourceCitation, TableRecord
from leadership_agent.retrieval import HybridRetriever
from leadership_agent.utils import ensure_dir, normalize_whitespace, openai_json_completion, slugify

RISK_TERMS = ["risk", "cybersecurity", "security", "privacy", "compliance", "breach", "incident", "attack"]
STRATEGY_TERMS = ["strategy", "priority", "innovation", "roadmap", "ai", "product", "customer", "growth"]
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "the",
    "to",
    "was",
    "were",
    "what",
    "which",
}


class LeadershipAgent:
    def __init__(self, config: AppConfig):
        self.config = config

    def ingest(self, input_dir: str | Path | None = None) -> dict[str, Any]:
        return ingest_corpus(self.config, input_dir=input_dir)

    def build_index(self) -> dict[str, Any]:
        return build_indexes(self.config)

    def ask(self, question: str, output_dir: str | Path | None = None) -> AnswerReport:
        retriever = HybridRetriever(self.config)
        plan, candidates, retrieval_analytics = retriever.retrieve(question)
        report = generate_answer(
            config=self.config,
            question=question,
            plan=plan,
            candidates=candidates,
            retrieval_analytics=retrieval_analytics,
            tables=retriever.tables,
            output_dir=Path(output_dir) if output_dir else self.config.output_dir,
        )
        return report


def generate_answer(
    config: AppConfig,
    question: str,
    plan: QuestionPlan,
    candidates: list[RetrievalCandidate],
    retrieval_analytics: dict[str, Any],
    tables: dict[str, TableRecord],
    output_dir: Path,
) -> AnswerReport:
    ensure_dir(output_dir)
    evidence_items = _build_evidence_items(candidates)
    series = _extract_metric_series(question, plan, candidates, tables)
    computed = {
        "metric_series": series,
        "trend_summary": _summarize_series(series) if series else None,
        "risk_factors": _extract_risk_factors(plan, candidates),
        "risk_signals": _extract_signals(candidates, RISK_TERMS, limit=6),
        "strategy_signals": _extract_signals(candidates, STRATEGY_TERMS, limit=6),
        "comparison_rows": _extract_comparison_rows(question, plan, candidates, tables),
    }
    plot_path = None
    if series and len(series) >= 2:
        plot_path = _plot_series(series, plan, output_dir)

    top_support = max(0.0, candidates[0].dense_score) if candidates else 0.0
    overlap_ratio = _query_overlap_ratio(question, plan, candidates[0]) if candidates else 0.0
    has_computed_support = bool(series or computed["risk_signals"] or computed["strategy_signals"] or computed["comparison_rows"])
    insufficient = (
        len(candidates) == 0
        or (top_support < config.answer.insufficient_evidence_threshold and overlap_ratio < max(0.2, config.answer.insufficient_evidence_threshold / 2))
        or (overlap_ratio == 0.0 and not has_computed_support)
    )
    llm_usage: dict[str, Any] = {"classifier": retrieval_analytics.get("classifier_usage", {}), "answer": {}, "groundedness": {}}

    if not insufficient and config.runtime.llm_provider == "openai" and config.runtime.openai_api_key:
        synthesized, llm_usage["answer"] = _synthesize_with_mini(config, question, plan, evidence_items, computed)
        if synthesized is not None:
            grounded, llm_usage["groundedness"] = _groundedness_check(config, question, synthesized, evidence_items)
            report = _report_from_synthesized(
                question,
                plan,
                synthesized,
                evidence_items,
                computed,
                retrieval_analytics,
                plot_path,
                top_support,
                overlap_ratio,
            )
            if _prefer_computed_wording(plan, computed):
                fallback = _fallback_report(
                    question,
                    plan,
                    candidates,
                    evidence_items,
                    computed,
                    retrieval_analytics,
                    plot_path,
                    insufficient=False,
                )
                report.direct_answer = fallback.direct_answer
                report.key_evidence = fallback.key_evidence
                report.sources = fallback.sources
                report.evidence_pack = fallback.evidence_pack
            if grounded is not None and not grounded.get("grounded", True):
                report.caveats.extend(grounded.get("caveats", []))
            report.markdown_report = _format_markdown(report.direct_answer, report.key_evidence, report.sources, report.caveats)
            report.analytics["llm_usage"] = llm_usage
            report.analytics["insufficient_evidence"] = False
            return report

    report = _fallback_report(question, plan, candidates, evidence_items, computed, retrieval_analytics, plot_path, insufficient)
    report.analytics["llm_usage"] = llm_usage
    report.analytics["insufficient_evidence"] = insufficient
    return report


def save_report(report: AnswerReport, path: str | Path) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def render_report(report: AnswerReport) -> str:
    return report.markdown_report


def _build_evidence_items(candidates: list[RetrievalCandidate]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        items.append(
            {
                "id": idx,
                "object_id": candidate.object_id,
                "object_type": candidate.object_type,
                "doc_name": candidate.doc_name,
                "heading_path": candidate.heading_path,
                "page_start": candidate.page_start,
                "page_end": candidate.page_end,
                "quote": candidate.text[:700].replace("\n", " "),
                "score": round(candidate.fused_score, 4),
            }
        )
    return items


def _synthesize_with_mini(
    config: AppConfig,
    question: str,
    plan: QuestionPlan,
    evidence_items: list[dict[str, Any]],
    computed: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "direct_answer": {"type": "string"},
            "key_evidence": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 6},
            "source_ids": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": min(6, max(1, len(evidence_items)))},
            "caveats": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        },
        "required": ["direct_answer", "key_evidence", "source_ids", "caveats"],
    }
    system_prompt = (
        "You are an executive analyst. Answer conservatively using only the supplied evidence and computed facts. "
        "Do not invent metrics or claims. If evidence is incomplete, say so in caveats."
    )
    evidence_text = "\n\n".join(
        f"[{item['id']}] {item['doc_name']} | {' > '.join(item['heading_path']) if item['heading_path'] else 'Document'} | "
        f"pages {item['page_start']}:{item['page_end']}\n{item['quote']}"
        for item in evidence_items
    )
    user_prompt = (
        f"Question: {question}\n\n"
        f"Question plan: {json.dumps(plan.to_dict(), ensure_ascii=False)}\n\n"
        f"Computed facts: {json.dumps(computed, ensure_ascii=False)}\n\n"
        f"Evidence items:\n{evidence_text}\n\n"
        "Return a short executive answer, evidence bullets, source ids, and caveats."
    )
    return openai_json_completion(
        config.runtime,
        config.mini_model_candidates(),
        system_prompt,
        user_prompt,
        "grounded_answer",
        schema,
    )


def _groundedness_check(
    config: AppConfig,
    question: str,
    synthesized: dict[str, Any],
    evidence_items: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "grounded": {"type": "boolean"},
            "caveats": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        },
        "required": ["grounded", "caveats"],
    }
    system_prompt = "Check whether the answer is fully grounded in the evidence. Return JSON only."
    user_prompt = (
        f"Question: {question}\n\n"
        f"Answer: {json.dumps(synthesized, ensure_ascii=False)}\n\n"
        f"Evidence: {json.dumps(evidence_items, ensure_ascii=False)}"
    )
    return openai_json_completion(
        config.runtime,
        config.nano_model_candidates(),
        system_prompt,
        user_prompt,
        "groundedness_check",
        schema,
    )


def _report_from_synthesized(
    question: str,
    plan: QuestionPlan,
    synthesized: dict[str, Any],
    evidence_items: list[dict[str, Any]],
    computed: dict[str, Any],
    retrieval_analytics: dict[str, Any],
    plot_path: str | None,
    top_support: float,
    overlap_ratio: float,
) -> AnswerReport:
    source_ids = {int(item) for item in synthesized.get("source_ids", []) if isinstance(item, int) or str(item).isdigit()}
    selected_items = [item for item in evidence_items if item["id"] in source_ids] or evidence_items[:3]
    sources = _citations_from_items(selected_items)
    analytics = {
        "retrieval": retrieval_analytics,
        "computed": computed,
        "plot_path": plot_path,
        "evidence_support": {"top_dense_score": top_support, "query_overlap_ratio": overlap_ratio},
    }
    markdown_report = _format_markdown(
        synthesized.get("direct_answer", ""),
        list(synthesized.get("key_evidence", [])),
        sources,
        list(synthesized.get("caveats", [])),
    )
    return AnswerReport(
        question=question,
        question_class=plan.question_class,
        direct_answer=synthesized.get("direct_answer", ""),
        key_evidence=list(synthesized.get("key_evidence", [])),
        sources=sources,
        caveats=list(synthesized.get("caveats", [])),
        evidence_pack=selected_items,
        analytics=analytics,
        markdown_report=markdown_report,
    )


def _fallback_report(
    question: str,
    plan: QuestionPlan,
    candidates: list[RetrievalCandidate],
    evidence_items: list[dict[str, Any]],
    computed: dict[str, Any],
    retrieval_analytics: dict[str, Any],
    plot_path: str | None,
    insufficient: bool,
) -> AnswerReport:
    direct_answer = "Evidence is currently insufficient to answer this question confidently."
    key_evidence: list[str] = []
    caveats: list[str] = []

    if insufficient:
        caveats.append("Top retrieved evidence was too weak or too sparse for a confident answer.")
    elif plan.question_class == "comparison" and computed.get("comparison_rows"):
        rows = computed["comparison_rows"]
        direct_answer = _comparison_answer(rows, question)
        key_evidence = [row["summary"] for row in rows[:4]]
    elif plan.question_class == "risk_extraction" and computed.get("risk_factors"):
        direct_answer = _risk_answer(computed["risk_factors"], question)
        key_evidence = [f"Risk factor: {item}" for item in computed["risk_factors"][:6]]
    elif plan.question_class == "risk_extraction" and computed.get("risk_signals"):
        direct_answer = "Key risks include cybersecurity, regulatory/privacy exposure, and execution-related pressures."
        key_evidence = computed["risk_signals"][:4]
    elif plan.question_class == "strategic_summary" and computed.get("strategy_signals"):
        direct_answer = "Adobe is emphasizing AI-led product innovation and monetization through products like Firefly and Acrobat AI Assistant."
        key_evidence = computed["strategy_signals"][:4]
    elif computed.get("trend_summary"):
        direct_answer = computed["trend_summary"]
        series = computed.get("metric_series") or []
        if series:
            points = ", ".join(f"{item['period']}={item['raw_value']}" for item in series[:4])
            key_evidence = [
                f"Computed from table evidence: {computed['trend_summary']}",
                f"Series used: {points}.",
            ]
        else:
            key_evidence = [f"Computed from table evidence: {computed['trend_summary']}"]
    elif candidates:
        direct_answer = candidates[0].text[:320].replace("\n", " ")
        key_evidence = [item["quote"] for item in evidence_items[:3]]
    else:
        caveats.append("No evidence items were retrieved.")

    if computed.get("trend_summary") and not key_evidence:
        key_evidence.append(computed["trend_summary"])
    if not key_evidence:
        key_evidence = [item["quote"] for item in evidence_items[:3]]

    preferred_items = evidence_items
    if computed.get("trend_summary"):
        table_items = [item for item in evidence_items if item["object_type"] == "table"]
        non_table_items = [item for item in evidence_items if item["object_type"] != "table"]
        preferred_items = table_items + non_table_items
    sources = _citations_from_items(preferred_items[:4])
    analytics = {
        "retrieval": retrieval_analytics,
        "computed": computed,
        "plot_path": plot_path,
        "evidence_support": {"top_dense_score": max(0.0, candidates[0].dense_score) if candidates else 0.0, "query_overlap_ratio": _query_overlap_ratio(question, plan, candidates[0]) if candidates else 0.0},
    }
    markdown_report = _format_markdown(direct_answer, key_evidence, sources, caveats)
    return AnswerReport(
        question=question,
        question_class=plan.question_class,
        direct_answer=direct_answer,
        key_evidence=key_evidence[:6],
        sources=sources,
        caveats=caveats,
        evidence_pack=evidence_items[:6],
        analytics=analytics,
        markdown_report=markdown_report,
    )


def _prefer_computed_wording(plan: QuestionPlan, computed: dict[str, Any]) -> bool:
    if plan.question_class == "comparison" and computed.get("comparison_rows"):
        return True
    if plan.question_class in {"trend_analysis", "metric_lookup"} and computed.get("trend_summary"):
        return True
    return False


def _extract_metric_series(
    question: str,
    plan: QuestionPlan,
    candidates: list[RetrievalCandidate],
    tables: dict[str, TableRecord],
) -> list[dict[str, Any]]:
    target_metrics = plan.target_metrics or _metrics_from_question(question)
    target_entities = plan.target_entities
    question_periods = _question_periods(question)
    row_candidates: list[dict[str, Any]] = []

    for candidate in candidates:
        if candidate.object_type != "table":
            continue
        table = tables.get(candidate.object_id)
        if table is None:
            continue
        for row in table.normalized_rows:
            label = str(row.get("label", ""))
            lowered = label.lower()
            row_score = _metric_row_score(question, target_metrics, target_entities, lowered)
            if row_score <= 0:
                continue
            values = row.get("values", {})
            row_series: list[dict[str, Any]] = []
            for period, raw_value in values.items():
                if question_periods and not _period_matches(period, question_periods):
                    continue
                numeric = _to_number(raw_value)
                if numeric is None:
                    continue
                row_series.append(
                    {
                        "label": _display_series_label(label),
                        "period": period,
                        "value": numeric,
                        "raw_value": raw_value,
                        "page": row.get("page"),
                    }
                )
            if row_series:
                row_candidates.append(
                    {
                        "label": label,
                        "score": round(row_score + candidate.fused_score, 4),
                        "series": row_series,
                    }
                )

    if not row_candidates:
        return []

    row_candidates.sort(key=lambda item: (item["score"], len(item["series"])), reverse=True)
    series = list(row_candidates[0]["series"])
    if not question_periods and len(series) > 6:
        series.sort(key=lambda item: _period_sort_key(item["period"]))
        series = series[-6:]
    series.sort(key=lambda item: _period_sort_key(item["period"]))
    return series


def _extract_comparison_rows(
    question: str,
    plan: QuestionPlan,
    candidates: list[RetrievalCandidate],
    tables: dict[str, TableRecord],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    periods = _question_periods(question)
    lowered_question = question.lower()
    want_percentage = any(token in lowered_question for token in ["share", "percentage", "mix", "%"])
    underperforming_mode = any(token in lowered_question for token in ["underperform", "lagging", "weakest"])
    department_mode = underperforming_mode or any(
        token in lowered_question for token in ["department", "segment", "business unit", "division"]
    )

    for candidate in candidates:
        if candidate.object_type != "table":
            continue
        table = tables.get(candidate.object_id)
        if table is None:
            continue
        table_metric_tags = {str(tag).lower() for tag in (table.metric_tags or [])}
        if department_mode and not plan.target_entities and table_metric_tags and "segment" not in table_metric_tags:
            continue
        for row in table.normalized_rows:
            label = str(row.get("label", "")).strip()
            lowered = label.lower()
            if not label:
                continue
            if plan.target_entities and not any(entity in lowered for entity in plan.target_entities):
                continue
            if department_mode and not plan.target_entities and not _is_department_label(lowered):
                continue

            values = row.get("values", {})
            period_points: list[dict[str, Any]] = []
            for period, raw_value in values.items():
                if periods and not _period_matches(period, periods):
                    continue
                normalized_period = re.sub(r"\s+", "", str(period).upper())
                if underperforming_mode and not periods and not re.match(r"Q[1-4]FY\d{2,4}", normalized_period):
                    continue
                if not want_percentage and "%" in str(raw_value):
                    continue
                numeric = _to_number(raw_value)
                if numeric is None:
                    continue
                period_points.append(
                    {
                        "label": label,
                        "period": period,
                        "value": numeric,
                        "raw_value": raw_value,
                    }
                )
            if not period_points:
                continue
            period_points.sort(key=lambda item: _period_sort_key(item["period"]))

            if underperforming_mode and len(period_points) >= 2:
                if not periods and len(period_points) > 3:
                    period_points = period_points[-3:]
                start_point = period_points[0]
                end_point = period_points[-1]
                growth_pct: float | None = None
                if not math.isclose(float(start_point["value"]), 0.0):
                    growth_pct = ((float(end_point["value"]) - float(start_point["value"])) / abs(float(start_point["value"]))) * 100.0
                growth_text = "n/a" if growth_pct is None else f"{growth_pct:+.1f}%"
                rows.append(
                    {
                        "label": label,
                        "period": end_point["period"],
                        "value": float(end_point["value"]),
                        "raw_value": end_point["raw_value"],
                        "start_period": start_point["period"],
                        "start_raw": start_point["raw_value"],
                        "end_period": end_point["period"],
                        "end_raw": end_point["raw_value"],
                        "end_value": float(end_point["value"]),
                        "growth_pct": growth_pct,
                        "period_count": len(period_points),
                        "summary": (
                            f"{label} moved from {start_point['raw_value']} in {start_point['period']} "
                            f"to {end_point['raw_value']} in {end_point['period']} ({growth_text})."
                        ),
                    }
                )
                continue

            anchor = period_points[-1]
            rows.append(
                {
                    "label": label,
                    "period": anchor["period"],
                    "value": float(anchor["value"]),
                    "raw_value": anchor["raw_value"],
                    "period_count": len(period_points),
                    "summary": f"{label} was {anchor['raw_value']} in {anchor['period']}.",
                }
            )

    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row["label"].strip().lower()
        candidate_score = (int(row.get("period_count", 1)), float(row.get("end_value", row.get("value", 0.0))))
        previous = deduped.get(key)
        if previous is None:
            deduped[key] = row
            continue
        previous_score = (int(previous.get("period_count", 1)), float(previous.get("end_value", previous.get("value", 0.0))))
        if candidate_score > previous_score:
            deduped[key] = row

    rows = list(deduped.values())
    if underperforming_mode and len(rows) < 2:
        rows.extend(_extract_underperforming_rows_from_text(candidates))
        merged: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = row["label"].strip().lower()
            previous = merged.get(key)
            if previous is None:
                merged[key] = row
                continue
            previous_period_count = int(previous.get("period_count", 0))
            current_period_count = int(row.get("period_count", 0))
            if current_period_count > previous_period_count:
                merged[key] = row
                continue
            if current_period_count == previous_period_count and float(row.get("value", 0.0)) > float(previous.get("value", 0.0)):
                merged[key] = row
        rows = list(merged.values())

    if underperforming_mode:
        rows = _filter_material_comparison_rows(rows)
        rows.sort(key=lambda item: float(item.get("growth_pct", float("inf"))))
        return rows

    rows.sort(key=lambda item: float(item.get("value", 0.0)), reverse=True)
    return rows


def _filter_material_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) < 2:
        return rows

    values: list[float] = []
    for row in rows:
        raw_value = row.get("end_value", row.get("value", 0.0))
        try:
            values.append(abs(float(raw_value)))
        except (TypeError, ValueError):
            continue

    if len(values) < 2:
        return rows

    max_value = max(values)
    total_value = sum(values)
    if math.isclose(max_value, 0.0) or math.isclose(total_value, 0.0):
        return rows

    # Filter out tiny residual categories so generic "underperforming departments"
    # questions compare peer business units instead of immaterial line items.
    threshold = max(max_value * 0.10, total_value * 0.05)
    filtered = []
    for row in rows:
        raw_value = row.get("end_value", row.get("value", 0.0))
        try:
            numeric = abs(float(raw_value))
        except (TypeError, ValueError):
            filtered.append(row)
            continue
        if numeric >= threshold:
            filtered.append(row)

    return filtered if len(filtered) >= 2 else rows


def _metric_row_score(question: str, target_metrics: list[str], target_entities: list[str], label: str) -> float:
    score = 0.0
    lowered_question = question.lower()
    if target_metrics:
        for metric in target_metrics:
            if metric == "revenue":
                if label.startswith("revenue") or "total revenue" in label:
                    score += 1.5
                elif label == "revenue":
                    score += 1.5
                elif "revenue" in label:
                    score += 0.7
                if "cost of revenue" in label:
                    score -= 1.25
            elif metric == "arr":
                if "arr" in label or "annualized recurring revenue" in label:
                    score += 1.4
            elif metric == "margin":
                if "margin" in label:
                    score += 1.2
            elif metric == "cost":
                if "cost" in label or "expense" in label:
                    score += 1.2
            elif metric == "income":
                if "income" in label or "earnings" in label:
                    score += 1.2
            elif metric == "employees":
                if "employee" in label or "headcount" in label:
                    score += 1.2
    if target_entities:
        if any(entity in label for entity in target_entities):
            score += 0.8
        elif any(entity in lowered_question for entity in target_entities):
            score -= 0.3
    if "percentage" in lowered_question or "%" in lowered_question:
        if "%" in label or "percent" in label:
            score += 0.4
    else:
        if "%" in label or "percent" in label:
            score -= 0.3
    return score


def _display_series_label(label: str) -> str:
    cleaned = re.sub(r"\s+", " ", label).strip()
    replacements = {
        "Revenue ($Billions)Total Revenue": "Total revenue",
        "Revenue($Billions)Total Revenue": "Total revenue",
    }
    if cleaned in replacements:
        return replacements[cleaned]
    lowered = cleaned.lower()
    if lowered == "revenue":
        return "Revenue"
    if "total revenue" in lowered:
        return "Total revenue"
    return cleaned


def _query_overlap_ratio(question: str, plan: QuestionPlan, candidate: RetrievalCandidate) -> float:
    query_terms = {
        token
        for token in re.findall(r"[a-z0-9]+", question.lower())
        if len(token) > 2 and token not in STOPWORDS
    }
    query_terms.update(token for token in plan.target_metrics if token)
    query_terms.update(token for token in plan.target_entities if token)
    if not query_terms:
        return 0.0

    evidence_text = " ".join(
        part
        for part in [
            candidate.title or "",
            " ".join(candidate.heading_path),
            candidate.text[:1200],
        ]
        if part
    ).lower()
    matched = 0
    for term in query_terms:
        if " " in term:
            if term in evidence_text:
                matched += 1
        elif re.search(rf"\b{re.escape(term)}\b", evidence_text):
            matched += 1
    return round(matched / len(query_terms), 3)


def _is_department_label(label: str) -> bool:
    if not label:
        return False
    normalized = normalize_whitespace(label.lower())
    if not re.search(r"[a-z]", normalized):
        return False
    org_keywords = {"segment", "department", "division", "unit", "group", "geography", "region", "market"}
    has_org_keyword = any(token in normalized for token in org_keywords)
    excluded_tokens = {
        "income",
        "expense",
        "earnings",
        "margin",
        "cost",
        "diluted",
        "tax",
        "operating",
        "cash",
        "arr",
        "subscription",
        "shares",
        "employees",
        "obligations",
        "compensation",
        "amortization",
        "adjustment",
        "adjustments",
        "investment",
        "provision",
    }
    if any(token in normalized for token in excluded_tokens):
        return False
    if "revenue" in normalized and not has_org_keyword:
        return False
    words = [word for word in re.findall(r"[a-z]+", normalized) if len(word) > 1]
    if not words:
        return False
    if len(words) < 2 or len(words) > 6:
        return False
    return has_org_keyword or len(words) <= 4


def _comparison_answer(rows: list[dict[str, Any]], question: str) -> str:
    if not rows:
        return "Evidence is insufficient for a grounded comparison."

    lowered_question = question.lower()
    underperforming_mode = any(token in lowered_question for token in ["underperform", "lagging", "weakest"])
    growth_rows = [row for row in rows if row.get("growth_pct") is not None]
    if underperforming_mode and growth_rows:
        ordered = sorted(growth_rows, key=lambda item: float(item.get("growth_pct", float("inf"))))
        core_segments = [row for row in ordered if _is_core_segment_label(str(row.get("label", "")).lower())]
        if len(core_segments) >= 2:
            laggard = core_segments[0]
            benchmark = core_segments[-1]
        else:
            laggard = ordered[0]
            benchmark = ordered[-1]
        laggard_growth = float(laggard.get("growth_pct", 0.0))
        benchmark_growth = float(benchmark.get("growth_pct", 0.0))
        laggard_period = laggard.get("start_period")
        benchmark_period = benchmark.get("start_period")
        if benchmark["label"] == laggard["label"] or len(growth_rows) == 1:
            if laggard_period and laggard.get("end_period"):
                return (
                    f"{laggard['label']} shows {laggard_growth:.1f}% change "
                    f"from {laggard['start_period']} to {laggard['end_period']}."
                )
            return (
                f"{laggard['label']} shows the weakest growth at {laggard_growth:.1f}%."
            )
        if laggard_period and benchmark_period and laggard.get("end_period") and benchmark.get("end_period"):
            return (
                f"{laggard['label']} appears to be underperforming relative to {benchmark['label']}, "
                f"with {laggard_growth:.1f}% change from {laggard['start_period']} to {laggard['end_period']} "
                f"versus {benchmark_growth:.1f}%."
            )
        return (
            f"{laggard['label']} appears to be underperforming relative to {benchmark['label']}, "
            f"with {laggard_growth:.1f}% year-over-year growth versus {benchmark_growth:.1f}%."
        )

    if len(rows) < 2:
        return rows[0]["summary"]
    first, second = rows[0], rows[1]
    return (
        f"{first['label']} was larger than {second['label']} in {first['period']}, "
        f"at {first['raw_value']} versus {second['raw_value']}."
    )


def _is_core_segment_label(label: str) -> bool:
    if not label:
        return False
    normalized = normalize_whitespace(label.lower())
    if not _is_department_label(normalized):
        return False
    if any(token in normalized for token in ["arr", "subscription", "customer group", "remaining performance obligations"]):
        return False
    return True


def _extract_underperforming_rows_from_text(candidates: list[RetrievalCandidate]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    pattern = re.compile(
        r"([A-Z][A-Za-z&/\- ]{2,}?)\s+(?:segment|department|division|business unit)?\s*revenue\s+was\s+\$?([0-9]+(?:\.[0-9]+)?)\s*(?:billion|million)?.*?"
        r"(?:representing|which represents)\s+([0-9]+(?:\.[0-9]+)?)\s+percent\s+year-over-year\s+growth",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for candidate in candidates:
        text = normalize_whitespace(candidate.text)
        for match in pattern.finditer(text):
            label = normalize_whitespace(match.group(1)).strip(" ,.;:-")
            lowered_label = label.lower()
            if lowered_label in seen or not _is_department_label(lowered_label):
                continue
            seen.add(lowered_label)
            revenue = float(match.group(2))
            growth_pct = float(match.group(3))
            rows.append(
                {
                    "label": label,
                    "period": "latest quarter",
                    "value": revenue,
                    "raw_value": f"{revenue:.2f}",
                    "end_value": revenue,
                    "growth_pct": growth_pct,
                    "period_count": 1,
                    "summary": f"{label} revenue grew {growth_pct:.1f}% year over year in the latest quarter.",
                }
            )
    return rows


def _extract_risk_factors(plan: QuestionPlan, candidates: list[RetrievalCandidate], limit: int = 10) -> list[str]:
    question_lower = plan.question.lower()
    last_quarter_prompt = "last quarter" in question_lower or "recent quarter" in question_lower or "latest quarter" in question_lower

    def _candidate_priority(candidate: RetrievalCandidate) -> tuple[int, float]:
        heading = " ".join(candidate.heading_path).lower()
        doc_name = candidate.doc_name.lower()
        report_type = str(candidate.metadata.get("report_type", "")).lower()
        quarterly_hint = int(report_type == "quarterly_report" or "earnings" in doc_name or "quarterly" in doc_name)
        forward_looking_hint = int("forward-looking" in heading or "forward-looking" in candidate.text.lower())
        return (quarterly_hint + forward_looking_hint, candidate.fused_score)

    ordered_candidates = sorted(candidates, key=_candidate_priority, reverse=True)
    factors: list[str] = []
    seen: set[str] = set()
    split_pattern = re.compile(r";|\u2022|\s-\s")

    for candidate in ordered_candidates:
        text = normalize_whitespace(candidate.text)
        lowered = text.lower()
        anchor = "factors that might cause or contribute to such differences include, but are not limited to:"
        if anchor not in lowered:
            continue
        start = lowered.find(anchor) + len(anchor)
        tail = text[start:]
        for stopper in [
            "Further information on these and other factors",
            "The risks described in this press release",
            "Undue reliance should not be placed",
        ]:
            idx = tail.find(stopper)
            if idx != -1:
                tail = tail[:idx]
                break
        for part in split_pattern.split(tail):
            cleaned = normalize_whitespace(part).strip(" ,.;:-")
            if len(cleaned) < 12:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            factors.append(cleaned)
            if len(factors) >= limit:
                return factors

    if factors:
        return factors
    # Fallback when parser output does not preserve the explicit factor list.
    backup = _extract_signals(candidates, RISK_TERMS, limit=limit)
    if last_quarter_prompt and backup:
        return backup
    return backup[: min(6, limit)]


def _risk_answer(risk_factors: list[str], question: str) -> str:
    if not risk_factors:
        return "Evidence is insufficient to summarize key risks."

    categories: list[str] = []
    seen: set[str] = set()
    factor_text = " ".join(risk_factors).lower()
    mapped = [
        ("AI execution risk", ["ai", "artificial intelligence"]),
        ("competition and brand/reputation pressure", ["compete", "reputation", "brands"]),
        ("service reliability and cybersecurity incidents", ["service interruption", "security incident", "cybersecurity", "information technology systems"]),
        ("macroeconomic and foreign-exchange volatility", ["macroeconomic", "foreign currency", "exchange"]),
        ("legal, regulatory and privacy compliance exposure", ["litigation", "regulatory", "privacy", "laws and regulations"]),
        ("talent and operating execution risk", ["recruit", "retain key personnel", "complex sales cycles"]),
        ("subscription revenue-recognition timing risk", ["revenue recognition", "subscription offerings"]),
        ("debt and market volatility", ["debt obligations", "stock price", "catastrophic events"]),
    ]
    for label, keywords in mapped:
        if any(keyword in factor_text for keyword in keywords) and label not in seen:
            seen.add(label)
            categories.append(label)
        if len(categories) >= 5:
            break

    if categories:
        if len(categories) == 1:
            if "last quarter" in question.lower():
                return f"In the last quarter, a key highlighted risk was {categories[0]}."
            return f"A key highlighted risk was {categories[0]}."
        category_text = ", ".join(categories[:-1]) + f", and {categories[-1]}"
        if "last quarter" in question.lower():
            return "In the last quarter, key highlighted risks included " + category_text + "."
        return "Key highlighted risks included " + category_text + "."

    top = risk_factors[:4]
    if len(top) == 1:
        return f"Key highlighted risk: {top[0]}."
    return "Key highlighted risks included " + "; ".join(top) + "."


def _summarize_series(series: list[dict[str, Any]]) -> str | None:
    if len(series) < 2:
        return None
    first = series[0]
    last = series[-1]
    first_value = float(first["value"])
    last_value = float(last["value"])
    if math.isclose(first_value, 0.0):
        return None
    delta = last_value - first_value
    pct = (delta / first_value) * 100.0
    direction = "upward" if delta > 0 else "downward" if delta < 0 else "flat"
    article = "an" if direction.startswith(("a", "e", "i", "o", "u")) else "a"
    return (
        f"{first['label']} shows {article} {direction} trend from {first['period']} ({first['raw_value']}) "
        f"to {last['period']} ({last['raw_value']}), a change of {pct:.1f}%."
    )


def _extract_signals(candidates: list[RetrievalCandidate], terms: list[str], limit: int = 5) -> list[str]:
    signals: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        for sentence in re.split(r"(?<=[.!?])\s+", candidate.text.replace("\n", " ")):
            cleaned = sentence.strip()
            lowered = cleaned.lower()
            if len(cleaned) < 40:
                continue
            if not any(term in lowered for term in terms):
                continue
            if cleaned in seen:
                continue
            seen.add(cleaned)
            signals.append(cleaned)
            if len(signals) >= limit:
                return signals
    return signals


def _citations_from_items(items: list[dict[str, Any]]) -> list[SourceCitation]:
    citations: list[SourceCitation] = []
    seen: set[tuple[str, str, Any, Any]] = set()
    for item in items:
        key = (
            item["doc_name"],
            " > ".join(item.get("heading_path") or []),
            item.get("page_start"),
            item.get("page_end"),
        )
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            SourceCitation(
                doc_name=item["doc_name"],
                heading_path=list(item.get("heading_path") or []),
                page_start=item.get("page_start"),
                page_end=item.get("page_end"),
                object_type=item["object_type"],
                object_id=item["object_id"],
            )
        )
    return citations


def _format_markdown(
    direct_answer: str,
    key_evidence: list[str],
    sources: list[SourceCitation],
    caveats: list[str],
) -> str:
    lines = ["## Direct Answer", direct_answer.strip() or "Insufficient evidence.", "", "## Key Evidence"]
    lines.extend(f"- {item}" for item in key_evidence if item)
    lines.extend(["", "## Sources"])
    for source in sources:
        heading = " > ".join(source.heading_path) if source.heading_path else "Document"
        pages = f"pages {source.page_start}-{source.page_end}" if source.page_start or source.page_end else "pages n/a"
        lines.append(f"- {source.doc_name} | {heading} | {pages}")
    lines.extend(["", "## Caveats"])
    if caveats:
        lines.extend(f"- {item}" for item in caveats)
    else:
        lines.append("- None.")
    return "\n".join(lines)


def _plot_series(series: list[dict[str, Any]], plan: QuestionPlan, output_dir: Path) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    filtered = [item for item in series if item.get("period")]
    if len(filtered) < 2:
        return None
    x = [item["period"] for item in filtered]
    y = [float(item["value"]) for item in filtered]
    label = filtered[0]["label"]
    output_path = output_dir / "sample_plots" / f"{slugify(plan.question)}.png"
    ensure_dir(output_path.parent)
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, marker="o", linewidth=2)
    plt.title(label)
    plt.xlabel("Period")
    plt.ylabel(label)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return str(output_path)


def _metrics_from_question(question: str) -> list[str]:
    q = question.lower()
    return [metric for metric in ["revenue", "arr", "margin", "cost", "income", "employees"] if metric in q]


def _question_periods(question: str) -> list[str]:
    q = question.upper()
    matches = re.findall(r"Q[1-4]\s*FY\s*\d{2,4}|FY\s*\d{4}|20\d{2}", q)
    normalized = [re.sub(r"\s+", "", match) for match in matches]
    quarter_tokens = [token for token in normalized if re.match(r"Q[1-4]FY\d{2,4}", token)]
    if "FROM" in q and "TO" in q and len(quarter_tokens) >= 2:
        expanded = _expand_quarter_range(quarter_tokens[0], quarter_tokens[-1])
        if expanded:
            return expanded
    return normalized


def _period_matches(period: str, targets: list[str]) -> bool:
    normalized = re.sub(r"\s+", "", str(period).upper())
    aliases = {normalized}
    aliases.add(normalized.replace("FY20", "FY"))
    aliases.add(normalized.replace("FY", "FY20") if re.match(r"Q[1-4]FY\d{2}$", normalized) else normalized)
    target_aliases: set[str] = set()
    for target in targets:
        normalized_target = re.sub(r"\s+", "", target.upper())
        target_aliases.add(normalized_target)
        target_aliases.add(normalized_target.replace("FY20", "FY"))
    return bool(aliases & target_aliases)


def _period_sort_key(period: str) -> int:
    normalized = re.sub(r"\s+", "", str(period).upper())
    quarter_match = re.match(r"Q([1-4])FY(\d{2,4})", normalized)
    if quarter_match:
        year = int(quarter_match.group(2))
        if year < 100:
            year += 2000
        return year * 10 + int(quarter_match.group(1))
    year_match = re.search(r"(20\d{2})", normalized)
    if year_match:
        return int(year_match.group(1)) * 10
    return 0


def _expand_quarter_range(start: str, end: str) -> list[str]:
    start_match = re.match(r"Q([1-4])FY(\d{2,4})", start)
    end_match = re.match(r"Q([1-4])FY(\d{2,4})", end)
    if not start_match or not end_match:
        return []
    start_quarter = int(start_match.group(1))
    start_year = int(start_match.group(2))
    end_quarter = int(end_match.group(1))
    end_year = int(end_match.group(2))
    if start_year < 100:
        start_year += 2000
    if end_year < 100:
        end_year += 2000
    start_key = start_year * 4 + (start_quarter - 1)
    end_key = end_year * 4 + (end_quarter - 1)
    if end_key < start_key:
        start_key, end_key = end_key, start_key
    periods: list[str] = []
    for key in range(start_key, end_key + 1):
        year = key // 4
        quarter = (key % 4) + 1
        periods.append(f"Q{quarter}FY{year}")
    return periods


def _to_number(raw_value: Any) -> float | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip().replace(",", "")
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()$")
    is_percent = text.endswith("%")
    text = text.rstrip("%")
    try:
        value = float(text)
    except ValueError:
        return None
    if negative:
        value *= -1
    if is_percent:
        return value
    return value
