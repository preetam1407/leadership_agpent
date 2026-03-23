from __future__ import annotations

import hashlib
import json
import logging
import math
import pickle
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

LOGGER = logging.getLogger("leadership_agent")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def stable_id(*parts: Any, length: int = 16) -> str:
    raw = "||".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]


def slugify(text: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return slug[:max_length] or "item"


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_pickle(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def dedupe_preserve_order(items: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    ordered: list[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def split_sentences(text: str) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [part.strip() for part in parts if part.strip()]


try:  # pragma: no cover - optional dependency
    import tiktoken
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None

_ENCODING_CACHE: dict[str, Any] = {}


def token_count(text: str, model: str = "gpt-5.4-mini") -> int:
    if not text:
        return 0
    if tiktoken is not None:
        try:
            encoding = _ENCODING_CACHE.get(model)
            if encoding is None:
                encoding = tiktoken.encoding_for_model(model)
                _ENCODING_CACHE[model] = encoding
        except Exception:
            try:
                encoding = _ENCODING_CACHE.get("cl100k_base")
                if encoding is None:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    _ENCODING_CACHE["cl100k_base"] = encoding
            except Exception:
                encoding = None
        if encoding is not None:
            try:
                return len(encoding.encode(text))
            except Exception:
                pass
    return len(re.findall(r"\S+", text))


def chunk_paragraphs(paragraphs: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current_parts, current_tokens
        if current_parts:
            chunks.append("\n\n".join(current_parts).strip())
        current_parts = []
        current_tokens = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        paragraph_tokens = token_count(paragraph)
        if paragraph_tokens > max_tokens:
            sentences = split_sentences(paragraph)
            if not sentences:
                sentences = [paragraph]
            for sentence in sentences:
                sentence_tokens = token_count(sentence)
                if current_tokens and current_tokens + sentence_tokens > max_tokens:
                    flush()
                current_parts.append(sentence)
                current_tokens += sentence_tokens
            continue

        if current_tokens and current_tokens + paragraph_tokens > max_tokens:
            overlap_parts: list[str] = []
            if overlap_tokens > 0 and current_parts:
                running = 0
                for part in reversed(current_parts):
                    overlap_parts.insert(0, part)
                    running += token_count(part)
                    if running >= overlap_tokens:
                        break
            flush()
            current_parts = overlap_parts
            current_tokens = sum(token_count(part) for part in current_parts)

        current_parts.append(paragraph)
        current_tokens += paragraph_tokens

    flush()
    return [chunk for chunk in chunks if chunk]


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


def cosine_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32)
    arr = matrix.astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def openai_client(runtime_cfg: Any) -> Any | None:
    api_key = getattr(runtime_cfg, "openai_api_key", "")
    if not api_key:
        return None
    try:  # pragma: no cover - requires external package and key
        from openai import OpenAI
    except Exception:
        return None
    kwargs = {"api_key": api_key}
    base_url = getattr(runtime_cfg, "openai_base_url", "")
    if base_url:
        kwargs["base_url"] = base_url
    timeout = getattr(runtime_cfg, "llm_timeout_sec", 60)
    kwargs["timeout"] = timeout
    return OpenAI(**kwargs)


def openai_json_completion(
    runtime_cfg: Any,
    model_candidates: list[str],
    system_prompt: str,
    user_prompt: str,
    schema_name: str,
    schema: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    client = openai_client(runtime_cfg)
    usage = {"model": None, "input_tokens": 0, "output_tokens": 0, "errors": []}
    if client is None:
        return None, usage

    for model in dedupe_preserve_order(model_candidates):
        if not model:
            continue
        try:  # pragma: no cover - requires network and key
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "strict": True, "schema": schema},
                },
                temperature=0,
            )
            content = completion.choices[0].message.content or ""
            parsed = parse_json_object(content)
            prompt_tokens = int(getattr(completion.usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(completion.usage, "completion_tokens", 0) or 0)
            usage = {
                "model": model,
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "errors": usage["errors"],
            }
            if parsed is not None:
                return parsed, usage
        except Exception as exc:  # pragma: no cover - requires network and key
            usage["errors"].append(f"{model}: {exc}")

    return None, usage

def min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}
