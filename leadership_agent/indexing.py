from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from leadership_agent.config import AppConfig
from leadership_agent.ingest import load_parsed_artifacts
from leadership_agent.models import ChunkRecord, DocumentRecord, SectionRecord, TableRecord
from leadership_agent.utils import (
    cosine_normalize,
    ensure_dir,
    load_pickle,
    save_pickle,
    write_json,
)

TOKEN_RE = re.compile(r"[A-Za-z0-9%\.\-$]+")


class DenseArtifact:
    def __init__(self, name: str, config: AppConfig, index_dir: Path):
        self.name = name
        self.config = config
        self.index_dir = index_dir
        self.ids: list[str] = []
        self.embeddings: np.ndarray | None = None
        self.backend: str = "tfidf"
        self.model: str = config.models.embedding
        self.vectorizer: Any = None
        self.local_embedder: Any = None
        self.faiss_index: Any = None
        self.usage: dict[str, Any] = {}

    @property
    def base_path(self) -> Path:
        return self.index_dir / "dense" / self.name

    def build(self, ids: list[str], texts: list[str]) -> dict[str, Any]:
        self.ids = ids
        ensure_dir(self.base_path.parent)

        vectors, usage = self._embed_local_texts(texts)
        if vectors is not None:
            self.backend = "fastembed"
            self.model = self.config.models.embedding
            self.embeddings = vectors
            self.usage = usage
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.backend = "tfidf"
            self.model = "sklearn_tfidf"
            self.vectorizer = TfidfVectorizer(max_features=8000)
            matrix = self.vectorizer.fit_transform(texts).toarray().astype(np.float32)
            self.embeddings = cosine_normalize(matrix)
            self.usage = usage
            save_pickle(self.base_path.with_suffix(".vectorizer.pkl"), self.vectorizer)

        if self.embeddings is None:
            raise RuntimeError(f"Unable to build dense artifact for {self.name}")

        if self.backend != "tfidf":
            vectorizer_path = self.base_path.with_suffix(".vectorizer.pkl")
            if vectorizer_path.exists():
                vectorizer_path.unlink()

        np.save(self.base_path.with_suffix(".npy"), self.embeddings)
        write_json(self.base_path.with_suffix(".ids.json"), ids)
        write_json(
            self.base_path.with_suffix(".meta.json"),
            {"name": self.name, "backend": self.backend, "model": self.model, "usage": self.usage},
        )
        self._build_faiss()
        return {"backend": self.backend, "model": self.model, "count": len(ids), "usage": self.usage}

    def load(self) -> None:
        meta_path = self.base_path.with_suffix(".meta.json")
        ids_path = self.base_path.with_suffix(".ids.json")
        emb_path = self.base_path.with_suffix(".npy")
        if not (meta_path.exists() and ids_path.exists() and emb_path.exists()):
            raise FileNotFoundError(f"Missing dense artifacts for {self.name}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.backend = meta["backend"]
        self.model = meta["model"]
        self.usage = meta.get("usage", {})
        self.ids = json.loads(ids_path.read_text(encoding="utf-8"))
        self.embeddings = np.load(emb_path)
        if self.backend == "tfidf":
            self.vectorizer = load_pickle(self.base_path.with_suffix(".vectorizer.pkl"))
        self._load_faiss()

    def query(self, text: str, top_k: int, allowed_ids: set[str] | None = None) -> dict[str, float]:
        if self.embeddings is None:
            raise RuntimeError(f"Dense artifact {self.name} is not loaded")
        query_vector = self._embed_query(text)
        if query_vector is None:
            return {}

        requested = min(max(top_k, 1) * self.config.retrieval.search_multiplier, len(self.ids))
        scores: dict[str, float] = {}
        if self.faiss_index is not None:
            values, indices = self.faiss_index.search(np.asarray(query_vector.reshape(1, -1), dtype=np.float32), requested)
            for score, idx in zip(values[0], indices[0]):
                if idx < 0:
                    continue
                object_id = self.ids[int(idx)]
                if allowed_ids is not None and object_id not in allowed_ids:
                    continue
                scores[object_id] = float(score)
        else:
            raw_scores = self.embeddings @ query_vector
            ranking = np.argsort(raw_scores)[::-1][:requested]
            for idx in ranking:
                object_id = self.ids[int(idx)]
                if allowed_ids is not None and object_id not in allowed_ids:
                    continue
                scores[object_id] = float(raw_scores[idx])

        if allowed_ids is not None and len(scores) < top_k:
            id_to_index = {object_id: idx for idx, object_id in enumerate(self.ids)}
            extra_scores = {
                object_id: float(self.embeddings[id_to_index[object_id]] @ query_vector)
                for object_id in allowed_ids
                if object_id in id_to_index
            }
            scores.update(extra_scores)

        ranked = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k])
        return ranked

    def _embed_query(self, text: str) -> np.ndarray | None:
        if self.backend == "fastembed":
            vectors, _ = self._embed_local_texts([text])
            if vectors is None:
                return None
            return vectors[0]
        if self.vectorizer is None:
            return None
        arr = self.vectorizer.transform([text]).toarray().astype(np.float32)
        return cosine_normalize(arr)[0]

    def _embed_local_texts(self, texts: list[str]) -> tuple[np.ndarray | None, dict[str, Any]]:
        usage = {"model": self.config.models.embedding, "errors": []}
        try:
            embedder = self._get_local_embedder()
            vectors = list(embedder.embed(texts, batch_size=self.config.models.embedding_batch_size))
            if not vectors:
                usage["errors"].append("fastembed returned no vectors")
                return None, usage
            matrix = np.asarray(vectors, dtype=np.float32)
            return cosine_normalize(matrix), usage
        except Exception as exc:
            usage["errors"].append(str(exc))
            return None, usage

    def _get_local_embedder(self) -> Any:
        if self.local_embedder is not None:
            return self.local_embedder
        from fastembed import TextEmbedding

        cache_dir = ensure_dir(Path(__file__).resolve().parents[1] / ".cache" / "fastembed")
        self.local_embedder = TextEmbedding(model_name=self.config.models.embedding, cache_dir=str(cache_dir))
        return self.local_embedder

    def _build_faiss(self) -> None:
        if self.embeddings is None:
            return
        try:
            import faiss  # type: ignore
        except Exception:
            self.faiss_index = None
            return
        index = faiss.IndexFlatIP(self.embeddings.shape[1])
        index.add(np.ascontiguousarray(self.embeddings.astype(np.float32)))
        self.faiss_index = index
        faiss.write_index(index, str(self.base_path.with_suffix(".faiss")))

    def _load_faiss(self) -> None:
        faiss_path = self.base_path.with_suffix(".faiss")
        if not faiss_path.exists():
            self.faiss_index = None
            return
        try:
            import faiss  # type: ignore
        except Exception:
            self.faiss_index = None
            return
        self.faiss_index = faiss.read_index(str(faiss_path))


class MetadataStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def rebuild(
        self,
        documents: list[DocumentRecord],
        sections: list[SectionRecord],
        chunks: list[ChunkRecord],
        tables: list[TableRecord],
    ) -> None:
        import duckdb

        ensure_dir(self.db_path.parent)
        if self.db_path.exists():
            self.db_path.unlink()
        conn = duckdb.connect(str(self.db_path))
        conn.execute(
            """
            CREATE TABLE documents (
                doc_id TEXT,
                doc_name TEXT,
                source_path TEXT,
                report_type TEXT,
                inferred_period TEXT,
                inferred_year INTEGER,
                inferred_quarter INTEGER,
                latest_sort_key INTEGER,
                likely_business_areas TEXT,
                metadata TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE sections (
                section_id TEXT,
                doc_id TEXT,
                doc_name TEXT,
                heading TEXT,
                heading_path TEXT,
                page_start INTEGER,
                page_end INTEGER,
                section_type TEXT,
                text TEXT,
                metadata TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE chunks (
                chunk_id TEXT,
                section_id TEXT,
                doc_id TEXT,
                doc_name TEXT,
                heading_path TEXT,
                page_start INTEGER,
                page_end INTEGER,
                chunk_type TEXT,
                chunk_index INTEGER,
                text TEXT,
                metadata TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE tables (
                table_id TEXT,
                section_id TEXT,
                doc_id TEXT,
                doc_name TEXT,
                page_start INTEGER,
                page_end INTEGER,
                title TEXT,
                heading_path TEXT,
                table_markdown TEXT,
                table_text_summary TEXT,
                metric_tags TEXT,
                normalized_rows TEXT,
                metadata TEXT
            );
            """
        )

        document_rows = [
            (
                row.doc_id,
                row.doc_name,
                row.source_path,
                row.report_type,
                row.inferred_period,
                row.inferred_year,
                row.inferred_quarter,
                row.latest_sort_key,
                json.dumps(row.likely_business_areas),
                json.dumps(row.metadata),
            )
            for row in documents
        ]
        section_rows = [
            (
                row.section_id,
                row.doc_id,
                row.doc_name,
                row.heading,
                json.dumps(row.heading_path),
                row.page_start,
                row.page_end,
                row.section_type,
                row.text,
                json.dumps(row.metadata),
            )
            for row in sections
        ]
        chunk_rows = [
            (
                row.chunk_id,
                row.section_id,
                row.doc_id,
                row.doc_name,
                json.dumps(row.heading_path),
                row.page_start,
                row.page_end,
                row.chunk_type,
                row.chunk_index,
                row.text,
                json.dumps(row.metadata),
            )
            for row in chunks
        ]
        table_rows = [
            (
                row.table_id,
                row.section_id,
                row.doc_id,
                row.doc_name,
                row.page_start,
                row.page_end,
                row.title,
                json.dumps(row.heading_path),
                row.table_markdown,
                row.table_text_summary,
                json.dumps(row.metric_tags),
                json.dumps(row.normalized_rows),
                json.dumps(row.metadata),
            )
            for row in tables
        ]

        if document_rows:
            conn.executemany("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", document_rows)
        if section_rows:
            conn.executemany("INSERT INTO sections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", section_rows)
        if chunk_rows:
            conn.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", chunk_rows)
        if table_rows:
            conn.executemany("INSERT INTO tables VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", table_rows)
        conn.close()

    def connect(self) -> Any:
        import duckdb

        return duckdb.connect(str(self.db_path), read_only=True)

    def filter_doc_ids(
        self,
        prioritize_latest: bool = False,
        report_types: list[str] | None = None,
        time_focus: str | None = None,
    ) -> list[str]:
        conn = self.connect()
        clauses = ["1=1"]
        params: list[Any] = []
        if report_types:
            placeholders = ",".join(["?"] * len(report_types))
            clauses.append(f"report_type IN ({placeholders})")
            params.extend(report_types)
        if time_focus:
            clauses.append("(lower(coalesce(inferred_period, '')) = lower(?) OR cast(coalesce(inferred_year, 0) as varchar) = ?)")
            params.extend([time_focus, re.sub(r"[^0-9]", "", time_focus)])
        sql = f"SELECT doc_id FROM documents WHERE {' AND '.join(clauses)} ORDER BY latest_sort_key DESC"
        if prioritize_latest:
            sql += " LIMIT 3"
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [row[0] for row in rows]

    def section_ids_for_docs(self, doc_ids: list[str]) -> list[str]:
        if not doc_ids:
            return []
        conn = self.connect()
        placeholders = ",".join(["?"] * len(doc_ids))
        rows = conn.execute(f"SELECT section_id FROM sections WHERE doc_id IN ({placeholders})", doc_ids).fetchall()
        conn.close()
        return [row[0] for row in rows]

    def chunk_ids_for_sections(self, section_ids: list[str]) -> list[str]:
        if not section_ids:
            return []
        conn = self.connect()
        placeholders = ",".join(["?"] * len(section_ids))
        rows = conn.execute(f"SELECT chunk_id FROM chunks WHERE section_id IN ({placeholders})", section_ids).fetchall()
        conn.close()
        return [row[0] for row in rows]

    def table_ids_for_docs(self, doc_ids: list[str]) -> list[str]:
        if not doc_ids:
            return []
        conn = self.connect()
        placeholders = ",".join(["?"] * len(doc_ids))
        rows = conn.execute(f"SELECT table_id FROM tables WHERE doc_id IN ({placeholders})", doc_ids).fetchall()
        conn.close()
        return [row[0] for row in rows]


def build_indexes(config: AppConfig) -> dict[str, Any]:
    documents, sections, chunks, tables = load_parsed_artifacts(config)
    if not documents or not sections or not chunks:
        raise RuntimeError("Parsed artifacts are missing or empty. Run ingest first.")

    ensure_dir(config.index_dir / "bm25")
    ensure_dir(config.index_dir / "dense")
    ensure_dir(config.index_dir / "metadata")

    _save_bm25_bundle(config.index_dir / "bm25" / "sections.pkl", [row.section_id for row in sections], [row.text for row in sections])
    _save_bm25_bundle(config.index_dir / "bm25" / "chunks.pkl", [row.chunk_id for row in chunks], [row.text for row in chunks])
    _save_bm25_bundle(
        config.index_dir / "bm25" / "tables.pkl",
        [row.table_id for row in tables],
        [row.table_text_summary or row.table_markdown for row in tables],
    )

    section_dense = DenseArtifact("sections", config, config.index_dir)
    chunk_dense = DenseArtifact("chunks", config, config.index_dir)
    table_dense = DenseArtifact("tables", config, config.index_dir)
    section_info = section_dense.build([row.section_id for row in sections], [row.text for row in sections])
    chunk_info = chunk_dense.build([row.chunk_id for row in chunks], [row.text for row in chunks])
    table_info = table_dense.build(
        [row.table_id for row in tables],
        [row.table_text_summary or row.table_markdown for row in tables] or [""],
    ) if tables else {"backend": "none", "model": None, "count": 0, "usage": {}}

    store = MetadataStore(config.index_dir / "metadata" / "leadership.duckdb")
    store.rebuild(documents, sections, chunks, tables)

    return {
        "documents": len(documents),
        "sections": len(sections),
        "chunks": len(chunks),
        "tables": len(tables),
        "dense": {
            "sections": section_info,
            "chunks": chunk_info,
            "tables": table_info,
        },
        "metadata_db": str(store.db_path),
    }


def load_bm25_bundle(path: str | Path) -> tuple[list[str], list[list[str]]]:
    payload = load_pickle(path)
    return payload["ids"], payload["tokens"]


def bm25_scores(path: str | Path, query: str, allowed_ids: set[str] | None = None, top_k: int = 10) -> dict[str, float]:
    from rank_bm25 import BM25Okapi

    ids, tokens = load_bm25_bundle(path)
    scorer = BM25Okapi(tokens)
    query_tokens = tokenize(query)
    raw_scores = scorer.get_scores(query_tokens)
    scores: dict[str, float] = {}
    for idx, score in enumerate(raw_scores):
        object_id = ids[idx]
        if allowed_ids is not None and object_id not in allowed_ids:
            continue
        scores[object_id] = float(score)
    ranked = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k])
    return ranked


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _save_bm25_bundle(path: Path, ids: list[str], texts: list[str]) -> None:
    tokens = [tokenize(text) for text in texts]
    save_pickle(path, {"ids": ids, "tokens": tokens})
