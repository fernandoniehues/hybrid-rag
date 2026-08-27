"""Ingestão: lê arquivos, quebra em chunks, vetoriza e grava.

Idempotente por hash: reprocessar a mesma pasta não duplica nem re-embeda o
que não mudou — embedding custa dinheiro e tempo.
"""
import hashlib
import logging
from pathlib import Path

from app.chunking import chunk_markdown
from app.config import get_settings
from app.db import cursor
from app.embeddings import embed

log = logging.getLogger(__name__)
EXTENSIONS = {".md", ".markdown", ".txt", ".rst"}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _title_of(path: Path, text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem.replace("-", " ").replace("_", " ")


def ingest_file(path: Path) -> dict:
    s = get_settings()
    text = path.read_text(encoding="utf-8", errors="ignore")
    digest = _sha256(text)
    source = str(path)

    with cursor() as cur:
        cur.execute("SELECT id, sha256 FROM documents WHERE source = %s", (source,))
        row = cur.fetchone()
        if row and row[1] == digest:
            return {"source": source, "status": "inalterado", "chunks": 0}

        if row:  # conteúdo mudou: apaga os chunks antigos e reindexa
            cur.execute("DELETE FROM chunks WHERE document_id = %s", (row[0],))
            cur.execute("UPDATE documents SET sha256 = %s WHERE id = %s", (digest, row[0]))
            doc_id = row[0]
        else:
            cur.execute(
                "INSERT INTO documents (source, title, sha256) VALUES (%s,%s,%s) RETURNING id",
                (source, _title_of(path, text), digest),
            )
            doc_id = cur.fetchone()[0]

    chunks = chunk_markdown(text, s.chunk_size, s.chunk_overlap)
    if not chunks:
        return {"source": source, "status": "vazio", "chunks": 0}

    vectors = embed([f"{c.heading}\n{c.content}".strip() for c in chunks])

    with cursor() as cur:
        for chunk, vector in zip(chunks, vectors):
            cur.execute(
                """
                INSERT INTO chunks (document_id, ordinal, heading, content, embedding)
                VALUES (%s,%s,%s,%s,%s::vector)
                ON CONFLICT (document_id, ordinal) DO UPDATE
                   SET content = EXCLUDED.content, embedding = EXCLUDED.embedding
                """,
                (doc_id, chunk.ordinal, chunk.heading, chunk.content, str(vector)),
            )

    return {"source": source, "status": "indexado", "chunks": len(chunks)}


def ingest_directory(directory: str) -> dict:
    base = Path(directory)
    if not base.exists():
        raise FileNotFoundError(directory)

    files = [p for p in sorted(base.rglob("*")) if p.suffix.lower() in EXTENSIONS]
    results = []
    for path in files:
        try:
            results.append(ingest_file(path))
        except Exception as e:
            log.error("falha ao indexar %s: %s", path, e)
            results.append({"source": str(path), "status": f"erro: {e}", "chunks": 0})

    return {
        "arquivos": len(files),
        "indexados": sum(1 for r in results if r["status"] == "indexado"),
        "inalterados": sum(1 for r in results if r["status"] == "inalterado"),
        "chunks": sum(r["chunks"] for r in results),
        "detalhes": results,
    }
