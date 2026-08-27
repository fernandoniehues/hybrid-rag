"""Pool de conexões e schema. Sem ORM — SQL direto, migração idempotente."""
import logging
from contextlib import contextmanager

from psycopg_pool import ConnectionPool

from app.config import get_settings

log = logging.getLogger(__name__)
_pool: ConnectionPool | None = None

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT NOT NULL UNIQUE,
    title       TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal      INT NOT NULL,
    heading      TEXT,
    content      TEXT NOT NULL,
    embedding    vector(%(dim)s),
    tsv          tsvector GENERATED ALWAYS AS (
                     to_tsvector('%(lang)s', coalesce(heading,'') || ' ' || content)
                 ) STORED,
    UNIQUE (document_id, ordinal)
);

-- Índice vetorial: HNSW com distância cosseno.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Índice de full-text, a outra perna da busca híbrida.
CREATE INDEX IF NOT EXISTS chunks_tsv_idx
    ON chunks USING gin (tsv);
"""


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        s = get_settings()
        _pool = ConnectionPool(s.database_url, min_size=1, max_size=8, open=True)
    return _pool


@contextmanager
def cursor():
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            yield cur


def init_db() -> None:
    """Cria extensões, tabelas e índices. Seguro rodar várias vezes."""
    s = get_settings()
    with cursor() as cur:
        cur.execute(SCHEMA % {"dim": s.embedding_dim, "lang": s.fts_language})
    log.info("schema pronto (dim=%s, fts=%s)", s.embedding_dim, s.fts_language)


def stats() -> dict:
    with cursor() as cur:
        cur.execute("SELECT count(*) FROM documents")
        docs = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM chunks")
        chunks = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NULL")
        pending = cur.fetchone()[0]
    return {"documents": docs, "chunks": chunks, "chunks_sem_embedding": pending}
