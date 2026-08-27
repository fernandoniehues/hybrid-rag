"""Busca híbrida: vetorial + full-text, fundidas por Reciprocal Rank Fusion.

Por que duas pernas:
  - A busca vetorial acha por significado, mas erra com nome próprio, sigla,
    número de versão e termo raro — coisas cujo vetor não carrega o que o
    usuário quis dizer.
  - A busca lexical acerta o termo exato, mas não entende sinônimo nem
    paráfrase.

RRF combina os dois rankings sem precisar normalizar escores de escalas
diferentes (distância cosseno x ts_rank): cada documento pontua 1/(k+posição)
em cada ranking, e os pontos somam. Um documento bem colocado nas duas pernas
vence um documento excelente em apenas uma.
"""
from dataclasses import dataclass

from app.config import get_settings
from app.db import cursor
from app.embeddings import embed


@dataclass
class Hit:
    chunk_id: int
    document_id: int
    source: str
    title: str
    heading: str
    content: str
    score: float
    vector_rank: int | None = None
    text_rank: int | None = None


def _vector_leg(query: str, limit: int) -> list[tuple[int, int]]:
    """Retorna [(chunk_id, posição)] por similaridade de cosseno."""
    vec = embed([query])[0]
    with cursor() as cur:
        cur.execute(
            """
            SELECT id
              FROM chunks
             WHERE embedding IS NOT NULL
             ORDER BY embedding <=> %s::vector
             LIMIT %s
            """,
            (str(vec), limit),
        )
        return [(row[0], i + 1) for i, row in enumerate(cur.fetchall())]


def _text_leg(query: str, limit: int) -> list[tuple[int, int]]:
    """Retorna [(chunk_id, posição)] por relevância lexical."""
    s = get_settings()
    with cursor() as cur:
        cur.execute(
            """
            SELECT id
              FROM chunks
             WHERE tsv @@ websearch_to_tsquery(%s, %s)
             ORDER BY ts_rank_cd(tsv, websearch_to_tsquery(%s, %s)) DESC
             LIMIT %s
            """,
            (s.fts_language, query, s.fts_language, query, limit),
        )
        return [(row[0], i + 1) for i, row in enumerate(cur.fetchall())]


def _fuse(legs: list[list[tuple[int, int]]], k: int) -> dict[int, float]:
    scores: dict[int, float] = {}
    for leg in legs:
        for chunk_id, rank in leg:
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def _hydrate(ids: list[int]) -> dict[int, dict]:
    if not ids:
        return {}
    with cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.document_id, d.source, d.title, c.heading, c.content
              FROM chunks c
              JOIN documents d ON d.id = c.document_id
             WHERE c.id = ANY(%s)
            """,
            (ids,),
        )
        return {
            r[0]: {"document_id": r[1], "source": r[2], "title": r[3],
                   "heading": r[4] or "", "content": r[5]}
            for r in cur.fetchall()
        }


def search(query: str, limit: int | None = None, mode: str = "hybrid") -> list[Hit]:
    """mode: 'vector' | 'text' | 'hybrid'. Os três existem para a avaliação
    poder medir o ganho de cada estratégia com o mesmo conjunto de perguntas."""
    s = get_settings()
    limit = limit or s.top_k
    n = s.candidates_per_leg

    vec_leg = _vector_leg(query, n) if mode in ("vector", "hybrid") else []
    txt_leg = _text_leg(query, n) if mode in ("text", "hybrid") else []

    if mode == "vector":
        fused = {cid: 1.0 / (s.rrf_k + rank) for cid, rank in vec_leg}
    elif mode == "text":
        fused = {cid: 1.0 / (s.rrf_k + rank) for cid, rank in txt_leg}
    else:
        fused = _fuse([vec_leg, txt_leg], s.rrf_k)

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    rows = _hydrate([cid for cid, _ in ordered])
    vpos = dict(vec_leg)
    tpos = dict(txt_leg)

    hits = []
    for cid, score in ordered:
        row = rows.get(cid)
        if not row:
            continue
        hits.append(Hit(chunk_id=cid, score=score, vector_rank=vpos.get(cid),
                        text_rank=tpos.get(cid), **row))
    return hits
