"""Camada de embeddings, isolada do resto da aplicação.

O provedor é escolhido por configuração. Trocar de fornecedor ou de modelo
não toca em nenhum outro módulo — só a dimensão precisa acompanhar o schema.
"""
import logging

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

log = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    pass


@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20), reraise=True)
def _openai_embed(texts: list[str]) -> list[list[float]]:
    s = get_settings()
    if not s.openai_api_key:
        raise EmbeddingError("OPENAI_API_KEY não configurada")

    r = httpx.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {s.openai_api_key}"},
        json={"model": s.embedding_model, "input": texts},
        timeout=60,
    )
    if r.status_code == 429:
        raise EmbeddingError("rate limit do provedor de embeddings")
    r.raise_for_status()
    data = sorted(r.json()["data"], key=lambda d: d["index"])
    return [d["embedding"] for d in data]


_PROVIDERS = {"openai": _openai_embed}


def embed(texts: list[str], batch: int = 96) -> list[list[float]]:
    """Vetoriza em lotes — uma chamada por text seria lenta e cara."""
    s = get_settings()
    fn = _PROVIDERS.get(s.embedding_provider)
    if fn is None:
        raise EmbeddingError(f"provedor desconhecido: {s.embedding_provider}")

    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        batch_slice = texts[i:i + batch]
        out.extend(fn(batch_slice))
        log.info("embeddings %s/%s", min(i + batch, len(texts)), len(texts))
    return out
