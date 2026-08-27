"""Reranking dos candidatos recuperados.

A busca é otimizada para trazer muitos candidatos plausíveis rápido; ela não
julga bem qual deles responde à pergunta. O reranker faz a leitura cara em
poucos candidatos: recebe pergunta e trecho, devolve uma nota de relevância.

Isso desloca o custo para onde ele rende — 20 avaliações curtas em vez de
comparar a pergunta com o corpus inteiro.
"""
import json
import logging
from concurrent.futures import ThreadPoolExecutor

from app.config import get_settings
from app.llm import complete
from app.search import Hit

log = logging.getLogger(__name__)

SYSTEM = (
    "Você avalia se um trecho de documentação responde a uma pergunta. "
    "Responda APENAS com JSON no formato {\"score\": <0 a 10>}. "
    "10 = responde diretamente e por completo. "
    "5 = tem contexto relacionado, mas não responde. "
    "0 = não tem relação com a pergunta."
)


def _score(question: str, hit: Hit) -> float:
    prompt = (
        f"PERGUNTA:\n{question}\n\n"
        f"TRECHO (de '{hit.title}', seção '{hit.heading}'):\n{hit.content[:1500]}"
    )
    try:
        raw = complete(SYSTEM, prompt, model=get_settings().rerank_model, max_tokens=32)
        return float(json.loads(raw[raw.find("{"):raw.rfind("}") + 1])["score"])
    except Exception as e:
        # Reranking é melhoria, não requisito: se o provedor falhar, a busca
        # híbrida sozinha ainda responde. Degradar é melhor que derrubar.
        log.warning("rerank falhou no chunk %s: %s", hit.chunk_id, e)
        return -1.0


def rerank(question: str, hits: list[Hit], top_k: int, workers: int = 8) -> list[Hit]:
    """Avalia os candidatos em paralelo.

    Sequencial, o reranking dominava a latência: uma chamada curta por
    candidato, uma depois da outra. As chamadas são independentes entre si,
    então rodam concorrentes — o tempo passa a ser o da mais lenta, não a
    soma de todas.
    """
    if not hits:
        return []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        scores = list(pool.map(lambda h: _score(question, h), hits))
    scored = list(zip(hits, scores))
    if all(s < 0 for _, s in scored):
        return hits[:top_k]
    # Empate mantém a ordem da fusão, que já é um sinal de qualidade.
    scored.sort(key=lambda hs: hs[1], reverse=True)
    out = []
    for hit, score in scored[:top_k]:
        if score >= 0:
            hit.score = score
        out.append(hit)
    return out
