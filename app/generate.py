"""Geração da resposta a partir dos trechos recuperados.

Duas regras que separam RAG útil de RAG que inventa:
  - o modelo responde SÓ com o que está nos trechos;
  - toda afirmação carrega a citação [n] do trecho que a sustenta.

Sem isso, o sistema fica confiante e errado — o pior modo de falha possível
para uma base de conhecimento.
"""
from app.llm import complete
from app.search import Hit

SYSTEM = """Você responde perguntas usando APENAS os trechos fornecidos.

Regras:
- Se os trechos não contêm a resposta, diga exatamente: "Não encontrei essa
  informação nos documentos indexados." Não complete com conhecimento próprio.
- Cite a fonte de cada afirmação com o número do trecho, assim: [1], [2].
- Seja direto. Sem introdução e sem repetir a pergunta.
- Responda no idioma da pergunta."""


def build_context(hits: list[Hit]) -> str:
    partes = []
    for i, h in enumerate(hits, 1):
        cabecalho = f"[{i}] {h.title}" + (f" — {h.heading}" if h.heading else "")
        partes.append(f"{cabecalho}\n{h.content}")
    return "\n\n---\n\n".join(partes)


def answer(question: str, hits: list[Hit]) -> str:
    if not hits:
        return "Não encontrei essa informação nos documentos indexados."
    user = f"TRECHOS:\n\n{build_context(hits)}\n\nPERGUNTA: {question}"
    return complete(SYSTEM, user, max_tokens=700)
