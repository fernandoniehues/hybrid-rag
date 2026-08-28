"""Geração da resposta a partir dos trechos recuperados.

Duas regras que separam RAG útil de RAG que inventa:
  - o modelo responde SÓ com o que está nos trechos;
  - toda afirmação carrega a citação [n] do trecho que a sustenta.

Sem isso, o sistema fica confiante e errado — o pior modo de falha possível
para uma base de conhecimento.
"""
from app.llm import complete
from app.search import Hit

# O prompt é escrito em inglês de propósito: com as instruções em português,
# o modelo respondia em português mesmo quando a pergunta vinha em inglês —
# o idioma do system prompt vazava para a resposta.
SYSTEM = """Answer the question using ONLY the provided excerpts.

Rules:
- ALWAYS answer in the same language as the question. If the question is in
  English, answer in English. If it is in Portuguese, answer in Portuguese.
  The language of the excerpts does not matter.
- If the excerpts do not contain the answer, say exactly (in the question's
  language): "I could not find this information in the indexed documents."
  Never fill the gap with your own knowledge.
- Cite the source of every claim with the excerpt number: [1], [2].
- Be direct. No preamble, no restating the question."""


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
