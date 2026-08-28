"""Agente de RAG corretivo, orquestrado com LangGraph.

O pipeline simples de RAG tem um ponto cego: ele confia no que recuperou.
Se a busca traz material ruim — pergunta mal formulada, termo que não existe
no corpus, assunto ausente — o gerador recebe lixo e responde mesmo assim.

Este grafo fecha esse buraco. Depois de recuperar, um nó avalia se o material
responde à pergunta. Se não responde, o agente reescreve a consulta e tenta de
novo, com um teto de tentativas. Se ainda assim não achar, admite que não sabe
em vez de inventar. É o padrão conhecido como Corrective RAG.

    START → retrieve → grade_documents ─┬─(suficiente)─→ generate → END
                                        │
                                        └─(insuficiente, ainda há tentativa)
                                              → rewrite → retrieve …
"""
import json
import logging
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.generate import answer as generate_answer
from app.llm import complete
from app.rerank import rerank
from app.search import Hit, search

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 3


class AgentState(TypedDict, total=False):
    question: str            # pergunta original, nunca é alterada
    query: str               # consulta corrente, que o agente pode reescrever
    documents: list[Hit]
    generation: str
    attempts: int
    grade: float             # 0 a 10 — quão bem o material recuperado serve
    trace: Annotated[list[str], lambda a, b: a + b]   # o caminho percorrido


# ── Nós ───────────────────────────────────────────────────────────────

def retrieve(state: AgentState) -> AgentState:
    s = get_settings()
    query = state.get("query") or state["question"]
    hits = search(query, limit=20, mode="hybrid")
    hits = rerank(state["question"], hits, s.top_k)
    return {
        "documents": hits,
        "attempts": state.get("attempts", 0) + 1,
        "trace": [f"retrieve(query={query!r}) → {len(hits)} trechos"],
    }


GRADE_SYSTEM = (
    "You judge whether the retrieved excerpts are enough to answer the question. "
    'Reply ONLY with JSON: {"score": <0-10>, "reason": "<short>"}. '
    "10 = the excerpts fully answer it. 5 = related but incomplete. "
    "0 = unrelated."
)


def grade(state: AgentState) -> AgentState:
    """Avalia o conjunto recuperado como um todo, não trecho a trecho.

    O reranking já ordena por relevância individual; o que falta é a pergunta
    que o gerador não faz: 'com isto aqui dá para responder?'
    """
    docs = state.get("documents") or []
    if not docs:
        return {"grade": 0.0, "trace": ["grade → 0 (nada recuperado)"]}

    contexto = "\n\n---\n\n".join(
        f"[{i}] {h.title} — {h.heading}\n{h.content[:800]}" for i, h in enumerate(docs, 1)
    )
    try:
        raw = complete(
            GRADE_SYSTEM,
            f"QUESTION:\n{state['question']}\n\nEXCERPTS:\n{contexto}",
            max_tokens=120,
        )
        parsed = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        nota = float(parsed["score"])
        motivo = str(parsed.get("reason", ""))[:80]
    except Exception as e:
        # Falha na avaliação não pode travar o agente: segue para a geração,
        # que é o comportamento do RAG comum.
        log.warning("grade falhou: %s", e)
        return {"grade": 10.0, "trace": [f"grade → indisponível ({e}), seguindo"]}

    return {"grade": nota, "trace": [f"grade → {nota:.0f}/10 · {motivo}"]}


REWRITE_SYSTEM = (
    "You rewrite a search query to retrieve better passages from a technical "
    "documentation corpus. Keep the user's intent, but prefer the terminology "
    "the documentation itself would use: expand abbreviations, add likely "
    "synonyms and key technical terms. Reply with the rewritten query only, "
    "no quotes and no explanation."
)


def rewrite(state: AgentState) -> AgentState:
    atual = state.get("query") or state["question"]
    try:
        nova = complete(
            REWRITE_SYSTEM,
            f"Original question: {state['question']}\n"
            f"Query already tried (returned weak results): {atual}",
            max_tokens=100,
        ).strip().strip('"')
    except Exception as e:
        log.warning("rewrite falhou: %s", e)
        return {"trace": [f"rewrite → falhou ({e})"]}
    return {"query": nova, "trace": [f"rewrite → {nova!r}"]}


def generate(state: AgentState) -> AgentState:
    docs = state.get("documents") or []
    if state.get("grade", 10.0) < 3.0:
        # Recuperação ruim mesmo depois das tentativas: dizer que não sabe é
        # melhor resposta do que uma síntese confiante de material errado.
        return {
            "generation": "I could not find this information in the indexed documents.",
            "trace": ["generate → recusa (material insuficiente)"],
        }
    return {"generation": generate_answer(state["question"], docs),
            "trace": ["generate → resposta com citações"]}


# ── Aresta condicional ────────────────────────────────────────────────

def decide(state: AgentState) -> Literal["generate", "rewrite"]:
    nota = state.get("grade", 0.0)

    if nota >= 7.0:
        return "generate"

    # Nota zero significa que o assunto não existe no corpus, não que a
    # consulta estava mal escrita. Reescrever não cria documento: medido, o
    # caminho completo gastava ~14s e duas chamadas extras de LLM para chegar
    # na mesma recusa. Reescrita só vale quando veio material relacionado
    # porém incompleto — a faixa intermediária.
    if nota <= 1.0:
        return "generate"

    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "generate"          # esgotou as tentativas; generate decide recusar
    return "rewrite"


# ── Montagem do grafo ─────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade)   # o nó não pode repetir o nome de
                                           # uma chave do estado (aqui, "grade")
    g.add_node("rewrite", rewrite)
    g.add_node("generate", generate)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges("grade_documents", decide, {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run(question: str) -> dict:
    final = get_graph().invoke({"question": question, "query": question, "trace": []})
    return {
        "answer": final.get("generation", ""),
        "documents": final.get("documents", []),
        "attempts": final.get("attempts", 0),
        "grade": final.get("grade"),
        "trace": final.get("trace", []),
    }
