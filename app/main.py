"""API HTTP do pipeline de RAG."""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import db, generate
from app.graph import run as run_agent
from app.config import get_settings
from app.ingest import ingest_directory
from app.rerank import rerank
from app.search import search

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(
    title="Hybrid RAG",
    description="Busca híbrida (vetorial + full-text) com reranking e respostas citadas.",
    version="1.0.0",
    lifespan=lifespan,
)


class IngestRequest(BaseModel):
    directory: str = Field(default="corpus", description="Pasta com os documentos")


class SearchResponse(BaseModel):
    chunk_id: int
    source: str
    title: str
    heading: str
    content: str
    score: float
    vector_rank: int | None
    text_rank: int | None


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None
    mode: str = Field(default="hybrid", pattern="^(vector|text|hybrid)$")
    use_rerank: bool = True


class AgentRequest(BaseModel):
    question: str


class AgentResponse(BaseModel):
    answer: str
    sources: list["SearchResponse"]
    attempts: int
    grade: float | None
    trace: list[str]
    elapsed_ms: int


class AskResponse(BaseModel):
    answer: str
    sources: list[SearchResponse]
    elapsed_ms: int
    mode: str
    reranked: bool


@app.get("/health")
def health():
    try:
        return {"status": "ok", **db.stats()}
    except Exception as e:
        raise HTTPException(503, f"banco indisponível: {e}")


@app.post("/ingest")
def ingest(req: IngestRequest):
    try:
        return ingest_directory(req.directory)
    except FileNotFoundError:
        raise HTTPException(404, f"pasta não encontrada: {req.directory}")


@app.get("/search", response_model=list[SearchResponse])
def search_endpoint(q: str, k: int = 5, mode: str = "hybrid", rerank_on: bool = False):
    """Busca sem geração — útil para inspecionar o que o retriever devolve."""
    settings = get_settings()
    hits = search(q, limit=settings.candidates_per_leg if rerank_on else k, mode=mode)
    if rerank_on:
        hits = rerank(q, hits, k)
    return [SearchResponse(**vars(h)) for h in hits]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    settings = get_settings()
    k = req.top_k or settings.top_k
    started = time.perf_counter()

    # Com reranking, recupera-se mais candidatos do que se entrega ao modelo:
    # o reranker precisa de material para escolher.
    hits = search(req.question, limit=20 if req.use_rerank else k, mode=req.mode)
    if req.use_rerank:
        hits = rerank(req.question, hits, k)

    texto = generate.answer(req.question, hits)
    return AskResponse(
        answer=texto,
        sources=[SearchResponse(**vars(h)) for h in hits],
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        mode=req.mode,
        reranked=req.use_rerank,
    )


@app.post("/agent", response_model=AgentResponse)
def agent(req: AgentRequest):
    """RAG corretivo com LangGraph.

    Diferente de /ask, aqui o sistema julga o que recuperou e reformula a
    consulta quando o material não serve. `trace` devolve o caminho percorrido
    — é o que permite depurar por que uma resposta saiu como saiu.
    """
    started = time.perf_counter()
    out = run_agent(req.question)
    return AgentResponse(
        answer=out["answer"],
        sources=[SearchResponse(**vars(h)) for h in out["documents"]],
        attempts=out["attempts"],
        grade=out["grade"],
        trace=out["trace"],
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )
