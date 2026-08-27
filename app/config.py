"""Configuração central. Tudo que muda de ambiente passa por aqui."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5433/rag"

    # ── Embeddings ────────────────────────────────────────────────
    # A dimensão precisa bater com a coluna vector() do schema.
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # ── Modelo de linguagem (rerank + geração) ────────────────────
    # base_url vazio = OpenAI. Qualquer provedor compatível com o
    # padrão OpenAI (Groq, Together, DeepSeek, vLLM local) entra só
    # apontando llm_base_url — sem tocar no código.
    llm_base_url: str = ""
    llm_model: str = "gpt-4o-mini"
    rerank_model: str = "gpt-4o-mini"

    openai_api_key: str = ""
    groq_api_key: str = ""

    # ── Chunking ──────────────────────────────────────────────────
    chunk_size: int = 900          # caracteres por chunk
    chunk_overlap: int = 150       # sobreposição entre chunks vizinhos

    # ── Busca ─────────────────────────────────────────────────────
    candidates_per_leg: int = 30   # quantos cada perna da busca traz
    rrf_k: int = 60                # constante do Reciprocal Rank Fusion
    top_k: int = 5                 # quantos chunks vão para o LLM

    # Idioma do dicionário de full-text search do Postgres
    fts_language: str = "english"


@lru_cache
def get_settings() -> Settings:
    return Settings()
