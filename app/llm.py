"""Cliente de LLM agnóstico de provedor.

Toda a aplicação fala com o modelo por esta função. Provedores compatíveis
com o padrão OpenAI (Groq, DeepSeek, Together, vLLM local) entram apenas
apontando LLM_BASE_URL — sem tocar em código de aplicação.
"""
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings


class LLMError(RuntimeError):
    pass


def _endpoint_and_key() -> tuple[str, str]:
    s = get_settings()
    if s.llm_base_url:
        base = s.llm_base_url.rstrip("/")
        key = s.groq_api_key or s.openai_api_key
    else:
        base = "https://api.openai.com/v1"
        key = s.openai_api_key
    if not key:
        raise LLMError("nenhuma chave de LLM configurada")
    return f"{base}/chat/completions", key


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=15), reraise=True)
def complete(system: str, user: str, model: str | None = None,
             temperature: float = 0.0, max_tokens: int = 800) -> str:
    s = get_settings()
    url, key = _endpoint_and_key()
    r = httpx.post(
        url,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model or s.llm_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=90,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()
