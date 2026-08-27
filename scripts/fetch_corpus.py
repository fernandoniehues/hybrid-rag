"""Baixa documentação pública para servir de base de conhecimento.

Fontes escolhidas por licença permissiva e por serem verificáveis: quem
avalia o projeto consegue conferir se a resposta está certa.
"""
import sys
from pathlib import Path

import httpx

RAW = "https://raw.githubusercontent.com"

DOCS = {
    # FastAPI — MIT
    "fastapi-first-steps.md":     f"{RAW}/fastapi/fastapi/master/docs/en/docs/tutorial/first-steps.md",
    "fastapi-query-params.md":    f"{RAW}/fastapi/fastapi/master/docs/en/docs/tutorial/query-params.md",
    "fastapi-body.md":            f"{RAW}/fastapi/fastapi/master/docs/en/docs/tutorial/body.md",
    "fastapi-dependencies.md":    f"{RAW}/fastapi/fastapi/master/docs/en/docs/tutorial/dependencies/index.md",
    "fastapi-security-intro.md":  f"{RAW}/fastapi/fastapi/master/docs/en/docs/tutorial/security/index.md",
    "fastapi-bigger-apps.md":     f"{RAW}/fastapi/fastapi/master/docs/en/docs/tutorial/bigger-applications.md",
    "fastapi-background-tasks.md": f"{RAW}/fastapi/fastapi/master/docs/en/docs/tutorial/background-tasks.md",
    "fastapi-testing.md":         f"{RAW}/fastapi/fastapi/master/docs/en/docs/tutorial/testing.md",
    "fastapi-async.md":           f"{RAW}/fastapi/fastapi/master/docs/en/docs/async.md",
    "fastapi-deployment-docker.md": f"{RAW}/fastapi/fastapi/master/docs/en/docs/deployment/docker.md",
    # pgvector — PostgreSQL License
    "pgvector.md":                f"{RAW}/pgvector/pgvector/master/README.md",
    # psycopg 3 — LGPL (documentação)
    "psycopg-basic-usage.md":     f"{RAW}/psycopg/psycopg/master/docs/basic/usage.rst",
    "psycopg-pool.md":            f"{RAW}/psycopg/psycopg/master/docs/advanced/pool.rst",
    # Docker Compose — Apache 2.0
    "compose-file-reference.md":  f"{RAW}/compose-spec/compose-spec/master/spec.md",
}


def main(destination: str = "corpus") -> int:
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)

    ok = 0
    for name, url in DOCS.items():
        try:
            r = httpx.get(url, timeout=30, follow_redirects=True)
            r.raise_for_status()
            (out / name).write_text(r.text, encoding="utf-8")
            print(f"  baixado  {name}  ({len(r.text) // 1024} KB)")
            ok += 1
        except Exception as e:
            print(f"  FALHOU   {name}: {e}", file=sys.stderr)

    print(f"\n{ok}/{len(DOCS)} documentos em {out}/")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "corpus"))
