"""Avaliação do retriever.

Mede, sobre o mesmo conjunto de perguntas, o que cada estratégia entrega:
vetorial pura, lexical pura, híbrida e híbrida com reranking.

Métricas:
  recall@k — em quantas perguntas o documento correto apareceu entre os k
             primeiros. É o teto do sistema: o que não é recuperado, o
             gerador não tem como responder.
  MRR      — média de 1/posição do primeiro acerto. Distingue "achou em
             primeiro" de "achou em quinto", que recall@k trata igual.

Sem isso, mexer no pipeline é chute: qualquer alteração parece boa em duas
ou três perguntas escolhidas a dedo.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rerank import rerank          # noqa: E402
from app.search import search          # noqa: E402


def _matches(hit, item: dict) -> bool:
    """Acerto = o CHUNK recuperado contém a resposta, não apenas o arquivo.

    Avaliar por arquivo é frouxo: com poucos documentos, qualquer estratégia
    acerta o arquivo e todas empatam. O que importa para o gerador é o trecho
    certo ter chegado até ele — por isso o gabarito é um trecho que precisa
    estar dentro do chunk.
    """
    if os.path.basename(hit.source) != item["expected"]:
        return False
    alvo = item.get("must_contain")
    return True if not alvo else alvo.lower() in hit.content.lower()


def evaluate(questions: list[dict], mode: str, k: int, use_rerank: bool) -> dict:
    acertos = 0
    reciprocos = []

    for item in questions:
        hits = search(item["question"], limit=20 if use_rerank else k, mode=mode)
        if use_rerank:
            hits = rerank(item["question"], hits, k)
        hits = hits[:k]

        posicao = next(
            (i for i, h in enumerate(hits, 1) if _matches(h, item)),
            None,
        )
        if posicao:
            acertos += 1
            reciprocos.append(1.0 / posicao)
        else:
            reciprocos.append(0.0)

    n = len(questions)
    return {
        "estrategia": f"{mode}{' + rerank' if use_rerank else ''}",
        f"recall@{k}": round(acertos / n, 3),
        "mrr": round(sum(reciprocos) / n, 3),
        "acertos": f"{acertos}/{n}",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--questions", default="eval/questions.json")
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    print(f"Avaliando {len(questions)} perguntas · k={args.k}\n")

    linhas = [
        evaluate(questions, "vector", args.k, False),
        evaluate(questions, "text",   args.k, False),
        evaluate(questions, "hybrid", args.k, False),
        evaluate(questions, "hybrid", args.k, True),
    ]

    largura = max(len(l["estrategia"]) for l in linhas) + 2
    print(f"{'estratégia'.ljust(largura)}{f'recall@{args.k}'.ljust(12)}{'MRR'.ljust(8)}acertos")
    print("-" * (largura + 32))
    for l in linhas:
        print(f"{l['estrategia'].ljust(largura)}"
              f"{str(l[f'recall@{args.k}']).ljust(12)}"
              f"{str(l['mrr']).ljust(8)}{l['acertos']}")

    Path("eval/results").mkdir(parents=True, exist_ok=True)
    Path("eval/results/latest.json").write_text(
        json.dumps(linhas, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nresultados em eval/results/latest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
