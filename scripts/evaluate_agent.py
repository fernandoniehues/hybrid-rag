"""Compara o pipeline direto com o agente corretivo, ponta a ponta.

A avaliação do retriever (scripts/evaluate.py) mede se o trecho certo foi
recuperado. Esta mede o que o usuário recebe: a RESPOSTA contém o que deveria?

São coisas diferentes. Recuperar o trecho certo e mesmo assim responder mal é
possível; recuperar mal e ainda assim acertar, também. Só a medição fim a fim
diz se o agente melhora a experiência ou só gasta mais tokens.

Mede também a taxa de recusa em perguntas cuja resposta não está no corpus —
onde o comportamento correto é dizer que não sabe.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.generate import answer as direct_answer    # noqa: E402
from app.graph import run as run_agent              # noqa: E402
from app.rerank import rerank                       # noqa: E402
from app.search import search                       # noqa: E402

RECUSA = ("could not find", "não encontrei", "not find this information")

# Perguntas cuja resposta não existe no corpus: o acerto aqui é RECUSAR.
FORA_DO_CORPUS = [
    "How do I configure horizontal pod autoscaling in Kubernetes?",
    "How do I set up a Kafka consumer group?",
    "What is the syntax for a Terraform module?",
    "How do I train a convolutional neural network in PyTorch?",
]


def _acertou(resposta: str, esperado: str) -> bool:
    return esperado.lower() in resposta.lower()


def _recusou(resposta: str) -> bool:
    return any(t in resposta.lower() for t in RECUSA)


def avaliar(questions: list[dict], modo: str) -> dict:
    acertos, tempos = 0, []
    for item in questions:
        t0 = time.perf_counter()
        if modo == "direto":
            hits = rerank(item["question"], search(item["question"], limit=20), 5)
            resposta = direct_answer(item["question"], hits)
        else:
            resposta = run_agent(item["question"])["answer"]
        tempos.append(time.perf_counter() - t0)
        if _acertou(resposta, item["must_contain"]):
            acertos += 1

    recusas, tempos_fora = 0, []
    for pergunta in FORA_DO_CORPUS:
        t0 = time.perf_counter()
        if modo == "direto":
            hits = rerank(pergunta, search(pergunta, limit=20), 5)
            resposta = direct_answer(pergunta, hits)
        else:
            resposta = run_agent(pergunta)["answer"]
        tempos_fora.append(time.perf_counter() - t0)
        if _recusou(resposta):
            recusas += 1

    n, m = len(questions), len(FORA_DO_CORPUS)
    return {
        "modo": modo,
        "acerto": f"{acertos}/{n}",
        "acerto_pct": round(acertos / n, 3),
        "recusa_correta": f"{recusas}/{m}",
        "latencia_media_s": round(sum(tempos + tempos_fora) / (n + m), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions-en.json")
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    print(f"{len(questions)} perguntas respondíveis + {len(FORA_DO_CORPUS)} fora do corpus\n")

    linhas = [avaliar(questions, "direto"), avaliar(questions, "agente")]

    print(f"{'modo':<10}{'resposta certa':<17}{'recusou quando devia':<23}{'latência média'}")
    print("-" * 68)
    for l in linhas:
        print(f"{l['modo']:<10}{l['acerto']+' ('+str(l['acerto_pct'])+')':<17}"
              f"{l['recusa_correta']:<23}{l['latencia_media_s']}s")

    Path("eval/results").mkdir(parents=True, exist_ok=True)
    Path("eval/results/agent-vs-direct.json").write_text(
        json.dumps(linhas, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nresultados em eval/results/agent-vs-direct.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
