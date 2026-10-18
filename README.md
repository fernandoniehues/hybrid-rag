# Hybrid RAG

A RAG (Retrieval-Augmented Generation) pipeline with **hybrid search**, **reranking**, and **measured evaluation** — built to answer questions over a document base while citing the source of every claim.

The whole stack comes up with one command. Every number in this README was measured in this repository, not estimated.

```bash
cp .env.example .env          # add your OPENAI_API_KEY
docker compose up -d --build
python scripts/fetch_corpus.py corpus
curl -X POST localhost:8000/ingest -d '{"directory":"corpus"}' -H 'Content-Type: application/json'
curl -X POST localhost:8000/ask   -d '{"question":"How do I create an HNSW index?"}' -H 'Content-Type: application/json'
```

---

## The problem

An LLM only knows what was in its training data and what fits in the prompt. It has no way to answer questions about internal documentation, manuals, or a knowledge base — and the worst failure mode is answering anyway, with confidence.

RAG solves this by retrieving the relevant passages and injecting them into the prompt. The hard part isn't the idea, it's **retrieving the right passage**: what never reaches the model never makes it into the answer.

## Architecture

```
                  ┌─────────────────────────────────────────┐
   documents ───► │ heading-based chunking + overlap        │
                  └────────────────┬────────────────────────┘
                                   ▼
                  ┌─────────────────────────────────────────┐
                  │ batched embeddings → PostgreSQL/pgvector│
                  │ + generated tsvector (full-text)        │
                  └─────────────────────────────────────────┘

   question ──┬──► vector search (HNSW, cosine) ────┐
              │                                     ├─► RRF ─► rerank ─► LLM ─► answer
              └──► lexical search (tsquery, GIN) ───┘                           + citations
```

**Hybrid search.** Vector search finds by meaning, but stumbles on proper nouns, acronyms, and version numbers. Lexical search nails the exact term, but doesn't understand synonyms. Both legs run on every query and are merged with **Reciprocal Rank Fusion** — each document scores `1/(k + rank)` in each ranking and the scores add up, which avoids normalizing scores on incompatible scales (cosine distance vs. `ts_rank`).

**Reranking.** Search is optimized to bring back many plausible candidates fast; it's not good at judging which one actually answers the question. The reranker reads 20 candidates and returns a relevance score for each — the expensive work is spent where it pays off. The calls are independent, so they run concurrently.

**Grounded generation.** The model answers using only the retrieved passages, cites `[n]` for every claim, and is explicitly instructed to say it found nothing when the material doesn't cover the question.

## Measured results

20 questions with ground-truth answers, over a corpus of 14 technical documents (647 chunks). A hit is counted **at the chunk level**: the right file showing up isn't enough — the retrieved passage must contain the answer.

### Questions in the corpus language (English)

| strategy | recall@1 | recall@5 | MRR@5 |
|---|---|---|---|
| vector | 0.55 | 0.75 | 0.637 |
| lexical | 0.35 | 0.40 | 0.362 |
| **hybrid** | **0.60** | 0.75 | 0.654 |
| **hybrid + rerank** | 0.60 | **0.80** | **0.667** |

### Portuguese questions, English corpus

| strategy | recall@5 | MRR@5 |
|---|---|---|
| vector | 0.80 | 0.683 |
| lexical | **0.00** | 0.000 |
| hybrid | 0.80 | 0.683 |
| **hybrid + rerank** | **0.95** | **0.712** |

The raw results of each run are in `eval/results/`. Reranking uses an LLM: separate runs differ by one or two questions even at `temperature=0`, so treat the third decimal place as noise.

**Lexical search drops to zero outside the corpus language** — `to_tsvector('english', …)` won't match "índice" with "index", and no amount of weight tuning rescues an empty ranking. Embeddings cross languages effortlessly. This is the most concrete argument for going hybrid: the two legs fail for different reasons, and that's exactly where reranking pays off the most (+15 recall points when one of the legs contributes nothing).

### Latency

| configuration | time |
|---|---|
| no reranking | ~2.8 s |
| sequential reranking | ~16.1 s |
| **parallel reranking** | **~4.2 s** |

Sequential reranking dominated latency: one short call per candidate, one after another. Since the calls are independent, they now run concurrently — total time became that of the slowest call, not the sum.

## Engineering decisions

**Chunk-level evaluation, not document-level.** The first version measured whether the right *file* showed up. With 14 documents, every strategy tied at 0.95 and the experiment told us nothing. Measuring the passage made the differences visible.

**Chunking follows the Markdown structure.** Cutting by character count mixes topics inside a chunk. Splitting follows headings first; size-based cutting only kicks in afterward, with overlap so nothing that falls on a boundary is lost, preferring to break at the end of a paragraph.

**Idempotent ingestion by hash.** Reprocessing the same folder neither duplicates nor re-embeds anything that hasn't changed — embeddings cost money and time.

**Provider swappable via configuration.** Embeddings and the LLM sit behind their own modules (`app/embeddings.py`, `app/llm.py`). Any OpenAI-compatible API — Groq, DeepSeek, Together, local vLLM — plugs in by pointing `LLM_BASE_URL` at it, without touching application code.

**Reranking degrades instead of failing.** If the provider fails, the error is logged and hybrid search alone answers. Reranking is a quality improvement, not an availability requirement.


## Corrective RAG agent (LangGraph)

Besides the direct pipeline (`POST /ask`), the project exposes an agent (`POST /agent`) orchestrated with **LangGraph**. The difference is that it doesn't blindly trust what it retrieved:

```
START → retrieve → grade_documents ─┬─(score ≥ 7)─────────→ generate → END
                                    │
                                    ├─(score 2–6, attempts remaining)
                                    │        → rewrite → retrieve …
                                    │
                                    └─(score ≤ 1)─────────→ generate (refusal)
```

One node judges whether the retrieved set answers the question. If the material is related but incomplete, the agent rewrites the query using the terminology the documentation would use and searches again, up to a maximum number of attempts. If nothing fits, it says it couldn't find the answer instead of synthesizing something confident and wrong.

**A score of zero ends it immediately.** Zero means the topic doesn't exist in the corpus — rewriting the query won't create a document. Measured: the full loop took **14.2 s** and two extra LLM calls to reach exactly the same refusal that now comes back in **5.9 s**. Rewriting only pays off in the middle band.

`trace` returns the path taken — which query was used, what score the material got and why, what was rewritten. Without it, debugging why an answer came out bad turns into guesswork.

### Did the agent beat the direct pipeline?

On this corpus, **no**. End-to-end measurement — does the answer contain what it should? — over the same 20 questions, plus 4 questions whose answer isn't in the corpus:

| mode | correct answer | refused when it should | avg. latency |
|---|---|---|---|
| direct pipeline | **13/20** | 4/4 | **4.4 s** |
| corrective agent | 12/20 | 4/4 | 5.8 s |

A one-question difference is within noise; the higher latency isn't. The result makes sense: **Corrective RAG is for when retrieval fails**, and with 14 documents and straightforward questions, hybrid search almost never fails — there's nothing to correct, only a cost to pay.

Where the agent should pay off, and what I would measure before adopting it in production:

- **large, heterogeneous corpora**, where the user's query rarely matches the documents' terminology;
- **poorly phrased questions**, with abbreviations, internal jargon, or ambiguous terms;
- **domains with their own vocabulary**, where rewriting brings the query closer to the language of the corpus.

The correct refusal rate of 4/4 was the same in both modes — the instruction in the generation prompt already handled it, no graph needed. That's useful information too: not every desirable behavior needs orchestration.

## API

| route | what it does |
|---|---|
| `GET /health` | database status and counts of documents, chunks, and pending embeddings |
| `POST /ingest` | indexes a folder (`{"directory": "corpus"}`) |
| `GET /search` | search without generation — `?q=&k=&mode=vector\|text\|hybrid&rerank_on=` |
| `POST /ask` | question with a cited answer (direct pipeline) |
| `POST /agent` | corrective agent in LangGraph, with a `trace` of the path taken |

Interactive docs at `localhost:8000/docs`.

## Evaluation

```bash
# retrieval quality, per strategy
docker compose exec api python scripts/evaluate.py --questions eval/questions-en.json --k 5
docker compose exec api python scripts/evaluate.py --questions eval/questions-pt.json --k 5

# answer quality, end to end: direct pipeline vs. agent
docker compose exec api python scripts/evaluate_agent.py --questions eval/questions-en.json
```

`evaluate.py` compares the four retrieval strategies on the same question set, and `evaluate_agent.py` compares the two answer modes; both write their results to `eval/results/`. Without this, changing the pipeline is guesswork: any change looks good on two or three hand-picked questions.

## Project structure

```
app/
  config.py       central configuration (env)
  db.py           connection pool, schema, and idempotent migration
  chunking.py     heading-based splitting + size-based cuts with overlap
  embeddings.py   embeddings layer, swappable provider
  search.py       vector search, lexical search, and RRF fusion
  rerank.py       LLM reranking, parallel, with graceful degradation
  generate.py     grounded answer with citations
  ingest.py       idempotent ingestion by hash
  llm.py          provider-agnostic LLM client
  graph.py        corrective RAG agent in LangGraph
  main.py         FastAPI app
scripts/
  fetch_corpus.py   downloads the public documentation used as the knowledge base
  evaluate.py       recall@k and MRR per strategy
  evaluate_agent.py correct answers and refusals, direct pipeline vs. agent
eval/
  questions-en.json / questions-pt.json   ground truth validated against the corpus
```

## Known limitations

- **Small corpus** (14 documents). The numbers are consistent, but 20 questions leave a wide margin of error; the difference between 0.75 and 0.80 is a single question.
- **Full-text tied to one language.** The Postgres dictionary is fixed by `FTS_LANGUAGE`. A multilingual corpus would need one `tsvector` column per language.
- **LLM reranking** is more expensive and slower than a dedicated cross-encoder. It was chosen for portability — it runs with any provider, with no model to download.
- **No cache for query embeddings.** Repeated questions pay for the embedding again.

## Knowledge base

The corpus is public documentation, downloaded by `scripts/fetch_corpus.py`: FastAPI (MIT), pgvector (PostgreSQL License), psycopg (LGPL), and the Compose Specification (Apache 2.0). It was chosen for its permissive licenses and because it's verifiable — anyone evaluating the project can check whether an answer is right.

The pipeline is domain-agnostic: point `/ingest` at any folder of `.md`, `.txt`, or `.rst` files.

## Stack

Python · FastAPI · **LangGraph** · PostgreSQL 17 + pgvector · psycopg 3 · Docker Compose · OpenAI API (embeddings and LLM, swappable)

---

Built by [Fernando Niehues](https://www.linkedin.com/in/fernando-niehues/).
