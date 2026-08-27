# Hybrid RAG

Pipeline de RAG (Retrieval-Augmented Generation) com **busca híbrida**, **reranking** e **avaliação medida** — construído para responder perguntas sobre uma base de documentos citando a fonte de cada afirmação.

Sobe inteiro com um comando. Todos os números deste README foram medidos neste repositório, não estimados.

```bash
cp .env.example .env          # coloque sua OPENAI_API_KEY
docker compose up -d --build
python scripts/fetch_corpus.py corpus
curl -X POST localhost:8000/ingest -d '{"directory":"corpus"}' -H 'Content-Type: application/json'
curl -X POST localhost:8000/ask   -d '{"question":"How do I create an HNSW index?"}' -H 'Content-Type: application/json'
```

---

## O problema

Um LLM só sabe o que estava no treino e o que cabe no prompt. Perguntas sobre documentação interna, manual ou base de conhecimento ele não tem como responder — e o pior modo de falha é responder mesmo assim, com confiança.

RAG resolve buscando os trechos relevantes e injetando no prompt. O difícil não é a ideia, é **recuperar o trecho certo**: o que não chega ao modelo não vira resposta.

## Arquitetura

```
                  ┌─────────────────────────────────────────┐
   documentos ──► │ chunking por cabeçalho + sobreposição   │
                  └────────────────┬────────────────────────┘
                                   ▼
                  ┌─────────────────────────────────────────┐
                  │ embeddings em lote → PostgreSQL/pgvector│
                  │ + tsvector gerado (full-text)           │
                  └─────────────────────────────────────────┘

   pergunta ──┬──► busca vetorial (HNSW, cosseno) ──┐
              │                                     ├─► RRF ─► rerank ─► LLM ─► resposta
              └──► busca lexical (tsquery, GIN)  ───┘                            + citações
```

**Busca híbrida.** A vetorial acha por significado, mas erra com nome próprio, sigla e número de versão. A lexical acerta o termo exato, mas não entende sinônimo. As duas rodam em paralelo e são fundidas por **Reciprocal Rank Fusion** — cada documento pontua `1/(k + posição)` em cada ranking e os pontos somam, o que dispensa normalizar escores de escalas incompatíveis (distância de cosseno × `ts_rank`).

**Reranking.** A busca é otimizada para trazer muitos candidatos plausíveis rápido; ela não julga bem qual responde à pergunta. O reranker lê 20 candidatos e devolve uma nota de relevância para cada — o custo caro é gasto onde rende. As chamadas são independentes, então rodam concorrentes.

**Geração ancorada.** O modelo responde apenas com os trechos recuperados, cita `[n]` em cada afirmação, e tem instrução explícita de dizer que não encontrou quando o material não cobre a pergunta.

## Resultados medidos

20 perguntas com gabarito, corpus de 14 documentos técnicos (647 chunks). O acerto é contado **em nível de chunk**: não basta o arquivo certo aparecer, o trecho recuperado precisa conter a resposta.

### Perguntas no mesmo idioma do corpus (inglês)

| estratégia | recall@1 | recall@5 | MRR@5 |
|---|---|---|---|
| vetorial | 0,55 | 0,75 | 0,637 |
| lexical | 0,35 | 0,40 | 0,362 |
| **híbrida** | **0,60** | 0,75 | 0,654 |
| **híbrida + rerank** | 0,60 | **0,80** | **0,667** |

### Perguntas em português, corpus em inglês

| estratégia | recall@5 | MRR@5 |
|---|---|---|
| vetorial | 0,80 | 0,683 |
| lexical | **0,00** | 0,000 |
| híbrida | 0,80 | 0,683 |
| **híbrida + rerank** | **0,90** | **0,693** |

**A busca lexical zera fora do idioma do corpus** — `to_tsvector('english', …)` não casa "índice" com "index", e nenhum ajuste de peso salva um ranking vazio. Os embeddings atravessam o idioma sem esforço. É o argumento mais concreto a favor do híbrido: as duas pernas falham por motivos diferentes, e é justamente aí que o reranking mais rende (+10 pontos de recall quando uma das pernas não contribui).

### Latência

| configuração | tempo |
|---|---|
| sem reranking | ~2,8 s |
| reranking sequencial | ~16,1 s |
| **reranking paralelo** | **~4,2 s** |

O reranking sequencial dominava a latência: uma chamada curta por candidato, uma depois da outra. Como são independentes, passaram a rodar concorrentes — o tempo virou o da chamada mais lenta, não a soma.

## Decisões de engenharia

**Avaliação em nível de chunk, não de documento.** A primeira versão media se o *arquivo* certo aparecia. Com 14 documentos, todas as estratégias empatavam em 0,95 e o experimento não dizia nada. Medindo o trecho, as diferenças apareceram.

**Chunking respeita a estrutura do Markdown.** Cortar por número de caracteres mistura assuntos dentro de um chunk. A divisão segue os cabeçalhos primeiro; o corte por tamanho só entra depois, com sobreposição para não perder o que cai na fronteira, preferindo quebrar em fim de parágrafo.

**Ingestão idempotente por hash.** Reprocessar a mesma pasta não duplica nem re-embeda o que não mudou — embedding custa dinheiro e tempo.

**Provedor trocável por configuração.** Embeddings e LLM ficam atrás de módulos próprios (`app/embeddings.py`, `app/llm.py`). Qualquer API compatível com o padrão OpenAI — Groq, DeepSeek, Together, vLLM local — entra apontando `LLM_BASE_URL`, sem tocar em código de aplicação.

**Reranking degrada, não derruba.** Se o provedor falha, o erro é registrado e a busca híbrida sozinha responde. Reranking é melhoria de qualidade, não requisito de disponibilidade.

## API

| rota | o que faz |
|---|---|
| `GET /health` | estado do banco e contagem de documentos, chunks e pendências |
| `POST /ingest` | indexa uma pasta (`{"directory": "corpus"}`) |
| `GET /search` | busca sem geração — `?q=&k=&mode=vector\|text\|hybrid&rerank_on=` |
| `POST /ask` | pergunta com resposta citada |

Documentação interativa em `localhost:8000/docs`.

## Avaliação

```bash
docker compose exec api python scripts/evaluate.py --questions eval/questions-en.json --k 5
docker compose exec api python scripts/evaluate.py --questions eval/questions-pt.json --k 5
```

Compara as quatro estratégias sobre o mesmo conjunto e grava em `eval/results/latest.json`. Sem isso, mexer no pipeline é chute: qualquer alteração parece boa em duas ou três perguntas escolhidas a dedo.

## Estrutura

```
app/
  config.py       configuração central (env)
  db.py           pool, schema e migração idempotente
  chunking.py     divisão por cabeçalho + tamanho com sobreposição
  embeddings.py   camada de embeddings, provedor trocável
  search.py       busca vetorial, lexical e fusão RRF
  rerank.py       reranking por LLM, paralelo, com degradação
  generate.py     resposta ancorada com citações
  ingest.py       ingestão idempotente por hash
  llm.py          cliente de LLM agnóstico de provedor
  main.py         API FastAPI
scripts/
  fetch_corpus.py baixa a documentação pública usada como base
  evaluate.py     recall@k e MRR por estratégia
eval/
  questions-en.json / questions-pt.json   gabarito validado contra o corpus
```

## Limitações conhecidas

- **Corpus pequeno** (14 documentos). Os números são consistentes, mas 20 perguntas têm margem de erro larga; a diferença entre 0,75 e 0,80 é de uma pergunta.
- **Full-text preso a um idioma.** O dicionário do Postgres é fixado em `FTS_LANGUAGE`. Corpus multilíngue exigiria uma coluna `tsvector` por idioma.
- **Reranking por LLM** é mais caro e mais lento que um cross-encoder dedicado. A escolha foi por portabilidade — roda com qualquer provedor, sem baixar modelo.
- **Sem cache de embeddings de query.** Perguntas repetidas pagam o embedding de novo.

## Base de conhecimento

O corpus é documentação pública, baixada pelo `scripts/fetch_corpus.py`: FastAPI (MIT), pgvector (PostgreSQL License), psycopg (LGPL) e Compose Specification (Apache 2.0). Escolhida por licença permissiva e por ser verificável — quem avalia consegue conferir se a resposta está certa.

O pipeline é agnóstico de domínio: aponte `/ingest` para qualquer pasta de `.md`, `.txt` ou `.rst`.

## Stack

Python · FastAPI · PostgreSQL 17 + pgvector · psycopg 3 · Docker Compose · OpenAI API (embeddings e LLM, trocáveis)

---

Construído por [Fernando Niehues](https://www.linkedin.com/in/fernando-niehues/).
