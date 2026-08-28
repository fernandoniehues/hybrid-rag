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
| **híbrida + rerank** | **0,95** | **0,712** |

Os resultados brutos de cada rodada ficam em `eval/results/`. O reranking usa um LLM: rodadas diferentes variam em uma ou duas perguntas mesmo com `temperature=0`, então trate a terceira casa como ruído.

**A busca lexical zera fora do idioma do corpus** — `to_tsvector('english', …)` não casa "índice" com "index", e nenhum ajuste de peso salva um ranking vazio. Os embeddings atravessam o idioma sem esforço. É o argumento mais concreto a favor do híbrido: as duas pernas falham por motivos diferentes, e é justamente aí que o reranking mais rende (+15 pontos de recall quando uma das pernas não contribui).

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


## Agente de RAG corretivo (LangGraph)

Além do pipeline direto (`POST /ask`), o projeto expõe um agente (`POST /agent`) orquestrado com **LangGraph**. A diferença é que ele não confia cegamente no que recuperou:

```
START → retrieve → grade_documents ─┬─(nota ≥ 7)──────────→ generate → END
                                    │
                                    ├─(nota 2–6, com tentativas restantes)
                                    │        → rewrite → retrieve …
                                    │
                                    └─(nota ≤ 1)──────────→ generate (recusa)
```

Um nó avalia se o conjunto recuperado responde à pergunta. Se veio material relacionado mas incompleto, o agente reescreve a consulta com a terminologia que a documentação usaria e busca de novo, até um teto de tentativas. Se nada serve, responde que não encontrou em vez de sintetizar algo confiante e errado.

**Nota zero encerra na hora.** Nota zero significa que o assunto não existe no corpus — reescrever a consulta não cria documento. Medido: o caminho completo gastava **14,2 s** e duas chamadas extras de LLM para chegar exatamente na mesma recusa que agora sai em **5,9 s**. Reescrita só vale na faixa intermediária.

`trace` devolve o caminho percorrido — qual consulta foi usada, que nota o material recebeu e por quê, o que foi reescrito. Sem isso, depurar por que uma resposta saiu ruim vira adivinhação.

### O agente venceu o pipeline direto?

Neste corpus, **não**. Medição ponta a ponta — a resposta contém o que deveria? — sobre as mesmas 20 perguntas, mais 4 perguntas cuja resposta não existe no corpus:

| modo | resposta correta | recusou quando devia | latência média |
|---|---|---|---|
| pipeline direto | **13/20** | 4/4 | **4,4 s** |
| agente corretivo | 12/20 | 4/4 | 5,8 s |

Uma pergunta de diferença está dentro do ruído; a latência maior não está. O resultado faz sentido: **Corrective RAG serve para quando a recuperação falha**, e com 14 documentos e perguntas diretas a busca híbrida quase nunca falha — não há o que corrigir, só custo a pagar.

Onde o agente deve render, e o que eu mediria antes de adotá-lo em produção:

- **corpus grande e heterogêneo**, em que a consulta do usuário raramente casa com a terminologia dos documentos;
- **perguntas mal formuladas**, com abreviação, gíria interna ou termo ambíguo;
- **domínios com vocabulário próprio**, em que a reescrita aproxima a consulta da linguagem do corpus.

A recusa correta em 4/4 saiu igual nos dois modos — a instrução no prompt de geração já resolvia isso, sem precisar de grafo. Também é informação útil: nem todo comportamento desejável precisa de orquestração.

## API

| rota | o que faz |
|---|---|
| `GET /health` | estado do banco e contagem de documentos, chunks e pendências |
| `POST /ingest` | indexa uma pasta (`{"directory": "corpus"}`) |
| `GET /search` | busca sem geração — `?q=&k=&mode=vector\|text\|hybrid&rerank_on=` |
| `POST /ask` | pergunta com resposta citada (pipeline direto) |
| `POST /agent` | agente corretivo em LangGraph, com `trace` do caminho percorrido |

Documentação interativa em `localhost:8000/docs`.

## Avaliação

```bash
# qualidade da recuperação, por estratégia
docker compose exec api python scripts/evaluate.py --questions eval/questions-en.json --k 5
docker compose exec api python scripts/evaluate.py --questions eval/questions-pt.json --k 5

# qualidade da resposta, ponta a ponta: pipeline direto vs agente
docker compose exec api python scripts/evaluate_agent.py --questions eval/questions-en.json
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
  graph.py        agente de RAG corretivo em LangGraph
  main.py         API FastAPI
scripts/
  fetch_corpus.py baixa a documentação pública usada como base
  evaluate.py       recall@k e MRR por estratégia
  evaluate_agent.py resposta correta e recusa, pipeline direto vs agente
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

Python · FastAPI · **LangGraph** · PostgreSQL 17 + pgvector · psycopg 3 · Docker Compose · OpenAI API (embeddings e LLM, trocáveis)

---

Construído por [Fernando Niehues](https://www.linkedin.com/in/fernando-niehues/).
