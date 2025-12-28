# Agentic RAG Playground

Use LlamdaIndex to build agentic RAG</br>
(Ref: [Rag Playground](https://github.com/zhangsquared/rag-playground))

## Set up env

```bash
uv sync
```

### First time set up env

```bash
uv init .
uv add ruff  # lint tool
```

Dependencies:

```bash
uv add llama-index
uv add llama-index-embeddings-gemini
uv add llama-index-llms-google-genai
uv add llama-index-embeddings-huggingface # embeding model
uv add FlagEmbedding
uv add llama-index-postprocessor-flag-embedding-reranker # reranker model
uv add llama-index-readers-github # knowledge base: github readme
uv add chromadb
uv add llama-index-vector-stores-chroma
uv add fastapi uvicorn
uv add python-dotenv
```

## Code structure

```bash
agentic-rag-playground/
├── __init__.py # basic setting for llamdaindex core
├── ingest/
│   └── load_docs.py # batch job to chuck, embed documents, and save to storage
├── query/
│   ├── agent.py # AI Agent
│   ├── api.py # FastAPI, stateless online service
│   └── rag_query_engine.py # given a user query, perform vector retrieval,
│                           # apply reranking, and generate a response.
└── storage/ # persisted index document
```

Run the batch ingestion job:
```bash
uv run python -m ingest.load_docs
```

Local test rag query engine:
```bash
uv run python -m query.rag_query_engine
```

Start the online service:
```bash
uv run uvicorn query.api:app --host 0.0.0.0 --port 8000
```

Run lint tool:

```bash
ruff check --select I --fix . | uv run ruff format .
```

## Choose embedding and reranking model

Requirement: free, fast, no requirement for GPU

- Embedding: `BAAI/bge-small-en-v1.5`
- Reranking: `BAAI/bge-reranker-base`

On the first run, `load_docs.py` or `rag_query_engine.py` may be slow because the embedding model and reranking model need to be downloaded. In particular, the reranking model can take up to an hour to download on the first run.</br>
e.g.
```bash
(agentic-rag-playground) $ uv run python -m query.rag_query_engine
2025-12-28 15:00:39,874 - INFO - Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.
Index successfully loaded from chroma 13 vectors.
tokenizer_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 443/443 [00:00<00:00, 4.07MB/s]
sentencepiece.bpe.model: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5.07M/5.07M [00:01<00:00, 2.58MB/s]
tokenizer.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 17.1M/17.1M [00:02<00:00, 6.27MB/s]
special_tokens_map.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 279/279 [00:00<00:00, 1.45MB/s]
config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 799/799 [00:00<00:00, 7.01MB/s]
model.safetensors:   0%|                                                                                                                                 | 0.00/1.11G [00:00<?, ?B/s]
```
