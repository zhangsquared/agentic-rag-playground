# Agentic RAG Playground

Use LlamdaIndex to build agentic RAG</br>
(Ref: [Rag Playground](https://github.com/zhangsquared/rag-playground))

## Set up env

```bash
uv init .
uv add ruff  # lint tool
```

Dependencies:

```bash
uv add llama-index
uv add llama-index-embeddings-huggingface
uv add llama-index-postprocessor-flag-embedding-reranker
uv add chromadb
uv add fastapi uvicorn
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
│   └── rag_query_engine.py # given a user query, perform vector retrieval, apply reranking, and generate a response.
└── storage/ # persisted index document
```

Run the batch ingestion job:

```bash

```

Start the online service:

```bash
uv run uvicorn query.api:app --host 0.0.0.0 --port 8000
```

Run lint tool:

```bash
uv run ruff format .
```

## Choose embedding and reranking model

Requirement: free, fast, no requirement for GPU

- Embedding: `BAAI/bge-small-en-v1.5`
- Reranking: `BAAI/bge-reranker-base`
