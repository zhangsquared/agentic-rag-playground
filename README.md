# Agentic RAG Playground

Ref: [Rag Playground]

Use LlamdaIndex to build agentic RAG

## Set up env

```bash
uv init .
uv add ruff  # lint tool
# format
uv run ruff format .
```

```bash
uv add llama-index llama-index-embeddings-huggingface llama-index-postprocessor-flag-embedding-reranker chromadb

uv add fastapi uvicorn
```

ingestion = batch job
querying = stateless service
index persisted

```bash
uv run uvicorn query.api:app --host 0.0.0.0 --port 8000
```

## Choose embedding and reranking model

Requirement: free, fast, no requirement for GPU

Embedding: `BAAI/bge-small-en-v1.5`
Reranking: `BAAI/bge-reranker-base`
# agentic-rag-playground
