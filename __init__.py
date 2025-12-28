from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
COLLECTION_NAME = "playground_collection"

Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
