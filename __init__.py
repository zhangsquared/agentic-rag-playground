from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"

Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

# chunk
Settings.chunk_size = 512
Settings.chunk_overlap = 64
Settings.node_parser = SentenceSplitter()
