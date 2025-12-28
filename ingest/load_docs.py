import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.github import GithubRepositoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore

from .. import COLLECTION_NAME, EMBED_MODEL_NAME


def load_and_chunk_docs():
    reader = GithubRepositoryReader(
        owner="OWNER_NAME",
        repo="REPO_NAME",
        filter_directories=("", ""),
        filter_file_extensions=(".md",),
    )
    documents = reader.load_data(branch="main")
    paser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=64,
    )
    nodes = paser.get_nodes_from_documents(documents)
    return nodes


def save_index(nodes):
    chroma_client = chromadb.PersistentClient(
        path="../storage",
        reset=True,  # temp for dev. delete this in prod
    )
    chroma_collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embedding_model=HuggingFaceEmbedding(
            model_name=EMBED_MODEL_NAME,
        ),
    )


if __name__ == "__main__":
    nodes = load_and_chunk_docs()
    save_index(nodes)
