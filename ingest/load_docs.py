import os

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.readers.github import GithubClient, GithubRepositoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import COLLECTION_NAME, EMBED_MODEL_NAME, LLM_MODEL_NAME

STORAGE = os.path.abspath("./storage")


def load_and_chunk_docs():
    reader = GithubRepositoryReader(
        github_client=GithubClient(),
        owner="zhangsquared",
        repo="zz-slack-bot",
        verbose=True,
        filter_file_extensions=([".md"], GithubRepositoryReader.FilterType.INCLUDE),
        concurrent_requests=1,  # to avoid rate limiting
    )

    print("Fetching files...")
    documents = reader.load_data(
        branch="main",
    )
    print(f"Found {len(documents)} markdown files in the folder.")

    paser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=64,
    )
    nodes = paser.get_nodes_from_documents(documents)
    return nodes


def save_index(nodes):
    chroma_client = chromadb.PersistentClient(
        path=STORAGE,
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
        llm=GoogleGenAI(
            model_name=LLM_MODEL_NAME,
        ),
    )
    storage_context.persist(persist_dir=STORAGE)
    print(f"Index and metadata successfully persisted to {STORAGE}.")


def load_index() -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(
        path=STORAGE,
    )
    chroma_collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index: VectorStoreIndex = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, storage_context=storage_context
    )
    print(f"Index successfully loaded from chroma {chroma_collection.count()} vectors.")
    return index


if __name__ == "__main__":
    nodes = load_and_chunk_docs()
    save_index(nodes)

    index: VectorStoreIndex = load_index()
    print(f"Index loaded {index}")
