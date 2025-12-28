import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.vector_stores.chroma import ChromaVectorStore

from .. import COLLECTION_NAME, RERANK_MODEL_NAME


def load_index():
    chroma_client = chromadb.Client(
        chromadb.config.Settings(
            persist_directory="../storage",
            reset=True,
        )
    )
    chroma_collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, storage_context=storage_context
    )
    return index


def query_engine():
    index = load_index()
    # vector retrieval
    retriever = index.as_retriever(similarity_top_k=10)

    # custom prompt template
    query_prompt_template = PromptTemplate(
        """You are a knowledge assistant.
        Use ONLY the context below.
        If the answer is not present, say "I don't know".

        Context:
        {context_str}

        Question:
        {query_str}
        """
    )
    # reranker model
    reranker = FlagEmbeddingReranker(model=RERANK_MODEL_NAME, top_n=3)

    # build query engine
    query_engine = index.as_query_engine(
        retriever=retriever,
        node_postprocessors=[reranker],
        prompt_template=query_prompt_template,
        response_mode="tree_summarize",
    )
    return query_engine


def test_query(user_query: str):
    engine = query_engine()
    response = engine.query(user_query)
    for node in response.source_nodes:
        print(node.metadata)
    return response


if __name__ == "__main__":
    test_query("What is RAG?")
