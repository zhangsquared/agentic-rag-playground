from llama_index.core.prompts import PromptTemplate
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

from config import RERANK_MODEL_NAME
from ingest.load_docs import load_index


def build_query_engine():
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


# global query engine instance
query_engine = build_query_engine()

if __name__ == "__main__":
    user_query = "Which slack workspace does this bot work in?"
    response = query_engine.query(user_query)
    for node in response.source_nodes:
        print(node.metadata)
    print(response.response)
