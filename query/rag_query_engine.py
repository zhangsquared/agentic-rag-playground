from llama_index.core.prompts import PromptTemplate
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

from config import RERANK_MODEL_NAME
from ingest.load_docs import load_index

RETRIVAL_TOP_K = 5
RERANK_TOP_N = 2


def build_query_engine():
    # load knowledge base index
    index = load_index()

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
    reranker = FlagEmbeddingReranker(model=RERANK_MODEL_NAME, top_n=RERANK_TOP_N)

    # build query engine
    query_engine = index.as_query_engine(
        similarity_top_k=RETRIVAL_TOP_K,
        node_postprocessors=[reranker],
        prompt_template=query_prompt_template,
        response_mode="compact",  # "tree_summarize" is too expensive for my free tier!
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
