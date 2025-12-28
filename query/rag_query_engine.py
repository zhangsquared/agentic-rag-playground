from llama_index.core import VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from .. import RERANK_MODEL_NAME

index = VectorStoreIndex.load_from_disk("storage")
user_query = "What is RAG?"

# vector retrieval
retriever = index.as_retriever(similarity_top_k=10)
nodes = retriever.retrieve(user_query)
for node in nodes:
    print(
        node.score, node.text[:100]
    )  # Print first 100 characters of each retrieved node

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

reranker = FlagEmbeddingReranker(model=RERANK_MODEL_NAME, top_n=3)

query_engine = index.as_query_engine(
    retriever=retriever,
    node_postprocessors=[reranker],
    prompt_template=query_prompt_template,
    response_mode="tree_summarize",
)
response = query_engine.query(user_query)

print(response)
for node in response.source_nodes:
    print(node.metadata)
