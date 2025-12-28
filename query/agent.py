from llama_index.core.agent import ReActAgent
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.tools import FuntionTool, QueryEngineTool

from query.rag_query_engine import query_engine


def query_tool():
    query_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="Query Tool",
        description="Search internal documents for relevant information.",
    )
    return query_tool


def summary_tool():
    def summarize(text: str) -> str:
        # your own basic summarizer; replace with an LLM call if desired
        return text[:500] + "..."

    summary_tool = FuntionTool.from_defaults(
        func=summarize,
        name="Summary Tool",
        description="Provides a brief summary of what RAG is.",
    )
    return summary_tool


def hyde_tool():
    hyde = HyDEQueryTransform()
    hyde_engine = TransformQueryEngine(query_engine, query_transform=hyde)
    hyde_tool = QueryEngineTool.from_defaults(
        query_engine=hyde_engine,
        name="Semantic Search Tool",
        description="Semantic search with query expansion.",
    )
    return hyde_tool


def build_agent():
    query_tool_instance = query_tool()
    # summary_tool_instance = summary_tool()
    # hyde_tool_instance = hyde_tool()
    agent = ReActAgent.from_tools(
        tools=[query_tool_instance],
        verbose=True,
        system_prompt="""
    You are a retrieval agent.
    Rules:
    - Always search documents before answering
    - If information is incomplete, search again with a refined query
    - Never hallucinate
    """,
    )
    return agent


# global agent instance
agent = build_agent()
