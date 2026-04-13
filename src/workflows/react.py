from typing import Any, List

from langgraph.prebuilt import create_react_agent


def get_react_agent(llm: Any, tools: List[Any]):
    """Returns a simple ReAct agent using LangGraph."""
    return create_react_agent(llm, tools)
