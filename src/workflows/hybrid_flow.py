from typing import Literal
from langgraph.graph import StateGraph, START, END

from src.state import AgentState
from src.agents.researcher import get_researcher_agent
from src.agents.writer import get_writer_node
from src.agents.educator import get_educator_node
from src.agents.visualizer import get_visualizer_node

def router(state: AgentState) -> Literal["researcher", "writer"]:
    """
    Decision node: Does this query need research or can we just write?
    """
    # Now that we use HumanMessage at the entry point, we can trust .content
    content = state["messages"][-1].content.lower()
    
    search_keywords = ["cherche", "trouve", "quelles sont", "actualité", "news", "search", "find", "who is"]
    
    if any(kw in content for kw in search_keywords):
        return "researcher"
    return "writer"

def get_hybrid_workflow(llm):
    # 1. Initialize our specialized components
    researcher_agent = get_researcher_agent(llm)
    educator_node = get_educator_node(llm)
    visualizer_node = get_visualizer_node(llm)
    writer_node = get_writer_node(llm)
    
    # 2. Define the Graph
    workflow = StateGraph(AgentState)
    
    # 3. Add our Nodes
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("educator", educator_node)
    workflow.add_node("visualizer", visualizer_node)
    workflow.add_node("writer", writer_node)
    
    # 4. Build the Flow
    workflow.add_conditional_edges(
        START,
        router,
        {
            "researcher": "researcher",
            "writer": "writer"
        }
    )
    
    workflow.add_edge("researcher", "educator")
    workflow.add_edge("educator", "visualizer")
    workflow.add_edge("visualizer", "writer")
    workflow.add_edge("writer", END)
    
    return workflow.compile()
