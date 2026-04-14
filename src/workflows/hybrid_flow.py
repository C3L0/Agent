from typing import Literal
from langgraph.graph import StateGraph, START, END

from src.state import AgentState
from src.agents.researcher import get_researcher_agent
from src.agents.writer import get_writer_node

def router(state: AgentState) -> Literal["researcher", "writer"]:
    """
    Decision node: Does this query need research or can we just write?
    This is a simple logic, but it could be a small LLM call.
    """
    last_message = state["messages"][-1].content.lower()
    
    # Keywords that suggest we need a fresh search
    search_keywords = ["cherche", "trouve", "quelles sont", "actualité", "news", "search", "find", "who is"]
    
    if any(kw in last_message for kw in search_keywords):
        return "researcher"
    return "writer"

def get_hybrid_workflow(llm):
    # 1. Initialize our specialized components
    researcher_agent = get_researcher_agent(llm)
    writer_node = get_writer_node(llm)
    
    # 2. Define a wrapper for the researcher agent to handle state correctly
    def researcher_node(state: AgentState):
        # We call the compiled researcher agent
        # It expects a dict with 'messages' and returns the same
        result = researcher_agent.invoke(state)
        return {"messages": result["messages"]}
    
    # 3. Define the Graph
    workflow = StateGraph(AgentState)
    
    # 4. Add our Nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)

    
    # 4. Build the Flow with logic
    
    # START -> Logic -> researcher OR writer
    workflow.add_conditional_edges(
        START,
        router,
        {
            "researcher": "researcher",
            "writer": "writer"
        }
    )
    
    # researcher -> writer (Sequence)
    workflow.add_edge("researcher", "writer")
    
    # writer -> END
    workflow.add_edge("writer", END)
    
    return workflow.compile()
