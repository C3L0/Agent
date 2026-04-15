from langchain_core.messages import SystemMessage
from src.state import AgentState

EDUCATOR_PROMPT = """You are an AI Concepts Educator.
Your mission is to analyze research data and identify technical concepts, acronyms, or architectures that might be difficult for a non-expert to understand.

Rules:
1. Identify 2-3 key technical concepts from the research findings.
2. Provide a brief (2-sentence) explanation for each, using simple analogies where possible.
3. If no complex concepts are found, just say "No complex concepts to explain."
4. Your output should be structured as:
   ## Educational Context
   - **Concept 1**: Explanation
   - **Concept 2**: Explanation

Objective: Ensure the user has the necessary background to fully understand the news."""

def get_educator_node(llm):
    """
    Returns a node function for the Educator.
    In LangGraph, a node is just a function that takes the state and returns an update.
    """
    def educator_node(state: AgentState):
        messages = state["messages"]
        # The educator analyzes the latest research findings (usually from the researcher)
        system_message = SystemMessage(content=EDUCATOR_PROMPT)
        
        # We call the LLM with the current messages (which include researcher's results)
        response = llm.invoke([system_message] + list(messages))
        
        # We return the new message to be added to the state
        return {"messages": [response]}
    
    return educator_node
