from langchain_core.messages import SystemMessage
from src.state import AgentState

VISUALIZER_PROMPT = """You are a Technical Visualizer.
Your mission is to create a Mermaid.js diagram to explain the core logic or architecture of the research findings.

Rules:
1. Use Mermaid.js syntax (e.g., graph TD, sequenceDiagram, etc.).
2. Focus on the most important technical flow or relationship found in the research.
3. Keep labels short and clear.
4. Wrap your output in a triple backtick block with the 'mermaid' language identifier.
5. If the content is too simple for a diagram, just say "No diagram needed."

Example Output:
```mermaid
graph TD
    A[User Query] --> B{Router}
    B -->|Search| C[Researcher]
    B -->|Direct| D[Writer]
```

Objective: Provide a visual representation that makes the technical concept intuitive."""

def get_visualizer_node(llm):
    """
    Returns a node function for the Visualizer.
    """
    def visualizer_node(state: AgentState):
        messages = state["messages"]
        system_message = SystemMessage(content=VISUALIZER_PROMPT)
        
        # Analyze everything so far to create the diagram
        response = llm.invoke([system_message] + list(messages))
        
        return {"messages": [response]}
    
    return visualizer_node
