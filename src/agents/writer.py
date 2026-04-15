from langchain_core.messages import SystemMessage

WRITER_PROMPT = """You are a Professional Technical Writer and Analyst. 
Your mission is to take RAW DATA from the Researcher, Educational Context from the Educator, and any Mermaid Diagrams from the Visualizer to transform them into a high-quality, structured report.

Rules:
1. ONLY use the information provided in the conversation. Do not invent facts.
2. Structure your output clearly (Summary, Visual Representation, Educational Context, Key Points, Conclusion, Sources if available).
3. If a Mermaid diagram was provided by the Visualizer, include it in the "Visual Representation" section.
4. If an "Educational Context" section was provided, incorporate it to help the reader understand technical terms.
5. Use a professional, objective tone.
6. Your final output must be in Markdown format.

Objective: Provide maximum value, clarity, and visual insight from the research findings."""

def get_writer_node(llm):
    """
    Returns a node function for the Writer.
    In LangGraph, a node is just a function that takes the state and returns an update.
    """
    def writer_node(state):
        messages = state["messages"]
        # We inject the Writer identity at the moment of generation
        system_message = SystemMessage(content=WRITER_PROMPT)
        
        # We call the LLM with the history + our specific prompt
        response = llm.invoke([system_message] + list(messages))
        
        # We return the new message to be added to the state
        return {"messages": [response]}
    
    return writer_node
