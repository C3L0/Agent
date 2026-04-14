from langchain_core.messages import SystemMessage

WRITER_PROMPT = """You are a Professional Technical Writer and Analyst. 
Your mission is to take RAW DATA provided by a Researcher and transform it into a high-quality, structured report.

Rules:
1. ONLY use the information provided in the conversation. Do not invent facts.
2. Structure your output clearly (Summary, Key Points, Conclusion, Sources if available).
3. Use a professional, objective tone.
4. If the data is insufficient, state what is missing instead of guessing.
5. Your final output must be in Markdown format.

Objective: Provide maximum value and clarity from the research findings."""

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
