from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

from src.tools.scrape import scrape_website
from src.tools.search import search_web
from src.tools.storage import save_to_knowledge_base

RESEARCHER_PROMPT = """You are a highly skilled Research Assistant. 
Your sole mission is to find, verify, and collect raw facts, data, and information.

Rules:
1. NEVER synthesize or write final reports. Just provide the raw findings.
2. Use 'search_web' to find relevant sources.
3. Use 'scrape_website' to get deep content from specific URLs.
4. Use 'save_to_knowledge_base' if the information is valuable for long-term storage.
5. If you cannot find information, state it clearly.
6. Your output should be structured as a list of facts or data points.

Focus on technical accuracy and sourcing."""

def get_researcher_agent(llm):
    """Returns a specialized researcher agent."""
    tools = [search_web, scrape_website, save_to_knowledge_base]
    
    # We create a ReAct agent but with a specific identity
    return create_react_agent(
        llm, 
        tools, 
        prompt=RESEARCHER_PROMPT
    )
