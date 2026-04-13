from ddgs import DDGS
from langchain_core.tools import tool


@tool
def search_web(query: str) -> str:
    """Search the web for news, technology updates, or information."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            results.append(f"Title: {r['title']}\nSnippet: {r['body']}\nSource: {r['href']}\n")
    return "\n".join(results)
