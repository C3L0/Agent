import trafilatura
from langchain_core.tools import tool


@tool
def scrape_website(url: str) -> str:
    """Extract and clean the main text content from a given URL."""
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return f"Error: Could not fetch content from {url}"

    # Extract text content, removing boilerplate like menus and ads
    result = trafilatura.extract(downloaded, include_comments=False, include_tables=True)

    if not result:
        return f"Error: No readable content extracted from {url}"

    return result[:5000]  # Limit to 5000 characters to stay within LLM context
