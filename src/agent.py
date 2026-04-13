from typing import Literal

from src.core.providers import get_llm
from src.tools.scrape import scrape_website
from src.tools.search import search_web
from src.tools.storage import save_to_knowledge_base
from src.tools.weather import get_weather
from src.workflows.react import get_react_agent


class MultiProviderAgent:
    def __init__(
        self,
        provider: Literal["openrouter", "ollama"] = "openrouter",
        model: str = "google/gemma-4-31b-it:free",
        temperature: float = 0,
    ):
        # 1. Initialize the LLM based on the provider
        self.llm = get_llm(provider, model, temperature)

        # 2. Define tools
        self.tools = [get_weather, search_web, scrape_website, save_to_knowledge_base]

        # 3. Setup the Agent Workflow
        self.agent_executor = get_react_agent(self.llm, self.tools)

    def ask(self, query: str) -> str:
        """Main entry point to interact with the agent."""
        # LangGraph agents return a stream of states. We just want the last one.
        result = self.agent_executor.invoke({"messages": [("user", query)]})
        # The last message in the state is the agent's response
        return result["messages"][-1].content
