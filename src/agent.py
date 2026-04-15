from typing import Literal

from langchain_core.messages import HumanMessage

from src.core.providers import get_llm
from src.workflows.hybrid_flow import get_hybrid_workflow


class MultiProviderAgent:
    def __init__(
        self,
        provider: Literal["openrouter", "ollama"] = "ollama",
        model: str = "google/gemma-4-31b-it:free",
        temperature: float = 0,
    ):
        # 1. Initialize the LLM based on the provider
        self.llm = get_llm(provider, model, temperature)

        # 2. Setup the Multi-Agent Hybrid Workflow
        # This workflow handles routing, research, and writing
        self.agent_executor = get_hybrid_workflow(self.llm)

    def ask(self, query: str) -> str:
        """Main entry point to interact with the multi-agent system."""
        # Use HumanMessage object instead of a tuple for better internal graph state
        result = self.agent_executor.invoke({"messages": [HumanMessage(content=query)]})

        # The last message in the 'messages' list is the Writer's final report.
        return result["messages"][-1].content
