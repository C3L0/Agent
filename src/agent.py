from typing import Literal

from src.core.providers import get_llm
from src.workflows.hybrid_flow import get_hybrid_workflow


class MultiProviderAgent:
    def __init__(
        self,
        provider: Literal["openrouter", "ollama"] = "openrouter",
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
        # The hybrid workflow returns the final state after all nodes have executed.
        result = self.agent_executor.invoke({"messages": [("user", query)]})
        
        # Robust way to get the last message content
        last_message = result["messages"][-1]
        
        # Handle both AIMessage objects and dict/tuple formats
        if hasattr(last_message, "content"):
            return last_message.content
        elif isinstance(last_message, dict):
            return last_message.get("content", str(last_message))
        elif isinstance(last_message, tuple):
            return last_message[1]
        
        return str(last_message)


