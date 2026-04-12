import os
from typing import Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# Load environment variables
load_dotenv()

class MultiProviderAgent:
    def __init__(self, 
                 provider: Literal["openrouter", "ollama"] = "openrouter",
                 model: str = "openai/gpt-4o-mini", 
                 temperature: float = 0):
        
        self.provider = provider
        self.model = model
        self.temperature = temperature
        
        # 1. Initialize the LLM based on the provider
        if provider == "openrouter":
            self.llm = self._init_openrouter()
        elif provider == "ollama":
            self.llm = self._init_ollama()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # 2. Define tools
        self.tools = [self.get_weather]
        
        # 3. Setup the Agent using LangGraph (the modern AgentExecutor)
        # It takes the LLM and the tools directly.
        self.agent_executor = create_react_agent(self.llm, self.tools)

    def _init_openrouter(self) -> ChatOpenAI:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY must be set in your .env file for OpenRouter")

        return ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://localhost:3000",
                "X-Title": "My Local Agent",
            }
        )

    def _init_ollama(self) -> ChatOllama:
        return ChatOllama(
            model=self.model,
            temperature=self.temperature,
            base_url="http://localhost:11434"
        )

    @tool
    def get_weather(location: str) -> str:
        """Get the current weather in a location."""
        return f"The weather in {location} is sunny and 22°C."

    def ask(self, query: str) -> str:
        """Main entry point to interact with the agent."""
        # LangGraph agents return a stream of states. We just want the last one.
        result = self.agent_executor.invoke({"messages": [("user", query)]})
        # The last message in the state is the agent's response
        return result["messages"][-1].content
