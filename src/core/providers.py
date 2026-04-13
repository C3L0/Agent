import os
from typing import Literal

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def init_openrouter(model: str, temperature: float = 0) -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY must be set in your .env file for OpenRouter"
        )

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://localhost:3000",
            "X-Title": "My Local Agent",
        },
    )


def init_ollama(model: str, temperature: float = 0) -> ChatOllama:
    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url="http://localhost:11434",
    )


def get_llm(
    provider: Literal["openrouter", "ollama"], model: str, temperature: float = 0
):
    if provider == "openrouter":
        return init_openrouter(model, temperature)
    elif provider == "ollama":
        return init_ollama(model, temperature)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
