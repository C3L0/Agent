# Tech Watch Agent Architecture

This project is designed as a modular sandbox for building and testing agentic AI systems, specifically for technology watch ("veille technologique").

## Structure

```text
/home/antoine/project/agent/
├── data/                  # Local storage (ignored in git)
│   └── knowledge_base.json# Saved articles and summaries
├── src/
│   ├── core/              # Foundational setup
│   │   ├── __init__.py
│   │   ├── config.py      # Environment variables and settings
│   │   └── providers.py   # LLM initialization (OpenRouter, Ollama)
│   ├── tools/             # The "Skills" of your agent
│   │   ├── __init__.py
│   │   ├── search.py      # Web search (Tavily, DuckDuckGo)
│   │   ├── scrape.py      # RSS reading, HTML extraction
│   │   └── storage.py     # Saving/loading from data/
│   ├── workflows/         # Agentic logic (LangGraph)
│   │   ├── __init__.py
│   │   ├── react.py       # Simple React agent (current implementation)
│   │   └── researcher.py  # Multi-step graph (search -> read -> summarize)
│   └── agent.py           # The main interface class orchestrating tools and workflows
├── tests/                 # Unit tests for tools and workflows
├── main.py                # CLI entry point
├── pyproject.toml         # Dependencies
└── ARCHITECTURE.md        # This file
```

## Key Components

1.  **Core**: Manages the initialization of LLM providers and configuration.
2.  **Tools**: Individual functions decorated with `@tool` that the agent can invoke.
3.  **Workflows**: Different agent logic structures using LangGraph.
4.  **Data**: Local persistence for gathered information.
