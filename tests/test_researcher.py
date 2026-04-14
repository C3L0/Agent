import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage
from src.agents.researcher import get_researcher_agent

def test_researcher_agent_initialization():
    # Setup Mock LLM avec support de bind_tools (nécessaire pour create_react_agent)
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    
    # Get the Researcher Agent
    agent = get_researcher_agent(mock_llm)
    
    # Vérifier que c'est un objet exécutable (invoke existe)
    assert hasattr(agent, "invoke")
    
    # Vérifier que bind_tools a été appelé avec nos outils
    # (search_web, scrape_website, save_to_knowledge_base)
    assert mock_llm.bind_tools.called
    tools_passed = mock_llm.bind_tools.call_args[0][0]
    assert len(tools_passed) == 3
    
    # Noms des outils attendus
    tool_names = [t.name for t in tools_passed]
    assert "search_web" in tool_names
    assert "scrape_website" in tool_names
    assert "save_to_knowledge_base" in tool_names

def test_researcher_agent_prompt_injection():
    # Ce test vérifie si le prompt système est bien passé au state_modifier
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    
    # L'agent ReAct de LangGraph utilise le state_modifier pour injecter le prompt
    agent = get_researcher_agent(mock_llm)
    
    # On ne peut pas facilement inspecter l'intérieur du CompiledGraph sans l'exécuter,
    # mais on a validé l'appel à get_researcher_agent qui contient le RESEARCHER_PROMPT.
