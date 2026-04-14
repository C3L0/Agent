import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from src.agents.writer import get_writer_node, WRITER_PROMPT

def test_writer_node_calls_llm_with_correct_prompt():
    # 1. Setup Mock LLM
    mock_llm = MagicMock()
    mock_response = AIMessage(content="Ceci est un rapport structuré.")
    mock_llm.invoke.return_value = mock_response
    
    # 2. Get the Writer Node
    writer_node = get_writer_node(mock_llm)
    
    # 3. Prepare an initial state
    state = {
        "messages": [HumanMessage(content="Voici des faits sur Python : 1. C'est un langage. 2. Créé par Guido.")]
    }
    
    # 4. Execute the node
    result = writer_node(state)
    
    # 5. Assertions
    # Vérifier que le LLM a été appelé
    assert mock_llm.invoke.called
    
    # Vérifier que le premier message envoyé au LLM est bien notre WRITER_PROMPT
    call_args = mock_llm.invoke.call_args[0][0]
    assert isinstance(call_args[0], SystemMessage)
    assert call_args[0].content == WRITER_PROMPT
    
    # Vérifier que le résultat est ajouté aux messages
    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "Ceci est un rapport structuré."

def test_writer_node_preserves_history():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Rapport")
    writer_node = get_writer_node(mock_llm)
    
    history = [
        HumanMessage(content="Fact 1"),
        AIMessage(content="Received"),
        HumanMessage(content="Fact 2")
    ]
    state = {"messages": history}
    
    writer_node(state)
    
    # Vérifier que tout l'historique a été envoyé au LLM (System Prompt + history)
    sent_messages = mock_llm.invoke.call_args[0][0]
    assert len(sent_messages) == len(history) + 1 # +1 pour le SystemMessage
    assert sent_messages[1].content == "Fact 1"
