from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage
from src.agents.educator import get_educator_node

def test_educator_node():
    # Setup
    llm = MagicMock()
    # Simulate LLM response for educator
    llm.invoke.return_value = AIMessage(content="## Educational Context\n- **Transformer**: A type of neural network architecture.")
    
    educator_node = get_educator_node(llm)
    
    # State containing researcher's output
    state = {
        "messages": [
            HumanMessage(content="What is a Transformer?"),
            AIMessage(content="The researcher found that Transformers are used in many AI models.")
        ]
    }
    
    # Execution
    result = educator_node(state)
    
    # Verification
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert "## Educational Context" in result["messages"][0].content
    assert "Transformer" in result["messages"][0].content
    llm.invoke.assert_called_once()
