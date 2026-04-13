from langchain_core.tools import tool


@tool
def get_weather(location: str) -> str:
    """Get the current weather in a location."""
    return f"The weather in {location} is sunny and 22°C."
