import argparse
import os

from dotenv import load_dotenv

from src.agent import MultiProviderAgent


def main():
    load_dotenv()

    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(
        description="Run the Agentic AI with OpenRouter or Ollama."
    )

    # 2. Add Provider Flags
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--openrouter", action="store_true", help="Use OpenRouter (cloud models)"
    )
    group.add_argument(
        "--ollama", action="store_true", help="Use Ollama (local models)"
    )

    # 3. Add Optional Model Flag
    parser.add_argument(
        "--model", type=str, help="Specify the model to use (overrides defaults)"
    )

    args = parser.parse_args()

    # 4. Set Provider and Default Model
    if args.openrouter:
        default_model = os.getenv("OPENROUTER_MODEL")
        provider = "openrouter"
        model = args.model or default_model
    else:  # args.ollama
        default_model = os.getenv("OLLAMA_MODEL")
        provider = "ollama"
        model = args.model or default_model

    print(f"--- Starting Agent (Provider: {provider}, Model: {model}) ---")

    # 5. Initialize and Run the Agent
    try:
        agent = MultiProviderAgent(provider=provider, model=model)

        query = "What's the weather like in Paris?"
        print(f"User: {query}")

        response = agent.ask(query)
        print(f"\nFinal Answer: {response}")

    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        if provider == "ollama":
            print(
                "\nMake sure Ollama is running (`ollama serve`) and the model is pulled (`ollama pull <your-model>`)!"
            )


if __name__ == "__main__":
    main()
