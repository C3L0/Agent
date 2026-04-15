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

    # 4. Add Query Argument
    parser.add_argument(
        "query", type=str, nargs='?', help="The question or task for the agent (optional, will prompt if missing)"
    )

    args = parser.parse_args()

    # 5. Set Provider and Default Model
    if args.openrouter:
        default_model = os.getenv("OPENROUTER_MODEL")
        provider = "openrouter"
        model = args.model or default_model
    else:  # args.ollama
        default_model = os.getenv("OLLAMA_MODEL")
        provider = "ollama"
        model = args.model or default_model

    print(f"--- Starting Agent (Provider: {provider}, Model: {model}) ---")

    # 6. Get Query
    query = args.query
    if not query:
        query = input("\nWhat is your question? ")

    # 7. Initialize and Run the Agent
    try:
        agent = MultiProviderAgent(provider=provider, model=model)

        print(f"\nUser: {query}")

        response = agent.ask(query)
        print(f"\nFinal Answer:\n{response}")

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
