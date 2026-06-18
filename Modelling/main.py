#main.py

from using_libraries.scientific_ai_engine import ScientificAIEngine

def print_banner():
    print("=" * 50)
    print("Scientific AI Framework (v0.1-alpha)")
    print("=" * 50)
    print("Please Enter Your Query")
    print("Type 'quit' or 'exit' to leave.")
    print()


def main():

    from pathlib import Path

    CHECKPOINT_PATH = (
        Path(__file__).resolve().parent
        / "models"
        / "checkpoints"
        / "scientific_intent_v1.pt"
    )

    engine = ScientificAIEngine(str(CHECKPOINT_PATH))
    print_banner()
    
    while True:

        query = input("> ").strip()

        if query.lower() in {"quit", "exit"}:
            break

        if not query:
            continue

        try:
            result = engine.run(query)
            print(result)

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
