import subprocess
import sys

from src.model import create_llm

def pull_model(model_name: str):
    try:
        # Attempt to pull the model using the command line
        subprocess.run([sys.executable, '-m', 'ollama', 'pull', model_name], check=True)
        print(f"Model {model_name} pulled successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to pull model {model_name}: {e}")

def initialize_model(model_name: str):
    try:
        # Attempt to create the language model instance
        llm = create_llm(model_name)  # Make sure your create_llm function accepts model names
        return llm
    except Exception as e:
        print(f"Error initializing model {model_name}: {e}")
        return None

if __name__ == "__main__":
    model_name = "llama-3.1:8b"  # Specify your model name

    # Pull the model if it's not already available
    pull_model(model_name)

    # Initialize the model
    llm = initialize_model(model_name)
    if llm is None:
        print(f"Model {model_name} could not be initialized. Exiting.")
        sys.exit(1)

    # Continue with the rest of your code, such as creating the chain
    # ...
