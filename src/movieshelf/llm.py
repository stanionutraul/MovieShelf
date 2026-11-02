
from ollama import Client


class MovieShelfLLM:
    """Handles communication between MovieShelf and Ollama."""

    def __init__(self, model: str = "qwen3:4b"):
        self.model = model
        self.client = Client(host="http://localhost:11434")

    def generate_response(self, prompt: str) -> str:
        """Send a prompt to the LLM and return its response."""

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response.message.content
