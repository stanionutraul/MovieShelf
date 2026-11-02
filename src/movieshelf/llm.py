
from ollama import Client


class MovieShelfLLM:
    """Handles communication with a local Ollama model."""

    def __init__(
        self,
        model: str = "qwen3:4b",
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.client = Client(host=host)

    def generate_response(self, prompt: str) -> str:
        """Generate a regular text response."""

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

    def generate_structured(
    self,
    prompt: str,
    schema: type,
    ) -> dict:
     """Generate and validate a structured JSON response."""

     response = self.client.chat(
        model=self.model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        format=schema.model_json_schema(),
        options={"temperature": 0},
    )

     content = response.message.content

     if not content or not content.strip():
        raise RuntimeError(
            f"Ollama returned an empty response. "
            f"Model: {self.model}. "
            f"Response: {response}"
        )

     try:
        
            return schema.model_validate_json(content).model_dump()

     except Exception as error:
        raise RuntimeError(
            f"Ollama returned invalid structured output.\n"
            f"Raw response: {content!r}"
        ) from error