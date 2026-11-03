
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

        self.options = {
            "num_ctx": 4096,
        }

        self.keep_alive = "10m"

    def generate_response(self, prompt: str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options=self.options,
            keep_alive=self.keep_alive,
        )
        self.log_performance(response, "LLM operation")

        return response.message.content

    def generate_structured(self, prompt: str, schema: type) -> dict:
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format=schema.model_json_schema(),
            options={
                **self.options,
                "temperature": 0,
            },
            keep_alive=self.keep_alive,
        )
        self.log_performance(response, "LLM operation")

        content = response.message.content

        if not content or not content.strip():
            raise RuntimeError(
                f"Ollama returned an empty response. Model: {self.model}"
            )

        try:
            return schema.model_validate_json(content).model_dump()
        except Exception as error:
            raise RuntimeError(
                f"Ollama returned invalid structured output.\n"
                f"Raw response: {content!r}"
            ) from error

    def chat_with_tools(
    self,
    messages: list[dict],
    tools: list[dict],
):
        """Sends a conversation to Ollama with tool-calling support."""

        response = self.client.chat(
            model=self.model,
            messages=messages,
            tools=tools,
            options=self.options,
            keep_alive=self.keep_alive,
        )

        self.log_performance(response, "Chat with tools")
        
        return response

    def log_performance(self, response, operation: str):
        metrics = {
            "total": response.total_duration / 1e9,
            "load": response.load_duration / 1e9,
            "prompt": response.prompt_eval_duration / 1e9,
            "generation": response.eval_duration / 1e9,
        }

        print(f"\n[{operation}]")
        print(f"Total: {metrics['total']:.2f}s")
        print(f"Loading: {metrics['load']:.2f}s")
        print(f"Prompt processing: {metrics['prompt']:.2f}s")
        print(f"Generation: {metrics['generation']:.2f}s")