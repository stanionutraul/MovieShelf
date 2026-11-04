
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
            "num_predict": 512,
        }

        self.keep_alive = "10m"

    def generate_response(self, prompt: str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            options=self.options,
            keep_alive=self.keep_alive,
        )

        self.log_performance(response, "LLM operation")

        return response.message.content or ""

   
    def generate_structured(self, prompt: str, schema: type) -> dict:   
        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            format=schema.model_json_schema(),
            think=False,
            options={
                **self.options,
                "num_predict": 1024,
                "temperature": 0,
            },
            keep_alive=self.keep_alive,
        )

        self.log_performance(response, "Structured LLM operation")

        content = response.message.content or ""
        thinking = getattr(response.message, "thinking", None) or ""

        print(f"Thinking length: {len(thinking)} characters")
        print(f"Content length: {len(content)} characters")

        if not content.strip():
            raise RuntimeError(
                "Ollama returned an empty structured response. "
                f"Model: {self.model}. "
                f"Done reason: {getattr(response, 'done_reason', None)}. "
                f"Thinking length: {len(thinking)}."
            )

        try:
            return schema.model_validate_json(content).model_dump()

        except Exception as error:
            raise RuntimeError(
                "Ollama returned invalid structured output.\n"
                f"Raw response: {content!r}"
            ) from error

    def chat_with_tools(self, messages: list, tools: list):
        response = self.client.chat(
            model=self.model,
            messages=messages,
            tools=tools,
            think=False,
            options={
                **self.options,
                "num_predict": 1024,
            },
            keep_alive=self.keep_alive,
        )

        self.log_performance(response, "Chat with tools")

        tool_calls = response.message.tool_calls or []

        if not tool_calls:
            content = response.message.content or ""

            print("\n[TOOL CALL DIAGNOSTICS]")
            print("No tool calls were generated.")
            print("Generated content preview:")
            print(content[:3000])

        return response
    def log_performance(self, response, operation: str):
        metrics = {
            "total": response.total_duration / 1e9,
            "load": response.load_duration / 1e9,
            "prompt": response.prompt_eval_duration / 1e9,
            "generation": response.eval_duration / 1e9,
        }

        message = response.message
        content = message.content or ""
        thinking = getattr(message, "thinking", None) or ""
        tool_calls = getattr(message, "tool_calls", None) or []

        print(f"\n[{operation}]")
        print(f"Total: {metrics['total']:.2f}s")
        print(f"Loading: {metrics['load']:.2f}s")
        print(f"Prompt processing: {metrics['prompt']:.2f}s")
        print(f"Generation: {metrics['generation']:.2f}s")
        print(f"Done reason: {response.done_reason}")
        print(f"Thinking length: {len(thinking)} characters")
        print(f"Content length: {len(content)} characters")
        print(f"Tool calls: {len(tool_calls)}")
