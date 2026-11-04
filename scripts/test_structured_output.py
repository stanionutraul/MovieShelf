
from movieshelf.llm import MovieShelfLLM
from movieshelf.schemas.movie_request import MovieRequest


def main():
    llm = MovieShelfLLM()

    prompt = """
Analyze this movie request and return structured data.

User request:
I want movies about secret agencies dealing with
supernatural phenomena or unexplained anomalies.

Examples:
- The Adjustment Bureau
- Men in Black

I do not want TV shows.

Do not recommend movies.
Do not invent movie titles or plot details.
Return concise data matching the JSON schema.
"""

    print("Testing structured output with minimal prompt...")
    print("Request: secret agencies and supernatural anomalies")

    result = llm.generate_structured(
        prompt=prompt,
        schema=MovieRequest,
    )

    print("\nStructured result:")
    print(result)


if __name__ == "__main__":
    main()
