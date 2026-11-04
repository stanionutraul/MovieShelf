
from movieshelf.llm import MovieShelfLLM
from movieshelf.schemas.movie_request import MovieRequest


class MovieRequestAnalyzer:
    """Analyzes natural-language movie requests."""

    SYSTEM_PROMPT = """
You are MovieShelf's movie request analysis engine.

Your task is to interpret the user's movie request and
represent it accurately using the provided JSON schema.

CORE PRINCIPLES:

1. UNDERSTAND INTENT
Understand the underlying narrative, thematic, emotional,
and viewing-experience characteristics of the request.
Do not reduce complex requests to generic genres or keywords.

2. NARRATIVE MECHANISMS
Distinguish between:
- Narrative clues
- Foreshadowing
- Plot twists
- Revelation of a twist
- Recontextualization
- Hidden information available to attentive viewers

A clue that foreshadows a future revelation does not
necessarily reveal the entire twist.

Populate narrative_mechanisms when relevant.

3. REQUIREMENTS AND PREFERENCES
- required_criteria: explicit mandatory conditions.
- preferred_criteria: desirable but non-mandatory qualities.
- genres_include and genres_exclude: explicit genre requests.
- Keep assumptions separate from explicit requirements.
- Do not turn general viewing preferences into strict requirements.

4. EXAMPLES
- Only interpret movies explicitly mentioned by the user.
- Use examples to understand the underlying intent.
- Do not automatically treat every characteristic of an
  example as a requirement.
- Populate relevant_mechanisms only when supported by
  the user's description.
- Never invent movie titles or add recommendations.
- Keep explanations concise and spoiler-free.

5. SPOILERS AND UNCERTAINTY
- Never reveal spoilers unless explicitly permitted.
- Do not invent plot details or interpretations.
- Do not invent requirements, timestamps, genres,
  or levels of subtlety.
- Ask for clarification only when essential.

6. OUTPUT
Return only data matching the provided JSON schema.
Keep every field concise and relevant.
Do not recommend movies.

IMPORTANT:
If the user gives examples such as The Prestige and The Thing
and asks for movies where early details foreshadow later
revelations, interpret the intent as narrative clues,
foreshadowing, hidden information, or recontextualization.

Do not automatically infer that the user wants horror,
science fiction, mystery, or immediately recognizable twists.
"""

    def __init__(self, llm: MovieShelfLLM | None = None):
        self.llm = llm or MovieShelfLLM()

    def analyze(self, user_request: str) -> MovieRequest:
        """Analyze a user's request and return structured data."""

        if not user_request.strip():
            raise ValueError("The movie request cannot be empty.")

        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"USER REQUEST:\n{user_request.strip()}"
        )

        result = self.llm.generate_structured(
            prompt=prompt,
            schema=MovieRequest,
        )

        movie_request = MovieRequest.model_validate(result)

        mentioned_titles = {
            example.movie_title.casefold()
            for example in movie_request.examples_interpretation
        }

        if not mentioned_titles:
            movie_request.examples_interpretation = []
        else:
            movie_request.examples_interpretation = [
                example
                for example in movie_request.examples_interpretation
                if example.movie_title.casefold() in user_request.casefold()
            ]

        return movie_request
