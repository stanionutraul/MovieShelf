
from pydantic import BaseModel, Field

class ExampleInterpretation(BaseModel):
    movie_title: str
    relevant_mechanisms: list[str] = Field(default_factory=list)
    explanation: str


class MovieRequest(BaseModel):
    """Structured representation of a user's movie search request."""

    intent_summary: str = Field(
        description="A concise description of what the user is looking for."
    )

    narrative_mechanisms: list[str] = Field(
        default_factory=list,
        description="Narrative techniques relevant to the request."
    )

    required_criteria: list[str] = Field(
        default_factory=list,
        description="Explicit requirements that a movie must satisfy."
    )

    preferred_criteria: list[str] = Field(
        default_factory=list,
        description="Preferences that are desirable but not mandatory."
    )

    genres_include: list[str] = Field(
        default_factory=list,
        description="Genres explicitly requested by the user."
    )

    genres_exclude: list[str] = Field(
        default_factory=list,
        description="Genres explicitly excluded by the user."
    )

    exclude: list[str] = Field(
        default_factory=list,
        description="Other explicit exclusions."
    )

    spoilers_allowed: bool = Field(
        default=False,
        description="Whether the user explicitly allows spoilers."
    )

    assumptions: list[str] = Field(
        default_factory=list,
        description="Interpretations inferred from the request but not explicitly stated."
    )
    examples_interpretation: list[ExampleInterpretation] = Field(
        default_factory=list,
        description="Interpretation of movies mentioned as examples."
    )

    clarification_question: str | None = Field(
        default=None,
        description="One useful clarification question, or null if none is needed."
    )






