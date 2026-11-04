
from pydantic import BaseModel, Field


class ExampleInterpretation(BaseModel):
    """Interpretation of a movie explicitly mentioned by the user."""

    movie_title: str = Field(
        description="Exact movie title mentioned by the user."
    )

    relevant_mechanisms: list[str] = Field(
        default_factory=list,
        description="Narrative mechanisms relevant to the request."
    )

    explanation: str = Field(
        description="Brief explanation based only on the user's description."
    )


class MovieRequest(BaseModel):
    """Structured representation of a movie search request."""

    intent_summary: str = Field(
        description="Concise summary of the user's movie search intent."
    )

    narrative_mechanisms: list[str] = Field(
        default_factory=list,
        description="Relevant narrative techniques and mechanisms."
    )

    required_criteria: list[str] = Field(
        default_factory=list,
        description="Explicit mandatory conditions."
    )

    preferred_criteria: list[str] = Field(
        default_factory=list,
        description="Desirable but non-mandatory characteristics."
    )

    genres_include: list[str] = Field(
        default_factory=list,
        description="Genres explicitly requested."
    )

    genres_exclude: list[str] = Field(
        default_factory=list,
        description="Genres explicitly excluded."
    )

    exclude: list[str] = Field(
        default_factory=list,
        description="Other explicit exclusions."
    )

    spoilers_allowed: bool = Field(
        default=False,
        description="True only when the user explicitly allows spoilers."
    )

    assumptions: list[str] = Field(
        default_factory=list,
        description="Reasonable interpretations not explicitly stated."
    )

    examples_interpretation: list[ExampleInterpretation] = Field(
        default_factory=list,
        description="Interpretations of explicitly mentioned movie examples."
    )

    clarification_question: str | None = Field(
        default=None,
        description="One necessary clarification question, otherwise null."
    )
