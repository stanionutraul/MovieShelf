
from typing import Literal

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


class ViewingExperience(BaseModel):
    atmosphere: list[str] = Field(
        default_factory=list,
        description=(
            "Explicitly requested atmosphere qualities, such as cozy, "
            "warm, comforting, or whimsical."
        ),
    )

    tone: list[str] = Field(
        default_factory=list,
        description=(
            "Explicitly requested tone qualities, such as lighthearted, "
            "optimistic, uplifting, or feel-good."
        ),
    )

    mental_effort: Literal["low", "moderate", "high"] | None = Field(
        default=None,
        description=(
            "Requested cognitive effort required from the viewer. "
            "Set only when supported by the user's wording."
        ),
    )

    emotional_intensity: Literal["low", "moderate", "high"] | None = Field(
        default=None,
        description=(
            "Requested emotional intensity. Do not infer this from "
            "atmosphere or tone."
        ),
    )

    pacing: Literal["slow", "moderate", "fast"] | None = Field(
        default=None,
        description=(
            "Requested pacing. Set only when explicitly supported."
        ),
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

    requires_plot_evidence: bool = Field(
        default=False,
        description=(
            "True when evaluating the request requires verified facts "
            "about a movie's plot, characters, or narrative events; false "
            "for genre, mood, style, or viewing-context preferences."
        ),
    )

    required_criteria: list[str] = Field(
        default_factory=list,
        description="Explicit mandatory conditions."
    )

    preferred_criteria: list[str] = Field(
        default_factory=list,
        description="Desirable but non-mandatory characteristics."
    )

    non_constraints: list[str] = Field(
        default_factory=list,
        description=(
            "Aspects the user explicitly says are not required. These are "
            "not preferences, exclusions, or search criteria."
        ),
    )

    genres_include: list[str] = Field(
        default_factory=list,
        description="Genres explicitly requested."
    )

    genres_match: Literal["any", "all"] = Field(
        default="any",
        description=(
            "Whether included genres use OR ('any') or AND ('all') "
            "matching. Defaults to 'any'."
        ),
    )

    genres_exclude: list[str] = Field(
        default_factory=list,
        description="Genres explicitly excluded."
    )

    release_year_from: int | None = Field(
        default=None,
        ge=1870,
        le=2100,
        description="Earliest release year explicitly requested."
    )

    release_year_to: int | None = Field(
        default=None,
        ge=1870,
        le=2100,
        description="Latest release year explicitly requested."
    )

    content_avoidances: list[Literal["gore", "jump scares"]] = Field(
        default_factory=list,
        description=(
            "Explicitly unwanted content types, populated only for clear "
            "requests to avoid them."
        ),
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

    viewing_experience: ViewingExperience = Field(
        default_factory=ViewingExperience,
        description=(
            "Structured viewing-experience preferences explicitly "
            "expressed by the user."
        ),
    )
