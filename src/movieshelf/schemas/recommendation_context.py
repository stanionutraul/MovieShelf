
from pydantic import BaseModel, Field

from movieshelf.schemas.movie_request import MovieRequest
from movieshelf.schemas.user_taste import UserTasteContext


class RecommendationContext(BaseModel):
    """Temporary context for a single movie recommendation request."""

    movie_request: MovieRequest | None = None

    taste_context: UserTasteContext | None = None

    filters: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="Explicit filters selected by the user."
    )

    random_recommendation: bool = Field(
        default=False,
        description="Whether the user requested a random movie recommendation."
    )
