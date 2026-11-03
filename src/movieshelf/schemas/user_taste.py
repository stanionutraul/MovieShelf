
from pydantic import BaseModel, Field


class UserTasteContext(BaseModel):
    """Temporary representation of movie preferences inferred from Letterboxd data."""

    liked_movie_ids: list[int] = Field(
        default_factory=list,
        description="MovieLens IDs of movies the user rated positively."
    )

    disliked_movie_ids: list[int] = Field(
        default_factory=list,
        description="MovieLens IDs of movies the user rated negatively."
    )

    watched_movie_ids: list[int] = Field(
        default_factory=list,
        description="MovieLens IDs of movies the user has already watched."
    )

    taste_summary: str | None = Field(
        default=None,
        description="A concise summary of the user's movie preferences inferred from their Letterboxd data."
    )
