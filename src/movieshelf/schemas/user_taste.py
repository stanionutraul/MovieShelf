
from pydantic import BaseModel, Field


class TasteReviewExample(BaseModel):
    """A short review excerpt that can ground a personalized recommendation."""

    title: str
    rating: float | None = None
    review: str


class TasteMovieRating(BaseModel):
    """A rated title and catalog genres used as a compact taste signal."""

    title: str
    rating: float
    genres: list[str] = Field(default_factory=list)


class GenreTasteSummary(BaseModel):
    """A genre rating average with the amount of supporting user data."""

    genre: str
    average_rating: float
    rated_movie_count: int


class TagTasteSummary(BaseModel):
    """MovieLens tag preference inferred from all mapped user ratings."""

    tag: str
    average_rating: float
    rated_movie_count: int


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

    preferred_genres: list[str] = Field(
        default_factory=list,
        description="Genres the user tends to rate highly."
    )

    less_preferred_genres: list[str] = Field(
        default_factory=list,
        description="Genres the user tends to rate poorly."
    )

    review_examples: list[TasteReviewExample] = Field(
        default_factory=list,
        description=(
            "A small selection of the user's own positive and negative "
            "reviews, truncated for recommendation context."
        )
    )

    rated_movie_examples: list[TasteMovieRating] = Field(
        default_factory=list,
        description=(
            "A compact sample of positively and negatively rated movies, "
            "with catalog genres for preference context."
        )
    )

    genre_rating_summary: list[GenreTasteSummary] = Field(
        default_factory=list,
        description=(
            "Mean Letterboxd ratings by catalog genre, with sample counts."
        )
    )

    tag_rating_summary: list[TagTasteSummary] = Field(
        default_factory=list,
        description=(
            "Mean Letterboxd ratings for MovieLens tags on rated movies, "
            "with sample counts."
        ),
    )

    taste_summary: str | None = Field(
        default=None,
        description="A concise summary of the user's movie preferences inferred from their Letterboxd data."
    )
