
import pandas as pd


def calculate_weighted_rating(
    movies: pd.DataFrame,
    global_mean: float,
    minimum_votes: int = 1_000,
) -> pd.DataFrame:
    """Calculate a weighted rating score for each movie."""

    if minimum_votes <= 0:
        raise ValueError("minimum_votes must be positive")

    if not 0.5 <= global_mean <= 5:
        raise ValueError("global_mean must be between 1 and 5")

    result = movies.copy()

    vote_count = result["rating_count"]
    movie_mean = result["average_rating"]

    result["weighted_rating"] = (
        vote_count / (vote_count + minimum_votes) * movie_mean
        + minimum_votes / (vote_count + minimum_votes) * global_mean
    )

    result.loc[vote_count == 0, "weighted_rating"] = float("nan")

    return result


def recommend_top_movies(
    movies: pd.DataFrame,
    n: int = 10,
    minimum_votes: int = 1_000,
) -> pd.DataFrame:
    """Return the highest-ranked movies meeting the minimum vote count."""

    if n <= 0:
        raise ValueError("n must be positive")

    if minimum_votes <= 0:
        raise ValueError("minimum_votes must be positive")

    required_columns = {"rating_count", "weighted_rating"}
    missing_columns = required_columns - set(movies.columns)

    if missing_columns:
        raise ValueError(
            "Movies DataFrame is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    eligible_movies = movies.loc[
        (movies["rating_count"] >= minimum_votes)
        & movies["weighted_rating"].notna()
    ].copy()

    ranked_movies = eligible_movies.sort_values(
        by="weighted_rating",
        ascending=False,
    )

    return ranked_movies.head(n).reset_index(drop=True)

