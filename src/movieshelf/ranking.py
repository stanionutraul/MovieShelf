
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
