
import pandas as pd


def build_movie_statistics(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row of rating statistics per movie."""
    rating_stats = (
        ratings.groupby("movieId")
        .agg(
            rating_count=("rating", "count"),
            average_rating=("rating", "mean"),
        )
        .reset_index()
    )

    movie_stats = movies.merge(
        rating_stats,
        on="movieId",
        how="left",
        validate="one_to_one",
    )

    movie_stats["rating_count"] = (
        movie_stats["rating_count"].fillna(0).astype("int64")
    )

    return movie_stats
