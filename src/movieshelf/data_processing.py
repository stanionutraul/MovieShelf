
import pandas as pd


def build_movie_statistics(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row of rating statistics per movie."""
    unknown_movie_ids = (
        set(ratings["movieId"]) - set(movies["movieId"])
    )

    if unknown_movie_ids:
        raise ValueError(
            "Ratings contain movieId values not found in movies: "
            f"{sorted(unknown_movie_ids)}"
        )


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



def normalize_tags(tags: pd.Series) -> pd.Series:
    """Normalize tag text by trimming whitespace and lowercasing."""
    return tags.str.strip().str.lower()


def summarize_movie_tags(tags: pd.DataFrame) -> pd.DataFrame:
    """Normalize movie tags and count each unique movie-tag pair."""

    required_columns = {"movieId", "tag"}
    missing_columns = required_columns - set(tags.columns)

    if missing_columns:
        raise ValueError(
            "Tags DataFrame is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    result = tags[["movieId", "tag"]].copy()

    result["tag"] = normalize_tags(result["tag"])
    result = result.dropna(subset=["tag"])
    result = result[result["tag"] != ""]

    if result.empty:
        return pd.DataFrame(
            columns=["movieId", "tag", "tag_count"]
        )

    summary = (
        result.groupby(["movieId", "tag"])
        .size()
        .reset_index(name="tag_count")
        .sort_values(["movieId", "tag"])
        .reset_index(drop=True)
    )

    return summary
