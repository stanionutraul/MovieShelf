
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
    normalized = tags.copy()

    suspicious_mask = normalized.str.contains(
        r"Ã|Â|â€|ï¿½",
        regex=True,
        na=False,
    )

    normalized.loc[suspicious_mask] = (
        normalized.loc[suspicious_mask].map(repair_mojibake)
    )

    return normalized.str.strip().str.lower()



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




def build_movie_profiles(
    movies: pd.DataFrame,
    movie_tags: pd.DataFrame,
    max_tags: int = 100,
) -> pd.DataFrame:
    """Build one text profile per movie using its title, genres, and tags."""
    if max_tags <= 0:
        raise ValueError("max_tags must be positive")

    required_movie_columns = {"movieId", "title", "genres"}
    missing_movie_columns = required_movie_columns - set(movies.columns)
    if missing_movie_columns:
        raise ValueError(
            "Movies DataFrame is missing required columns: "
            f"{', '.join(sorted(missing_movie_columns))}"
        )

    required_tag_columns = {"movieId", "tag", "tag_count"}
    missing_tag_columns = required_tag_columns - set(movie_tags.columns)
    if missing_tag_columns:
        raise ValueError(
            "Movie tags DataFrame is missing required columns: "
            f"{', '.join(sorted(missing_tag_columns))}"
        )

    tags = movie_tags.sort_values(
        ["movieId", "tag_count", "tag"],
        ascending=[True, False, True],
    )

    tags = tags.drop_duplicates(subset=["movieId", "tag"])
    tags = tags.groupby("movieId", sort=False).head(max_tags)

    tags_by_movie = (
        tags.groupby("movieId", sort=False)["tag"]
        .agg(list)
        .to_dict()
    )

    profiles = movies[["movieId", "title", "genres"]].copy()

    def create_profile(row: pd.Series) -> str:
        title = str(row["title"])
        genres = row["genres"]

        if pd.isna(genres) or not str(genres).strip():
            genres_text = "Unknown"
        else:
            genres_text = str(genres).replace("|", ", ")

        movie_tag_list = tags_by_movie.get(row["movieId"], [])

        if movie_tag_list:
            tags_text = ", ".join(movie_tag_list)
        else:
            tags_text = "None available"

        return (
            f"{title}. "
            f"Genres: {genres_text}. "
            f"Descriptive tags: {tags_text}."
        )

    profiles["profile_text"] = profiles.apply(create_profile, axis=1)

    return profiles[["movieId", "profile_text"]]




def repair_mojibake(text: str, max_attempts: int = 3) -> str:
    """Attempt to repair text decoded with the wrong character encoding."""
    if not isinstance(text, str):
        return text

    # Avoid modifying text that has no common encoding-corruption markers.
    markers = ("Ã", "Â", "â€", "ï¿½")
    if not any(marker in text for marker in markers):
        return text

    current = text

    for _ in range(max_attempts):
        try:
            corrected = current.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break

        if corrected == current:
            break

        current = corrected

    return current
