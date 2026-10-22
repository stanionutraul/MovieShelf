from pathlib import Path

import pandas as pd


DEFAULT_DATA_DIR = Path("data/raw/ml-32m")


def load_movies(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the MovieLens movies metadata."""
    file_path = data_dir / "movies.csv"

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Movies file not found: {file_path}. "
            "Check that the MovieLens dataset is extracted correctly."
        )

    movies = pd.read_csv(file_path)

    required_columns = {"movieId", "title", "genres"}

    missing_columns = required_columns - set(movies.columns)

    if missing_columns:
        raise ValueError(
            f"Movies.csv is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    if movies["movieId"].duplicated().any():
        raise ValueError("Movies.csv contains duplicate movieId values.")

    return movies




def load_ratings(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load MovieLens ratings using memory-efficient data types."""
    file_path = data_dir / "ratings.csv"

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Ratings file not found: {file_path}. "
            "Check that the MovieLens dataset is extracted correctly."
        )

    ratings = pd.read_csv(
        file_path,
        dtype={
            "userId": "Int32",
            "movieId": "Int32",
            "rating": "float32",
            "timestamp": "int32",
        },
    )

    required_columns = {"userId", "movieId", "rating", "timestamp"}
    missing_columns = required_columns - set(ratings.columns)

    if missing_columns:
        raise ValueError(
            f"ratings.csv is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

   
    if ratings[["userId", "movieId"]].isna().any().any():
        raise ValueError("ratings.csv contains missing userId or movieId values.")

    ratings["userId"] = ratings["userId"].astype("int32")
    ratings["movieId"] = ratings["movieId"].astype("int32")

    return ratings



def load_tags(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load MovieLens tags using memory-efficient data types."""
    file_path = data_dir / "tags.csv"

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Tags file not found: {file_path}. "
            "Check that the MovieLens dataset is extracted correctly."
        )

    tags = pd.read_csv(
        file_path,
        dtype={
            "userId": "int32",
            "movieId": "int32",
            "timestamp": "int32",
            "tag": "string",
        },
    )

    required_columns = {"userId", "movieId", "tag", "timestamp"}
    missing_columns = required_columns - set(tags.columns)

    if missing_columns:
        raise ValueError(
            f"tags.csv is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    return tags



         