from pathlib import Path

import pandas as pd


DEFAULT_DATA_DIR = Path("data/raw/ml-32m")


def load_movies(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the MovieLens movies metadata."""
    file_path = data_dir / "movies.csv"

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Movies file not found: ${file_path}."
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