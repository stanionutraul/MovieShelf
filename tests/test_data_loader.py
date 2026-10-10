
import pandas as pd
import pytest

from movieshelf.data_loader import load_movies


def test_load_movies_returns_expected_columns():
    movies = load_movies()

    assert {"movieId", "title", "genres"}.issubset(movies.columns)


def test_load_movies_has_unique_movie_ids():
    movies = load_movies()

    assert movies["movieId"].is_unique


def test_load_movies_returns_non_empty_dataframe():
    movies = load_movies()

    assert not movies.empty


def test_load_movies_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Movies file not found"):
        load_movies(tmp_path)


def test_load_movies_rejects_missing_columns(tmp_path):
    pd.DataFrame({"movieId": [1], "title": ["Toy Story"]}).to_csv(
        tmp_path / "movies.csv",
        index=False,
    )

    with pytest.raises(ValueError, match="missing required columns"):
        load_movies(tmp_path)


def test_load_movies_rejects_duplicate_movie_ids(tmp_path):
    pd.DataFrame(
        {
            "movieId": [1, 1],
            "title": ["Toy Story (1995)", "Toy Story (1995)"],
            "genres": ["Animation|Comedy", "Animation|Comedy"],
        }
    ).to_csv(tmp_path / "movies.csv", index=False)

    with pytest.raises(ValueError, match="duplicate movieId"):
        load_movies(tmp_path)
