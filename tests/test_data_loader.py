
import pandas as pd
import pytest

from movieshelf.data_loader import load_movies, load_ratings, load_tags


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




def test_load_ratings_returns_expected_columns(tmp_path):
    pd.DataFrame(
        {
            "userId": [1, 2],
            "movieId": [17, 25],
            "rating": [4.0, 3.5],
            "timestamp": [944249077, 944250228],
        }
    ).to_csv(tmp_path / "ratings.csv", index=False)

    ratings = load_ratings(tmp_path)

    assert {"userId", "movieId", "rating", "timestamp"}.issubset(
        ratings.columns
    )


def test_load_ratings_uses_memory_efficient_dtypes(tmp_path):
    pd.DataFrame(
        {
            "userId": [1, 2],
            "movieId": [17, 25],
            "rating": [4.0, 3.5],
            "timestamp": [944249077, 944250228],
        }
    ).to_csv(tmp_path / "ratings.csv", index=False)

    ratings = load_ratings(tmp_path)

    assert str(ratings["userId"].dtype) == "int32"
    assert str(ratings["movieId"].dtype) == "int32"
    assert str(ratings["rating"].dtype) == "float32"
    assert str(ratings["timestamp"].dtype) == "int32"



def test_load_ratings_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Ratings file not found"):
        load_ratings(tmp_path)


def test_load_ratings_rejects_missing_columns(tmp_path):
    pd.DataFrame(
        {
            "userId": [1],
            "movieId": [17],
            "rating": [4.0],
        }
    ).to_csv(tmp_path / "ratings.csv", index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_ratings(tmp_path)


def test_load_ratings_rejects_missing_identifiers(tmp_path):
    pd.DataFrame(
        {
            "userId": [1, None],
            "movieId": [17, 25],
            "rating": [4.0, 3.5],
            "timestamp": [944249077, 944250228],
        }
    ).to_csv(tmp_path / "ratings.csv", index=False)

    with pytest.raises(ValueError, match="missing userId or movieId"):
        load_ratings(tmp_path)



def test_load_tags_returns_expected_columns_and_types(tmp_path):
    file_path = tmp_path / "tags.csv"

    file_path.write_text(
        "userId,movieId,tag,timestamp\n"
        "1,10,War,1000\n"
        "2,10,Realistic,2000\n",
        encoding="utf-8",
    )

    result = load_tags(tmp_path)

    assert list(result.columns) == [
        "userId",
        "movieId",
        "tag",
        "timestamp",
    ]
    assert len(result) == 2
    assert str(result["userId"].dtype) == "int32"
    assert str(result["movieId"].dtype) == "int32"
    assert str(result["timestamp"].dtype) == "int32"
    assert result["tag"].iloc[0] == "War"


def test_load_tags_raises_for_missing_file(tmp_path):
    try:
        load_tags(tmp_path)
    except FileNotFoundError as error:
        assert "tags.csv" in str(error)
    else:
        raise AssertionError("Expected FileNotFoundError")


def test_load_tags_raises_for_missing_columns(tmp_path):
    file_path = tmp_path / "tags.csv"

    file_path.write_text(
        "userId,movieId,tag\n"
        "1,10,War\n",
        encoding="utf-8",
    )

    try:
        load_tags(tmp_path)
    except ValueError as error:
        assert "timestamp" in str(error)
    else:
        raise AssertionError("Expected ValueError")
