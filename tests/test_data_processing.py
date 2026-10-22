
import pandas as pd
import pytest

from movieshelf.data_processing import (
    build_movie_statistics,
    normalize_tags,
    summarize_movie_tags,
)


def test_build_movie_statistics_calculates_count_and_average():
    movies = pd.DataFrame(
        {
            "movieId": [1, 2],
            "title": ["Toy Story (1995)", "Jumanji (1995)"],
            "genres": ["Animation|Comedy", "Adventure|Children"],
        }
    )

    ratings = pd.DataFrame(
        {
            "userId": [10, 11, 12, 10],
            "movieId": [1, 1, 1, 2],
            "rating": [2.0, 4.0, 5.0, 3.0],
            "timestamp": [100, 200, 300, 400],
        }
    )

    result = build_movie_statistics(movies, ratings)

    toy_story = result.loc[result["movieId"] == 1].iloc[0]

    assert toy_story["rating_count"] == 3
    assert abs(toy_story["average_rating"] - 11 / 3) < 1e-6


def test_build_movie_statistics_keeps_movies_without_ratings():
    movies = pd.DataFrame(
        {
            "movieId": [1, 2],
            "title": ["Toy Story (1995)", "Jumanji (1995)"],
            "genres": ["Animation|Comedy", "Adventure|Children"],
        }
    )

    ratings = pd.DataFrame(
        {
            "userId": [10, 11],
            "movieId": [1, 1],
            "rating": [4.0, 5.0],
            "timestamp": [100, 200],
        }
    )

    result = build_movie_statistics(movies, ratings)

    jumanji = result.loc[result["movieId"] == 2].iloc[0]

    assert len(result) == 2
    assert jumanji["rating_count"] == 0
    assert pd.isna(jumanji["average_rating"])


def test_build_movie_statistics_rejects_unknown_movie_ids():
    movies = pd.DataFrame(
        {
            "movieId": [1],
            "title": ["Toy Story (1995)"],
            "genres": ["Animation|Comedy"],
        }
    )

    ratings = pd.DataFrame(
        {
            "userId": [10, 11],
            "movieId": [1, 99],
            "rating": [4.0, 5.0],
            "timestamp": [100, 200],
        }
    )

    with pytest.raises(
        ValueError,
        match="Ratings contain movieId values not found in movies",
    ):
        build_movie_statistics(movies, ratings)


def test_normalize_tags_lowercases_and_strips_whitespace():
    tags = pd.Series([
        "World War II",
        "  Atmospheric  ",
        "REALISTIC",
        "dark comedy",
    ])

    result = normalize_tags(tags)

    expected = pd.Series([
        "world war ii",
        "atmospheric",
        "realistic",
        "dark comedy",
    ])

    pd.testing.assert_series_equal(result, expected)


def test_normalize_tags_preserves_missing_values():
    tags = pd.Series(["War", None, "  "])

    result = normalize_tags(tags)

    assert result.iloc[0] == "war"
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == ""


def test_summarize_movie_tags_normalizes_and_counts_tags():
    tags = pd.DataFrame({
        "movieId": [10, 10, 10, 20, 20],
        "tag": ["War", "war", "Tanks", "War", "  war  "],
    })

    result = summarize_movie_tags(tags)

    expected = pd.DataFrame({
        "movieId": [10, 10, 20],
        "tag": ["tanks", "war", "war"],
        "tag_count": [1, 2, 2],
    })

    pd.testing.assert_frame_equal(
        result,
        expected,
        check_dtype=False,
    )


def test_summarize_movie_tags_ignores_missing_and_empty_tags():
    tags = pd.DataFrame({
        "movieId": [10, 10, 10, 20],
        "tag": ["War", None, "   ", ""],
    })

    result = summarize_movie_tags(tags)

    expected = pd.DataFrame({
        "movieId": [10],
        "tag": ["war"],
        "tag_count": [1],
    })

    pd.testing.assert_frame_equal(
        result,
        expected,
        check_dtype=False,
    )


def test_summarize_movie_tags_returns_empty_result_when_no_valid_tags():
    tags = pd.DataFrame({
        "movieId": [10, 20],
        "tag": [None, "   "],
    })

    result = summarize_movie_tags(tags)

    assert result.empty
    assert result.columns.tolist() == [
        "movieId",
        "tag",
        "tag_count",
    ]


def test_summarize_movie_tags_rejects_missing_columns():
    tags = pd.DataFrame({
        "movieId": [10],
        "label": ["war"],
    })

    with pytest.raises(ValueError, match="tag"):
        summarize_movie_tags(tags)
