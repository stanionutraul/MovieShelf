
import pandas as pd
import pytest

from movieshelf.ranking import calculate_weighted_rating


def test_weighted_rating_moves_small_samples_toward_global_mean():
    movies = pd.DataFrame(
        {
            "movieId": [1, 2],
            "average_rating": [5.0, 4.5],
            "rating_count": [2, 10_000],
        }
    )

    result = calculate_weighted_rating(
        movies,
        global_mean=3.5,
        minimum_votes=1_000,
    )

    small_sample_score = result.loc[
        result["movieId"] == 1, "weighted_rating"
    ].iloc[0]

    large_sample_score = result.loc[
        result["movieId"] == 2, "weighted_rating"
    ].iloc[0]

    assert small_sample_score < 5.0
    assert small_sample_score > 3.5
    assert large_sample_score > small_sample_score


def test_weighted_rating_leaves_unrated_movies_without_score():
    movies = pd.DataFrame(
        {
            "movieId": [1],
            "average_rating": [float("nan")],
            "rating_count": [0],
        }
    )

    result = calculate_weighted_rating(
        movies,
        global_mean=3.5,
        minimum_votes=1_000,
    )

    assert pd.isna(result.loc[0, "weighted_rating"])


def test_weighted_rating_rejects_invalid_minimum_votes():
    movies = pd.DataFrame(
        {
            "movieId": [1],
            "average_rating": [4.0],
            "rating_count": [10],
        }
    )

    with pytest.raises(ValueError, match="minimum_votes must be positive"):
        calculate_weighted_rating(
            movies,
            global_mean=3.5,
            minimum_votes=0,
        )
