
import pandas as pd

from movieshelf.data_processing import build_movie_statistics


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
