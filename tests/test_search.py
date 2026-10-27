
import pandas as pd
import pytest

from movieshelf.search import MovieSearchEngine


@pytest.fixture
def movie_profiles():
    """Small movie catalog used to test text search."""
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4],
            "profile_text": [
                (
                    "Saving Private Ryan. Genres: War, Drama. "
                    "Descriptive tags: World War II, soldiers, realistic war."
                ),
                (
                    "The Hurt Locker. Genres: Drama, Thriller. "
                    "Descriptive tags: Iraq War, bomb defusal, military, realistic action."
                ),
                (
                    "Finding Nemo. Genres: Animation, Adventure. "
                    "Descriptive tags: ocean, fish, family, underwater adventure."
                ),
                (
                    "Black Hawk Down. Genres: War, Action. "
                    "Descriptive tags: military, Somalia, soldiers, rescue mission."
                ),
            ],
        }
    )


def test_search_returns_most_relevant_movie_first(movie_profiles):
    engine = MovieSearchEngine(movie_profiles)

    results = engine.search("Iraq War bomb defusal", n=3)

    assert not results.empty
    assert results.iloc[0]["movieId"] == 2


def test_search_scores_are_descending(movie_profiles):
    engine = MovieSearchEngine(movie_profiles)

    results = engine.search("military soldiers", n=4)

    scores = results["similarity_score"].tolist()

    assert scores == sorted(scores, reverse=True)


def test_search_respects_result_limit(movie_profiles):
    engine = MovieSearchEngine(movie_profiles)

    results = engine.search("war military", n=2)

    assert len(results) <= 2


def test_search_excludes_movies_with_no_textual_match(movie_profiles):
    engine = MovieSearchEngine(movie_profiles)

    results = engine.search("spaceship on Mars", n=4)

    assert results.empty


def test_search_rejects_empty_query(movie_profiles):
    engine = MovieSearchEngine(movie_profiles)

    with pytest.raises(ValueError):
        engine.search("   ", n=5)


def test_search_rejects_non_positive_result_limit(movie_profiles):
    engine = MovieSearchEngine(movie_profiles)

    with pytest.raises(ValueError):
        engine.search("military", n=0)
