from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from movieshelf.semantic_search import SemanticMovieSearchEngine


def _engine():
    engine = object.__new__(SemanticMovieSearchEngine)
    engine.profiles = pd.DataFrame({
        "movieId": [1, 2, 3],
        "profile_text": ["First", "Second", "Third"],
        "genres": ["Fantasy|Animation", "Fantasy", "Drama"],
    })
    engine.profile_embeddings = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    engine.model = SimpleNamespace(encode=lambda *args, **kwargs: np.array(
        [[1.0, 0.0]]
    ))
    engine.has_genre_data = True
    engine.has_rating_data = False
    return engine


def test_semantic_search_defaults_to_any_included_genre():
    results = _engine().search(
        "animated fantasy",
        include_genres=["Fantasy", "Animation"],
    )

    assert results["movieId"].tolist() == [1, 2]


def test_semantic_search_can_require_all_included_genres():
    results = _engine().search(
        "animated fantasy",
        include_genres=["Fantasy", "Animation"],
        genre_match="all",
    )

    assert results["movieId"].tolist() == [1]


@pytest.mark.parametrize("genre_match", [None, [], "either"])
def test_semantic_search_rejects_invalid_genre_match(genre_match):
    with pytest.raises(ValueError, match="genre_match"):
        _engine().search("animated fantasy", genre_match=genre_match)
