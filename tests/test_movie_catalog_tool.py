import pandas as pd
import pytest

from movieshelf.tools.movie_catalog_tool import MovieCatalogTool


def _catalog_tool(tmp_path, links=None):
    catalog_path = tmp_path / "catalog.csv"
    tags_path = tmp_path / "tags.csv"
    pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "title": ["Both (2020)", "Fantasy (2021)", "Drama (2022)"],
            "genres": [
                "Fantasy|Animation",
                "Fantasy",
                "Drama",
            ],
            "rating_count": [100, 100, 100],
            "average_rating": [4.0, 4.0, 4.0],
            "weighted_rating": [4.0, 4.0, 4.0],
        }
    ).to_csv(catalog_path, index=False)
    pd.DataFrame(
        {
            "movieId": [1],
            "tag": ["family"],
            "tag_count": [1],
        }
    ).to_csv(tags_path, index=False)

    return MovieCatalogTool(catalog_path, tags_path, links=links)


def test_catalog_filter_defaults_to_any_included_genre(tmp_path):
    tool = _catalog_tool(tmp_path)

    results = tool.filter_movies(
        include_genres=["Fantasy", "Animation"],
    )

    assert [movie["movieId"] for movie in results] == [1, 2]


def test_catalog_filter_can_require_all_included_genres(tmp_path):
    tool = _catalog_tool(tmp_path)

    results = tool.filter_movies(
        include_genres=["Fantasy", "Animation"],
        genre_match="all",
    )

    assert [movie["movieId"] for movie in results] == [1]


def test_catalog_filter_rejects_unknown_genre_match_mode(tmp_path):
    tool = _catalog_tool(tmp_path)

    with pytest.raises(ValueError, match="genre_match"):
        tool.filter_movies(
            include_genres=["Fantasy"],
            genre_match="either",
        )


def test_catalog_merges_external_imdb_ids_by_movie_id(tmp_path):
    links = pd.DataFrame({
        "movieId": [1, 2, 3],
        "imdbId": pd.Series(["0012345", "tt0000002", pd.NA], dtype="string"),
    })
    tool = _catalog_tool(tmp_path, links=links)

    results = tool.search_movies("Both")

    assert results[0]["imdbId"] == "0012345"
