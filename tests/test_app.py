import io
import zipfile

import pandas as pd

from movieshelf.app import (
    _build_search_profiles,
    build_recommendation_filters,
    load_letterboxd_upload,
)


def test_recommendation_filters_preserve_user_choices_and_omit_defaults():
    filters = build_recommendation_filters(
        include_genres=["Drama", "Mystery"],
        genre_match="all",
        exclude_genres=["Horror"],
        minimum_rating_count=500,
        minimum_rating=3.5,
        release_years=(1990, 2020),
    )

    assert filters == {
        "release_year_from": 1990,
        "release_year_to": 2020,
        "include_genres": "Drama|Mystery",
        "genre_match": "all",
        "exclude_genres": "Horror",
        "minimum_rating_count": 500,
        "minimum_rating": 3.5,
    }


def test_recommendation_filters_leave_optional_filters_disabled_by_default():
    assert build_recommendation_filters(
        include_genres=[],
        genre_match="any",
        exclude_genres=[],
        minimum_rating_count=0,
        minimum_rating=0,
        release_years=(1900, 2026),
    ) == {
        "release_year_from": 1900,
        "release_year_to": 2026,
    }


def test_search_profiles_keep_movies_without_imported_plot():
    class PartialPlotStore:
        def add_plots_to_profiles(self, profiles):
            return profiles.assign(
                profile_text=profiles["profile_text"] + " enriched"
            )

        def movie_ids_with_status(self, status):
            assert status == "available"
            return {1}

    profiles = pd.DataFrame({
        "movieId": [1, 2, 3],
        "profile_text": ["one", "two", "three"],
    })

    result = _build_search_profiles(profiles, PartialPlotStore())

    assert result["movieId"].tolist() == [1, 2, 3]
    assert result["profile_text"].tolist() == [
        "one enriched",
        "two enriched",
        "three enriched",
    ]


def test_letterboxd_zip_upload_imports_without_persisting_upload_file(
    tmp_path,
    monkeypatch,
):
    import movieshelf.app as app

    monkeypatch.setattr(app.tempfile, "tempdir", str(tmp_path))
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as letterboxd_export:
        letterboxd_export.writestr(
            "watched.csv",
            "Date,Name,Year,Letterboxd URI\n"
            "2024-01-01,Example Movie,2020,https://letterboxd.com/film/example/\n",
        )
        letterboxd_export.writestr(
            "ratings.csv",
            "Date,Name,Year,Letterboxd URI,Rating\n"
            "2024-01-01,Example Movie,2020,https://letterboxd.com/film/example/,5\n",
        )

    catalog = pd.DataFrame({
        "movieId": [42],
        "title": ["Example Movie (2020)"],
        "genres": ["Drama"],
    })

    result = load_letterboxd_upload(archive.getvalue(), catalog)

    assert result.imported_entries == 1
    assert result.matched_entries == 1
    assert result.taste_context.watched_movie_ids == [42]
    assert list(tmp_path.iterdir()) == []
