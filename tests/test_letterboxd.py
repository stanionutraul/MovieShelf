import zipfile

import pandas as pd
import pytest

from movieshelf.letterboxd import load_letterboxd_export


@pytest.fixture
def catalog():
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4],
            "title": [
                "Amelie (2001)",
                "Portrait of a Lady on Fire (2019)",
                "Cats (2019)",
                "Cats (1998)",
            ],
            "genres": [
                "Drama|Romance",
                "Drama|Romance",
                "Musical|Comedy",
                "Musical|Comedy",
            ],
        }
    )


def _write_export(path, files):
    with zipfile.ZipFile(path, "w") as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)
    return path


def test_imports_ratings_watched_and_review_examples(tmp_path, catalog):
    source = _write_export(
        tmp_path / "export.zip",
        {
            "watched.csv": (
                "Date,Name,Year,Letterboxd URI\n"
                ",Amélie,2001,https://letterboxd.com/film/amelie/\n"
                ",Portrait of a Lady on Fire,2019,"
                "https://letterboxd.com/film/portrait/\n"
                ",Unknown Movie,2024,https://letterboxd.com/film/unknown/\n"
                ",Cats,1998,https://letterboxd.com/film/cats-1998/\n"
            ),
            "ratings.csv": (
                "Date,Name,Year,Letterboxd URI,Rating\n"
                "2024-01-01,Amélie,2001,"
                "https://letterboxd.com/film/amelie/,5\n"
                "2024-01-02,Portrait of a Lady on Fire,2019,"
                "https://letterboxd.com/film/portrait/,4.5\n"
                "2024-01-03,Cats,2019,"
                "https://letterboxd.com/film/cats/,1\n"
            ),
            "reviews.csv": (
                "Date,Name,Year,Letterboxd URI,Rating,Review\n"
                "2024-01-01,Amélie,2001,"
                "https://letterboxd.com/film/amelie/,5,"
                "\"Tender, playful and beautifully observed.\"\n"
                "2024-01-03,Cats,2019,"
                "https://letterboxd.com/film/cats/,1,"
                "\"Overlong and visually unpleasant.\"\n"
                "2024-01-04,Cats,1998,"
                "https://letterboxd.com/film/cats-1998/,,"
                "\"A strange, entertaining adaptation.\"\n"
            ),
        },
    )

    result = load_letterboxd_export(source, catalog)

    assert result.imported_entries == 5
    assert result.matched_entries == 4
    assert result.unmatched_titles == ["Unknown Movie (2024)"]
    assert result.taste_context.watched_movie_ids == [1, 2, 3, 4]
    assert result.taste_context.liked_movie_ids == [1, 2]
    assert result.taste_context.disliked_movie_ids == [3]
    assert result.taste_context.preferred_genres == ["Drama", "Romance"]
    genre_summary = {
        item.genre: item
        for item in result.taste_context.genre_rating_summary
    }
    assert genre_summary["Drama"].average_rating == 4.75
    assert genre_summary["Drama"].rated_movie_count == 2
    assert [
        example.title
        for example in result.taste_context.rated_movie_examples
    ] == ["Amélie", "Portrait of a Lady on Fire", "Cats"]
    assert result.taste_context.rated_movie_examples[0].genres == [
        "Drama",
        "Romance",
    ]
    assert [example.title for example in result.taste_context.review_examples] == [
        "Amélie",
        "Cats",
        "Cats",
    ]
    assert "Tender, playful" in result.taste_context.review_examples[0].review
    assert result.taste_context.review_examples[2].rating is None


def test_imported_folder_and_ambiguous_title_are_supported(
    tmp_path,
    catalog,
):
    export_dir = tmp_path / "letterboxd"
    export_dir.mkdir()
    (export_dir / "watched.csv").write_text(
        "Date,Name,Year,Letterboxd URI\n"
        ",Cats,,https://letterboxd.com/film/cats/\n",
        encoding="utf-8",
    )

    result = load_letterboxd_export(export_dir, catalog)

    assert result.matched_entries == 0
    assert result.taste_context.watched_movie_ids == []
    assert result.unmatched_titles == ["Cats"]


def test_catalog_alternate_titles_are_matched_without_ignoring_year(
    tmp_path,
):
    catalog = pd.DataFrame(
        {
            "movieId": [10, 11, 12, 13],
            "title": [
                "Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001)",
                "Bicycle Thieves (a.k.a. The Bicycle Thief) (1948)",
                "Seven (a.k.a. Se7en) (1995)",
                "300 (2007)",
            ],
        }
    )
    source = _write_export(
        tmp_path / "alternate-titles.zip",
        {
            "watched.csv": (
                "Date,Name,Year,Letterboxd URI\n"
                ",Amélie,2001,https://letterboxd.com/film/amelie/\n"
                ",The Bicycle Thief,1948,"
                "https://letterboxd.com/film/bicycle-thieves/\n"
                ",Se7en,1995,https://letterboxd.com/film/se7en/\n"
                ",300,2006,https://letterboxd.com/film/300/\n"
            ),
        },
    )

    result = load_letterboxd_export(source, catalog)

    assert result.taste_context.watched_movie_ids == [10, 11, 12]
    assert result.unmatched_titles == ["300 (2006)"]


def test_catalog_title_without_year_can_match_unique_export_title(
    tmp_path,
):
    catalog = pd.DataFrame({
        "movieId": [20],
        "title": ["Nocturnal Animals"],
    })
    source = _write_export(
        tmp_path / "unknown-catalog-year.zip",
        {
            "watched.csv": (
                "Date,Name,Year,Letterboxd URI\n"
                ",Nocturnal Animals,2016,"
                "https://letterboxd.com/film/nocturnal-animals/\n"
            ),
        },
    )

    result = load_letterboxd_export(source, catalog)

    assert result.taste_context.watched_movie_ids == [20]
    assert result.unmatched_titles == []


def test_synthetic_exports_produce_distinct_taste_profiles(
    tmp_path,
    catalog,
):
    drama_export = _write_export(
        tmp_path / "drama-profile.zip",
        {
            "ratings.csv": (
                "Date,Name,Year,Letterboxd URI,Rating\n"
                "2024-01-01,Amelie,2001,"
                "https://letterboxd.com/film/amelie/,5\n"
                "2024-01-02,Portrait of a Lady on Fire,2019,"
                "https://letterboxd.com/film/portrait/,4.5\n"
            ),
        },
    )
    musical_export = _write_export(
        tmp_path / "musical-profile.zip",
        {
            "ratings.csv": (
                "Date,Name,Year,Letterboxd URI,Rating\n"
                "2024-01-01,Cats,2019,"
                "https://letterboxd.com/film/cats-2019/,5\n"
                "2024-01-02,Cats,1998,"
                "https://letterboxd.com/film/cats-1998/,4.5\n"
            ),
        },
    )

    drama_profile = load_letterboxd_export(
        drama_export,
        catalog,
    ).taste_context
    musical_profile = load_letterboxd_export(
        musical_export,
        catalog,
    ).taste_context

    assert drama_profile.preferred_genres == ["Drama", "Romance"]
    assert musical_profile.preferred_genres == ["Comedy", "Musical"]
    assert {
        item.genre
        for item in drama_profile.genre_rating_summary
    } == {"Drama", "Romance"}
    assert {
        item.genre
        for item in musical_profile.genre_rating_summary
    } == {"Comedy", "Musical"}
    assert drama_profile.rated_movie_examples[0].title == "Amelie"
    assert musical_profile.rated_movie_examples[0].title == "Cats"
    assert set(drama_profile.watched_movie_ids).isdisjoint(
        musical_profile.watched_movie_ids
    )


def test_import_builds_tag_preferences_from_all_rated_movies(
    tmp_path,
    catalog,
):
    source = _write_export(
        tmp_path / "tag-profile.zip",
        {
            "ratings.csv": (
                "Date,Name,Year,Letterboxd URI,Rating\n"
                "2024-01-01,Amelie,2001,"
                "https://letterboxd.com/film/amelie/,5\n"
                "2024-01-02,Portrait of a Lady on Fire,2019,"
                "https://letterboxd.com/film/portrait/,4\n"
                "2024-01-03,Cats,2019,"
                "https://letterboxd.com/film/cats/,1\n"
            ),
        },
    )
    movie_tags = pd.DataFrame([
        {"movieId": 1, "tag": "warm", "tag_count": 2},
        {"movieId": 2, "tag": "warm", "tag_count": 1},
        {"movieId": 3, "tag": "warm", "tag_count": 1},
        {"movieId": 1, "tag": "romance", "tag_count": 1},
    ])

    result = load_letterboxd_export(source, catalog, movie_tags)

    tag_preferences = {
        item.tag: item
        for item in result.taste_context.tag_rating_summary
    }
    assert tag_preferences["warm"].average_rating == 3.3333333333333335
    assert tag_preferences["warm"].rated_movie_count == 3
    assert "romance" not in tag_preferences


def test_invalid_rating_is_reported_with_file_and_row(tmp_path, catalog):
    source = _write_export(
        tmp_path / "invalid.zip",
        {
            "ratings.csv": (
                "Date,Name,Year,Letterboxd URI,Rating\n"
                "2024-01-01,Amélie,2001,"
                "https://letterboxd.com/film/amelie/,6\n"
            ),
        },
    )

    with pytest.raises(ValueError, match=r"ratings.csv, row 2"):
        load_letterboxd_export(source, catalog)


def test_non_export_input_is_rejected(tmp_path, catalog):
    source = tmp_path / "not-an-export.txt"
    source.write_text("not a Letterboxd export", encoding="utf-8")

    with pytest.raises(ValueError, match="ZIP file or an extracted directory"):
        load_letterboxd_export(source, catalog)
