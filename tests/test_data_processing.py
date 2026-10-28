
import pandas as pd
import pytest

from movieshelf.data_processing import (
    build_movie_statistics,
    build_movie_profiles,
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


def test_build_movie_profiles_combines_title_genres_and_tags():
    movies = pd.DataFrame({
        "movieId": [1, 2],
        "title": ["War Movie (2020)", "Space Movie (2021)"],
        "genres": ["Action|War", "Sci-Fi"],
    })
    movie_tags = pd.DataFrame({
        "movieId": [1, 1, 2],
        "tag": ["realistic", "military", "space"],
        "tag_count": [5, 3, 2],
    })

    result = build_movie_profiles(movies, movie_tags)

    war_movie = result.loc[result["movieId"] == 1].iloc[0]

    assert len(result) == 2
    assert "War Movie (2020)" in war_movie["profile_text"]
    assert "Action, War" in war_movie["profile_text"]
    assert "realistic" in war_movie["profile_text"]
    assert "military" in war_movie["profile_text"]


def test_build_movie_profiles_handles_movies_without_tags():
    movies = pd.DataFrame({
        "movieId": [1, 2],
        "title": ["War Movie (2020)", "Silent Movie (2021)"],
        "genres": ["Action|War", "Drama"],
    })
    movie_tags = pd.DataFrame({
        "movieId": [1],
        "tag": ["military"],
        "tag_count": [3],
    })

    result = build_movie_profiles(movies, movie_tags)

    silent_movie = result.loc[result["movieId"] == 2].iloc[0]

    assert len(result) == 2
    assert "Silent Movie (2021)" in silent_movie["profile_text"]
    assert "Drama" in silent_movie["profile_text"]
    assert "military" not in silent_movie["profile_text"]


def test_build_movie_profiles_limits_tags_to_100():
    movies = pd.DataFrame({
        "movieId": [1],
        "title": ["War Movie (2020)"],
        "genres": ["Action|War"],
    })
    movie_tags = pd.DataFrame({
        "movieId": [1] * 32,
        "tag": [f"tag_{i:02d}" for i in range(32)],
        "tag_count": list(range(1, 33)),
    })

    result = build_movie_profiles(movies, movie_tags)
    profile = result.iloc[0]["profile_text"]

    assert "tag_31" in profile
    assert "tag_30" in profile
    assert "tag_01" in profile
    assert "tag_00" in profile



def test_build_movie_profiles_rejects_missing_movie_columns():
    movies = pd.DataFrame({
        "movieId": [1],
        "title": ["War Movie (2020)"],
    })
    movie_tags = pd.DataFrame({
        "movieId": [1],
        "tag": ["military"],
        "tag_count": [3],
    })

    with pytest.raises(ValueError, match="genres"):
        build_movie_profiles(movies, movie_tags)


def test_build_movie_profiles_rejects_missing_tag_columns():
    movies = pd.DataFrame({
        "movieId": [1],
        "title": ["War Movie (2020)"],
        "genres": ["Action|War"],
    })
    movie_tags = pd.DataFrame({
        "movieId": [1],
        "tag": ["military"],
    })

    with pytest.raises(ValueError, match="tag_count"):
        build_movie_profiles(movies, movie_tags)


def test_build_movie_profiles_rejects_nonpositive_max_tags():
    movies = pd.DataFrame({
        "movieId": [1],
        "title": ["War Movie (2020)"],
        "genres": ["Action|War"],
    })
    movie_tags = pd.DataFrame({
        "movieId": [1],
        "tag": ["military"],
        "tag_count": [3],
    })

    with pytest.raises(ValueError, match="max_tags must be positive"):
        build_movie_profiles(movies, movie_tags, max_tags=0)



def test_repair_mojibake_fixes_single_encoding_error():
    from movieshelf.data_processing import repair_mojibake

    result = repair_mojibake("TÃ©a Leoni")

    assert result == "Téa Leoni"


def test_repair_mojibake_fixes_repeated_encoding_errors():
    from movieshelf.data_processing import repair_mojibake

    result = repair_mojibake("comedinha de velhinhos engraÃƒÂ§ada")

    assert result == "comedinha de velhinhos engraçada"


def test_repair_mojibake_preserves_correct_text():
    from movieshelf.data_processing import repair_mojibake

    result = repair_mojibake("Téa Leoni")

    assert result == "Téa Leoni"


def test_repair_mojibake_preserves_text_when_repair_fails():
    from movieshelf.data_processing import repair_mojibake

    text = "Gael GarcÃa Bernal"

    result = repair_mojibake(text)

    assert result == text



def test_normalize_tags_repairs_mojibake_before_lowercasing():
    from movieshelf.data_processing import normalize_tags

    tags = pd.Series(["TÃ©a Leoni", "action"])

    result = normalize_tags(tags)

    assert result.tolist() == ["téa leoni", "action"]



def test_build_movie_profiles_preserves_rare_relevant_tags():
    import pandas as pd

    from movieshelf.data_processing import build_movie_profiles

    movies = pd.DataFrame(
        {
            "movieId": [1],
            "title": ["Hurt Locker, The (2008)"],
            "genres": ["Action|Drama|War"],
        }
    )

    movie_tags = pd.DataFrame(
        {
            "movieId": [1] * 4,
            "tag": [
                "war",
                "realistic",
                "iraq war",
                "bomb defusal",
            ],
            "tag_count": [100, 80, 60, 1],
        }
    )

    profiles = build_movie_profiles(
        movies,
        movie_tags,
        max_tags=4,
    )

    profile = profiles.loc[0, "profile_text"]

    assert "war" in profile
    assert "realistic" in profile
    assert "iraq war" in profile
    assert "bomb defusal" in profile
