import pandas as pd
import pytest

from movieshelf.tools.wikipedia_plot_index import (
    WikipediaPlotImporter,
    WikipediaPlotStore,
)


class FakeWikipedia:
    def __init__(self):
        self.mapping_calls = []
        self.fetch_calls = []

    @staticmethod
    def normalize_imdb_id(value):
        if not value or str(value) == "0":
            return None
        digits = str(value).removeprefix("tt")
        return "tt" + digits.zfill(7)

    def find_wikipedia_articles(self, imdb_ids):
        self.mapping_calls.append(set(imdb_ids))
        return {
            "tt0000001": ("Q1", "First_film"),
        }

    def fetch_plot_section(self, wikidata_item, page_title):
        self.fetch_calls.append((wikidata_item, page_title))
        return {
            "plot_summary": "A verified plot section.",
            "narrative_source": {
                "title": "First film",
                "url": "https://en.wikipedia.org/wiki/First_film",
                "revision": 123,
                "license_title": "CC BY-SA 4.0",
                "license_url": (
                    "https://creativecommons.org/licenses/by-sa/4.0/"
                ),
            },
        }


def test_plot_store_persists_attribution_and_enriches_local_profiles(
    tmp_path,
):
    store = WikipediaPlotStore(tmp_path / "plots.sqlite")
    store.save(
        1,
        "tt0000001",
        "available",
        plot={
            "plot_summary": "A verified plot section.",
            "narrative_source": {
                "title": "First film",
                "url": "https://en.wikipedia.org/wiki/First_film",
                "revision": 123,
                "license_title": "CC BY-SA 4.0",
                "license_url": (
                    "https://creativecommons.org/licenses/by-sa/4.0/"
                ),
            },
        },
    )

    profiles = store.add_plots_to_profiles(pd.DataFrame({
        "movieId": [1, 2],
        "profile_text": [
            "First. Genres: Comedy. Descriptive tags: funny, warm.",
            "Second. Genres: Drama. Descriptive tags: dramatic.",
        ],
    }))
    candidates = store.enrich_candidates([
        {"movieId": 1, "title": "First (1980)"},
        {"movieId": 2, "title": "Second (1981)"},
    ])

    assert profiles.iloc[0]["profile_text"] == (
        "First. Genres: Comedy. Plot: A verified plot section. "
        "Descriptive tags: funny, warm."
    )
    assert "Plot:" not in profiles.iloc[1]["profile_text"]
    assert candidates[0]["plot_summary"] == "A verified plot section."
    assert candidates[0]["narrative_source"]["revision_url"] == (
        "https://en.wikipedia.org/wiki/First_film?oldid=123"
    )
    assert "plot_summary" not in candidates[1]
    assert store.coverage() == {"available": 1}


def test_plot_profiles_keep_only_the_most_relevant_descriptive_tags(tmp_path):
    store = WikipediaPlotStore(tmp_path / "plots.sqlite")
    store.save(
        1,
        "tt0000001",
        "available",
        plot={
            "plot_summary": "A verified plot section.",
            "narrative_source": {
                "title": "First film",
                "url": "https://en.wikipedia.org/wiki/First_film",
                "revision": 123,
                "license_title": "CC BY-SA 4.0",
                "license_url": (
                    "https://creativecommons.org/licenses/by-sa/4.0/"
                ),
            },
        },
    )
    tags = ", ".join(f"tag-{index}" for index in range(20))

    profile = store.add_plots_to_profiles(pd.DataFrame({
        "movieId": [1],
        "profile_text": [
            f"First. Genres: Comedy. Descriptive tags: {tags}.",
        ],
    })).iloc[0]["profile_text"]

    assert "tag-0" in profile
    assert "tag-14" in profile
    assert "tag-15" not in profile
    assert profile.endswith(
        "Descriptive tags: "
        + ", ".join(f"tag-{index}" for index in range(15))
        + "."
    )


def test_plot_store_rejects_unattributed_available_text(tmp_path):
    store = WikipediaPlotStore(tmp_path / "plots.sqlite")

    with pytest.raises(ValueError, match="CC BY-SA 4.0 attribution"):
        store.save(
            1,
            "tt0000001",
            "available",
            plot={"plot_summary": "Unattributed plot."},
        )


def test_importer_is_resumable_and_records_unmatched_titles(tmp_path):
    store = WikipediaPlotStore(tmp_path / "plots.sqlite")
    wikipedia = FakeWikipedia()
    importer = WikipediaPlotImporter(
        store,
        wikipedia=wikipedia,
        request_delay=0,
    )
    movies = pd.DataFrame([
        {"movieId": 1, "imdbId": "1"},
        {"movieId": 2, "imdbId": "2"},
        {"movieId": 3, "imdbId": "0"},
    ])

    result = importer.import_movies(movies, batch_size=2)
    resumed = importer.import_movies(movies, batch_size=2)

    assert result == {
        "available": 1,
        "unavailable": 0,
        "no_article": 1,
        "unlinked": 1,
        "error": 0,
    }
    assert all(value == 0 for value in resumed.values())
    assert len(wikipedia.mapping_calls) == 1
    assert store.coverage() == {
        "available": 1,
        "no_article": 1,
        "unlinked": 1,
    }


def test_importer_can_retry_previously_unavailable_pages(tmp_path):
    store = WikipediaPlotStore(tmp_path / "plots.sqlite")
    store.save(1, "tt0000001", "unavailable")
    wikipedia = FakeWikipedia()
    importer = WikipediaPlotImporter(
        store,
        wikipedia=wikipedia,
        request_delay=0,
    )

    result = importer.import_movies(
        [
            {"movieId": 1, "imdbId": "1"},
            {"movieId": 2, "imdbId": "2"},
        ],
        retry_unavailable=True,
    )

    assert result["available"] == 1
    assert store.coverage() == {"available": 1}
