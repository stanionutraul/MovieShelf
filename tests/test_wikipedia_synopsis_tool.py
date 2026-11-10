import json
from io import BytesIO
from urllib.error import HTTPError

import pytest

from movieshelf.tools.wikipedia_synopsis_tool import (
    WikipediaSynopsisError,
    WikipediaSynopsisTool,
)


def _wikidata_response():
    return {
        "results": {
            "bindings": [
                {
                    "imdbId": {
                        "value": "tt0000001",
                    },
                    "item": {
                        "value": "http://www.wikidata.org/entity/Q123",
                    },
                    "article": {
                        "value": (
                            "https://en.wikipedia.org/wiki/Example_film"
                        ),
                    },
                },
            ],
        },
    }


def _summary_response(
    *,
    wikidata_item="Q123",
    page_title="Example film",
    extract="A film about starting over.",
):
    return {
        "type": "standard",
        "title": page_title,
        "wikibase_item": wikidata_item,
        "revision": 456,
        "extract": extract,
        "content_urls": {
            "desktop": {
                "page": "https://en.wikipedia.org/wiki/Example_film",
            },
        },
    }


def _page_response(license_url=None):
    return {
        "license": {
            "title": "Creative Commons Attribution-Share Alike 4.0",
            "url": license_url or (
                "https://creativecommons.org/licenses/by-sa/4.0/deed.en"
            ),
        },
    }


def _fake_urlopen(responses, requests):
    def open_request(request, timeout, context=None):
        requests.append((request, timeout))
        response_data = responses.pop(0)
        if isinstance(response_data, Exception):
            raise response_data
        return BytesIO(json.dumps(response_data).encode("utf-8"))

    return open_request


def test_enrich_candidates_resolves_imdb_id_and_checks_page_license(
    monkeypatch,
):
    requests = []
    responses = [
        _wikidata_response(),
        _summary_response(),
        _page_response(),
    ]
    monkeypatch.setattr(
        "movieshelf.tools.wikipedia_synopsis_tool.urlopen",
        _fake_urlopen(responses, requests),
    )
    tool = WikipediaSynopsisTool()

    enriched = tool.enrich_candidates([
        {"movieId": 10, "imdbId": "0000001"},
    ])

    assert enriched[0]["plot_summary"] == "A film about starting over."
    assert enriched[0]["narrative_source"] == {
        "title": "Example film",
        "url": "https://en.wikipedia.org/wiki/Example_film",
        "revision": 456,
        "revision_url": (
            "https://en.wikipedia.org/wiki/Example_film?oldid=456"
        ),
        "license_title": "Creative Commons Attribution-Share Alike 4.0",
        "license_url": (
            "https://creativecommons.org/licenses/by-sa/4.0/deed.en"
        ),
    }
    assert all(
        request.get_header("User-agent") == tool.USER_AGENT
        for request, _ in requests
    )
    assert len(requests) == 3


def test_candidates_without_imdb_ids_do_not_call_external_apis(monkeypatch):
    def unexpected_request(*args, **kwargs):
        raise AssertionError("Unexpected Wikimedia request.")

    monkeypatch.setattr(
        "movieshelf.tools.wikipedia_synopsis_tool.urlopen",
        unexpected_request,
    )

    candidates = [{"movieId": 10, "title": "Unknown (2020)"}]
    assert WikipediaSynopsisTool().enrich_candidates(candidates) == candidates


@pytest.mark.parametrize(
    ("summary", "page", "expected_summary"),
    [
        (
            _summary_response(wikidata_item="Q999"),
            _page_response(),
            None,
        ),
        (
            _summary_response(),
            _page_response("https://example.com/license"),
            None,
        ),
        (
            {
                **_summary_response(),
                "type": "disambiguation",
            },
            _page_response(),
            None,
        ),
    ],
)
def test_unverified_or_incompatible_pages_are_not_attached(
    monkeypatch,
    summary,
    page,
    expected_summary,
):
    responses = [_wikidata_response(), summary, page]
    monkeypatch.setattr(
        "movieshelf.tools.wikipedia_synopsis_tool.urlopen",
        _fake_urlopen(responses, []),
    )
    enriched = WikipediaSynopsisTool().enrich_candidates([
        {"movieId": 10, "imdbId": "tt0000001"},
    ])

    assert enriched[0].get("plot_summary") is expected_summary
    assert "narrative_source" not in enriched[0]


def test_http_rate_limit_is_reported_with_retry_after(monkeypatch):
    error = HTTPError(
        "https://query.wikidata.org/sparql",
        429,
        "Too Many Requests",
        {"Retry-After": "30"},
        None,
    )
    monkeypatch.setattr(
        "movieshelf.tools.wikipedia_synopsis_tool.urlopen",
        _fake_urlopen([error], []),
    )

    with pytest.raises(WikipediaSynopsisError, match="Retry after 30"):
        WikipediaSynopsisTool().enrich_candidates([
            {"movieId": 10, "imdbId": "tt0000001"},
        ])


def test_enriched_synopsis_is_cached_in_memory(monkeypatch):
    requests = []
    monkeypatch.setattr(
        "movieshelf.tools.wikipedia_synopsis_tool.urlopen",
        _fake_urlopen(
            [
                _wikidata_response(),
                _summary_response(),
                _page_response(),
            ],
            requests,
        ),
    )
    tool = WikipediaSynopsisTool()
    candidate = {"movieId": 10, "imdbId": "tt0000001"}

    tool.enrich_candidates([candidate])
    tool.enrich_candidates([candidate])

    assert len(requests) == 3


def test_numeric_string_revision_from_wikipedia_is_normalized(monkeypatch):
    responses = [
        _wikidata_response(),
        _summary_response() | {"revision": "456"},
        _page_response(),
    ]
    monkeypatch.setattr(
        "movieshelf.tools.wikipedia_synopsis_tool.urlopen",
        _fake_urlopen(responses, []),
    )

    enriched = WikipediaSynopsisTool().enrich_candidates([
        {"movieId": 10, "imdbId": "tt0000001"},
    ])

    source = enriched[0]["narrative_source"]
    assert source["revision"] == 456
    assert source["revision_url"].endswith("?oldid=456")
