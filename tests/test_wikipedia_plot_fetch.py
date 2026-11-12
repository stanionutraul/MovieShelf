from movieshelf.tools.wikipedia_synopsis_tool import WikipediaSynopsisTool


def _article_response(
    *,
    wikidata_item="Q123",
    extract=(
        "Intro.\n\n== Plot ==\n\n"
        "A character makes a difficult choice.\n\n"
        "== Cast ==\n\nActor details."
    ),
):
    return {
        "query": {
            "pages": {
                "10": {
                    "title": "Example film",
                    "fullurl": "https://en.wikipedia.org/wiki/Example_film",
                    "lastrevid": 456,
                    "pageprops": {"wikibase_item": wikidata_item},
                    "extract": extract,
                },
            },
        },
    }


def _license_response(url=(
    "https://creativecommons.org/licenses/by-sa/4.0/deed.en"
)):
    return {
        "license": {
            "title": "Creative Commons Attribution-Share Alike 4.0",
            "url": url,
        },
    }


def test_fetch_plot_section_verifies_identity_and_preserves_attribution(
    monkeypatch,
):
    responses = [_article_response(), _license_response()]
    tool = WikipediaSynopsisTool()
    monkeypatch.setattr(
        tool,
        "_request_json",
        lambda *args, **kwargs: responses.pop(0),
    )

    plot = tool.fetch_plot_section("Q123", "Example_film")

    assert plot == {
        "plot_summary": "A character makes a difficult choice.",
        "narrative_source": {
            "title": "Example film",
            "url": "https://en.wikipedia.org/wiki/Example_film",
            "revision": 456,
            "revision_url": (
                "https://en.wikipedia.org/wiki/Example_film?oldid=456"
            ),
            "license_title": (
                "Creative Commons Attribution-Share Alike 4.0"
            ),
            "license_url": (
                "https://creativecommons.org/licenses/by-sa/4.0/deed.en"
            ),
        },
    }


def test_fetch_plot_section_rejects_mismatched_page_identity(monkeypatch):
    tool = WikipediaSynopsisTool()
    monkeypatch.setattr(
        tool,
        "_request_json",
        lambda *args, **kwargs: _article_response(
            wikidata_item="Q999"
        ),
    )

    assert tool.fetch_plot_section("Q123", "Example_film") is None


def test_fetch_plot_section_rejects_missing_plot_or_unlicensed_page(
    monkeypatch,
):
    tool = WikipediaSynopsisTool()
    responses = [
        _article_response(extract="== Overview ==\n\nNo plot section."),
        _article_response(),
        _license_response("https://example.com/license"),
    ]
    monkeypatch.setattr(
        tool,
        "_request_json",
        lambda *args, **kwargs: responses.pop(0),
    )

    assert tool.fetch_plot_section("Q123", "Example_film") is None
    assert tool.fetch_plot_section("Q123", "Example_film") is None


def test_wikidata_article_mapping_decodes_percent_encoded_page_titles(
    monkeypatch,
):
    tool = WikipediaSynopsisTool()
    monkeypatch.setattr(
        tool,
        "_request_json",
        lambda *args, **kwargs: {
            "results": {
                "bindings": [
                    {
                        "imdbId": {"value": "tt0000001"},
                        "item": {
                            "value": "http://www.wikidata.org/entity/Q123",
                        },
                        "article": {
                            "value": (
                                "https://en.wikipedia.org/wiki/"
                                "Weekend_at_Bernie%27s"
                            ),
                        },
                    },
                ],
            },
        },
    )

    articles = tool.find_wikipedia_articles({"tt0000001"})

    assert articles == {"tt0000001": ("Q123", "Weekend_at_Bernie's")}
