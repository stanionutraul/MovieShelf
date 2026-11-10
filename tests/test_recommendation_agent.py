import json
from types import SimpleNamespace

from movieshelf.agents.recommendation_agent import RecommendationAgent
from movieshelf.schemas.movie_request import MovieRequest
from movieshelf.schemas.recommendation_context import RecommendationContext
from movieshelf.schemas.user_taste import UserTasteContext


def _movie(
    movie_id,
    title=None,
    genres="Sci-Fi|Mystery",
    average_rating=4.0,
    rating_count=100,
):
    return {
        "movieId": movie_id,
        "title": title or f"Catalog Movie {movie_id} (2020)",
        "genres": genres,
        "average_rating": average_rating,
        "rating_count": rating_count,
        "weighted_rating": 5.0,
        "profile_text": "Must not be sent to Qwen.",
    }


class FakeCatalogTool:
    def __init__(self, query_results=None, details=None, search_results=None):
        self.query_results = query_results or {}
        self.details = details or {}
        self.search_results = search_results or []
        self.tag_queries = []
        self.tag_query_limits = []
        self.filter_calls = []
        self.search_calls = []

    def get_movies_by_tags(self, query, limit=20):
        self.tag_queries.append(query)
        self.tag_query_limits.append(limit)
        return self.query_results.get(query, [])

    def filter_movies(self, **kwargs):
        self.filter_calls.append(kwargs)
        return []

    def search_movies(self, **kwargs):
        self.search_calls.append(kwargs)
        return self.search_results

    def get_random_movies(self, limit=10):
        return []

    def get_movie_details(self, movie_id):
        return {
            **_movie(movie_id),
            "tags": self.details.get(movie_id, [
                {"tag": "secret agency", "tag_count": 5},
                {"tag": "unrelated user tag", "tag_count": 50},
            ]),
        }


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []
        self.tools = []
        self.structured_prompts = []

    def chat_with_tools(self, messages, tools):
        self.messages.append([dict(message) for message in messages])
        self.tools.append(tools)
        return next(self.responses)

    def generate_structured(self, prompt, schema):
        self.structured_prompts.append(prompt)
        response = next(self.responses)
        return schema.model_validate_json(
            response.message.content
        ).model_dump()


class FakeSynopsisTool:
    def __init__(self, synopses=None):
        self.synopses = synopses or {}
        self.calls = []

    def enrich_candidates(self, candidates):
        self.calls.append([candidate["movieId"] for candidate in candidates])
        enriched = []
        for candidate in candidates:
            synopsis = self.synopses.get(candidate["movieId"])
            enriched.append({
                **candidate,
                **synopsis,
            } if synopsis else dict(candidate))
        return enriched


def _response(content, tool_calls=None):
    return SimpleNamespace(
        message=SimpleNamespace(
            content=content,
            tool_calls=tool_calls or [],
        )
    )


def _context(**request_values):
    values = {
        "intent_summary": (
            "Movies about secret agencies and supernatural anomalies"
        ),
        "required_criteria": [
            "secret agency",
            "supernatural anomalies",
        ],
        "genres_include": ["science fiction"],
        "genres_exclude": ["horror"],
    }
    values.update(request_values)
    return RecommendationContext(movie_request=MovieRequest(**values))


def _final_response(movie_id=10, uncertainty="The plot is not verified."):
    return _response(json.dumps({
        "recommendations": [
            {
                "movieId": movie_id,
                "description": (
                    "A former prisoner tries to rebuild his life after "
                    "release. The story matches your request about "
                    "reintegration."
                ),
                "reason": "The catalog tags suggest a possible match.",
                "uncertainty": uncertainty,
            }
        ]
    }))


def test_narrative_request_continues_without_a_qwen_tool_call():
    candidate = _movie(10)
    catalog = FakeCatalogTool({
        "secret agency": [candidate],
        "supernatural anomalies": [],
    })
    llm = FakeLLM([
        _response("I should call get_movies_by_tags, but did not."),
        _final_response(),
    ])
    agent = RecommendationAgent(catalog_tool=catalog, llm=llm)

    result = agent.run(_context())

    assert "Catalog Movie 10" in result
    assert len(llm.messages) == 1
    assert len(llm.structured_prompts) == 1
    assert len(catalog.tag_queries) == 3
    assert [tool["function"]["name"] for tool in llm.tools[0]] == [
        "search_movies",
        "filter_movies",
        "get_random_movies",
        "get_movies_by_tags",
        "get_movie_details",
    ]


def test_full_initial_candidate_pool_skips_unneeded_tool_chat():
    catalog = FakeCatalogTool({
        "secret agency": [_movie(movie_id) for movie_id in range(1, 9)],
    })
    llm = FakeLLM([_final_response(1)])
    agent = RecommendationAgent(catalog_tool=catalog, llm=llm)

    result = agent.run(_context())

    assert "Catalog Movie 1" in result
    assert llm.messages == []
    assert len(llm.structured_prompts) == 1


def test_payload_contains_only_relevant_limited_movielens_tags():
    candidate = _movie(10)
    catalog = FakeCatalogTool(
        {"secret agency": [candidate], "supernatural anomalies": []},
        details={
            10: [
                {
                    "tag": f"secret agency clue {count}",
                    "tag_count": count,
                }
                for count in range(1, 7)
            ] + [{"tag": "unrelated user tag", "tag_count": 50}]
        },
    )
    agent = RecommendationAgent(catalog_tool=catalog, llm=FakeLLM([]))

    candidates = agent._initial_retrieval(_context())
    payload = json.loads(agent._build_user_message(_context(), candidates))
    payload_candidate = payload["initial_catalog_candidates"][0]

    assert len(payload_candidate["tags"]) == agent.MAX_TAGS_PER_CANDIDATE
    assert [tag["tag_count"] for tag in payload_candidate["tags"]] == [
        6, 5, 4, 3, 2,
    ]
    assert all(
        "unrelated" not in tag["tag"]
        for tag in payload_candidate["tags"]
    )
    assert "profile_text" not in payload_candidate
    assert "weighted_rating" not in payload_candidate
    assert "not verified" in payload["catalog_evidence_note"]


def test_taste_profile_is_included_as_a_personalization_signal():
    taste = UserTasteContext(
        preferred_genres=["Drama"],
        less_preferred_genres=["Horror"],
        taste_summary="Genres rated highly: Drama.",
        review_examples=[
            {
                "title": "Example Film",
                "rating": 5,
                "review": "Quiet, thoughtful and emotionally grounded.",
            },
        ],
        rated_movie_examples=[
            {
                "title": "Another Film",
                "rating": 5,
                "genres": ["Drama", "Mystery"],
            },
        ],
    )
    context = RecommendationContext(
        movie_request=MovieRequest(intent_summary="Find a thoughtful film."),
        taste_context=taste,
    )
    agent = RecommendationAgent(
        catalog_tool=FakeCatalogTool(),
        llm=FakeLLM([]),
    )

    payload = json.loads(agent._build_user_message(context, []))

    assert payload["recommendation_context"]["taste_context"][
        "preferred_genres"
    ] == ["Drama"]
    assert payload["recommendation_context"]["taste_context"][
        "review_examples"
    ][0]["review"] == "Quiet, thoughtful and emotionally grounded."
    assert payload["recommendation_context"]["taste_context"][
        "rated_movie_examples"
    ][0] == {
        "title": "Another Film",
        "rating": 5.0,
        "genres": ["Drama", "Mystery"],
    }
    assert "personalize" in payload["instructions"]


def test_taste_only_request_retrieves_movies_from_preferred_genres():
    class TasteCatalog(FakeCatalogTool):
        def filter_movies(self, **kwargs):
            self.filter_calls.append(kwargs)
            return [_movie(20, genres="Drama|Romance")]

    catalog = TasteCatalog()
    agent = RecommendationAgent(
        catalog_tool=catalog,
        llm=FakeLLM([]),
    )
    context = RecommendationContext(
        taste_context=UserTasteContext(
            watched_movie_ids=[10],
            preferred_genres=["Drama", "Romance"],
            less_preferred_genres=["Horror"],
        ),
    )

    candidates = agent._initial_retrieval(context)

    assert [candidate["movieId"] for candidate in candidates] == [20]
    assert catalog.filter_calls[0] == {
        "include_genres": ["Drama", "Romance"],
        "genre_match": "any",
        "limit": agent.INITIAL_CANDIDATE_LIMIT,
    }


def test_distinct_synthetic_profiles_give_model_distinct_rating_examples():
    class EmptyCatalog(FakeCatalogTool):
        pass

    catalog = EmptyCatalog()
    agent = RecommendationAgent(
        catalog_tool=catalog,
        llm=FakeLLM([]),
    )
    request_context = _context(
        required_criteria=[],
        preferred_criteria=[],
        narrative_mechanisms=[],
        examples_interpretation=[],
        genres_include=[],
        genres_exclude=[],
    )
    drama_profile = UserTasteContext(
        preferred_genres=["Drama", "Romance"],
        rated_movie_examples=[
            {
                "title": "Amelie",
                "rating": 5,
                "genres": ["Comedy", "Romance"],
            },
            {
                "title": "Portrait of a Lady on Fire",
                "rating": 4.5,
                "genres": ["Drama", "Romance"],
            },
        ],
    )
    musical_profile = UserTasteContext(
        preferred_genres=["Comedy", "Musical"],
        rated_movie_examples=[
            {
                "title": "Cats",
                "rating": 5,
                "genres": ["Comedy", "Musical"],
            },
            {
                "title": "Another Musical",
                "rating": 4.5,
                "genres": ["Musical"],
            },
        ],
    )

    drama_prompt = json.loads(agent._build_user_message(
        request_context.model_copy(update={"taste_context": drama_profile}),
        [],
    ))
    musical_prompt = json.loads(agent._build_user_message(
        request_context.model_copy(update={"taste_context": musical_profile}),
        [],
    ))
    drama_taste = drama_prompt["recommendation_context"]["taste_context"]
    musical_taste = musical_prompt["recommendation_context"]["taste_context"]

    assert drama_taste["preferred_genres"] == ["Drama", "Romance"]
    assert musical_taste["preferred_genres"] == ["Comedy", "Musical"]
    assert drama_taste["rated_movie_examples"][0]["title"] == "Amelie"
    assert musical_taste["rated_movie_examples"][0]["title"] == "Cats"


def test_taste_genre_rating_only_breaks_ties_after_query_match_breadth():
    catalog = FakeCatalogTool({
        "thoughtful": [
            _movie(10, genres="Drama"),
            _movie(11, genres="Adventure"),
            _movie(12, genres="Horror"),
            {
                **_movie(13, genres="Horror"),
                "matched_queries": ["thoughtful", "psychological"],
            },
        ],
    })
    context = _context(
        required_criteria=["thoughtful"],
        preferred_criteria=[],
        narrative_mechanisms=[],
        examples_interpretation=[],
        genres_include=[],
        genres_exclude=[],
    )
    context.taste_context = UserTasteContext(
        genre_rating_summary=[
            {
                "genre": "Drama",
                "average_rating": 4.8,
                "rated_movie_count": 10,
            },
            {
                "genre": "Horror",
                "average_rating": 1.5,
                "rated_movie_count": 8,
            },
        ],
    )
    agent = RecommendationAgent(
        catalog_tool=catalog,
        llm=FakeLLM([]),
    )

    candidates = agent._initial_retrieval(context)

    assert [candidate["movieId"] for candidate in candidates] == [
        13,
        10,
        11,
        12,
    ]
    assert candidates[0]["taste_genre_match"]["average_rating"] == 1.5
    assert candidates[1]["taste_genre_match"]["average_rating"] == 4.8
    assert "taste_genre_match" not in candidates[2]


def test_personalization_scores_candidates_without_precomputed_relevance():
    taste = UserTasteContext(
        genre_rating_summary=[
            {
                "genre": "Sci-Fi",
                "average_rating": 4.5,
                "rated_movie_count": 5,
            },
        ],
    )
    candidates = [
        {
            **_movie(1),
            "matched_queries": ["disturbing"],
        },
        {
            **_movie(2),
            "matched_queries": ["psychologically intense"],
        },
    ]

    ranked = RecommendationAgent._personalize_candidates(
        candidates,
        taste,
    )

    assert [candidate["movieId"] for candidate in ranked] == [2, 1]


def test_multiple_tag_searches_prioritize_cross_query_matches():
    shared_candidate = _movie(50)
    query_results = {
        "secret agency": [
            *[_movie(movie_id) for movie_id in range(1, 9)],
            shared_candidate,
        ],
        "supernatural anomalies": [
            *[_movie(movie_id) for movie_id in range(9, 17)],
            shared_candidate,
        ],
    }
    catalog = FakeCatalogTool(query_results)
    agent = RecommendationAgent(catalog_tool=catalog, llm=FakeLLM([]))
    context = _context(genres_include=[])

    candidates = agent._initial_retrieval(context)

    assert catalog.tag_queries == [
        "secret agency",
        "supernatural anomalies",
    ]
    assert catalog.tag_query_limits == [
        agent.INITIAL_CANDIDATE_LIMIT * agent.TAG_SEARCH_RESULT_MULTIPLIER,
    ] * 2
    assert [candidate["movieId"] for candidate in candidates] == [
        50, 1, 2, 3, 4, 5, 6, 7,
    ]
    assert len(candidates) == agent.INITIAL_CANDIDATE_LIMIT


def test_multi_concept_match_beats_a_single_weak_tag_match():
    context = _context(
        narrative_mechanisms=[],
        required_criteria=[
            "foreshadowing",
            "psychologically intense",
            "shocking twists",
        ],
        preferred_criteria=[],
        intent_summary=(
            "A dark, psychologically intense movie with an unpredictable "
            "twist that leaves a lasting impression."
        ),
    )
    prestige = _movie(100, title="The Prestige (2006)")
    query_results = {
        "foreshadowing": [
            _movie(1, title="Avengers: Infinity War (2018)"),
        ],
        "psychologically intense": [
            *[_movie(movie_id) for movie_id in range(10, 18)],
            prestige,
        ],
        "shocking twists": [
            *[_movie(movie_id) for movie_id in range(20, 28)],
            prestige,
        ],
    }
    catalog = FakeCatalogTool(query_results)
    agent = RecommendationAgent(catalog_tool=catalog, llm=FakeLLM([]))

    candidates = agent._initial_retrieval(context)

    assert candidates[0]["title"] == "The Prestige (2006)"
    assert candidates[0]["matched_queries"] == [
        "psychologically intense",
        "shocking twists",
    ]
    assert all(
        candidate["title"] != "Avengers: Infinity War (2018)"
        for candidate in candidates
    )


def test_narrative_phrases_expand_to_catalog_tag_vocabulary():
    request = MovieRequest(
        intent_summary="A psychological film with shocking, unpredictable twists.",
        narrative_mechanisms=[
            "recontextualization",
            "hidden information available to attentive viewers",
        ],
        required_criteria=["psychologically intense"],
        preferred_criteria=[
            "unpredictable narratives",
            "shocking twists",
            "ambiguous endings",
        ],
    )

    queries = RecommendationAgent._tag_search_queries(request)

    assert "psychological" in queries
    assert "plot twist" in queries
    assert "mindfuck" in queries
    assert "unpredictable" in queries
    assert "twist" in queries
    assert "ambiguous" in queries


def test_vampire_request_searches_subject_and_predatory_evidence_not_foreshadowing():
    request = MovieRequest(
        intent_summary=(
            "User seeks movies featuring vampires that are unrefined, direct, "
            "and predatory in their behavior, avoiding traditional vampire "
            "tropes of elegance and social graces."
        ),
        narrative_mechanisms=["foreshadowing"],
        required_criteria=[
            "vampires exhibit predatory behavior without social refinement",
        ],
        preferred_criteria=[
            "vampires avoid traditional vampire tropes like politeness",
        ],
    )

    queries = RecommendationAgent._tag_search_queries(request)

    assert "vampire" in queries
    assert "blood" in queries
    assert "foreshadowing" not in queries


def test_vampire_candidates_matching_subject_and_style_beat_unrelated_tag_hits():
    request = MovieRequest(
        intent_summary=(
            "Movies featuring vampires that hunt people in a direct, "
            "predatory way."
        ),
        required_criteria=["vampires that hunt people"],
    )
    target = _movie(30, title="30 Days of Night (2007)")
    catalog = FakeCatalogTool({
        "vampire": [
            *[
                _movie(movie_id, title=f"Elegant Vampire Film {movie_id}")
                for movie_id in range(1, 18)
            ],
            target,
        ],
        "blood": [target],
    })
    agent = RecommendationAgent(catalog_tool=catalog, llm=FakeLLM([]))
    context = RecommendationContext(movie_request=request)

    candidates = agent._initial_retrieval(context)

    assert candidates[0]["title"] == "30 Days of Night (2007)"
    assert candidates[0]["matched_queries"] == ["vampire", "blood"]


def test_explicit_genres_are_included_in_tag_retrieval():
    context = RecommendationContext(
        movie_request=MovieRequest(
            intent_summary="Animated films",
            genres_include=["Animation"],
        )
    )
    catalog = FakeCatalogTool({
        "Animation": [_movie(1, genres="Animation|Children")],
    })
    agent = RecommendationAgent(catalog_tool=catalog, llm=FakeLLM([]))

    candidates = agent._initial_retrieval(context)

    assert catalog.tag_queries == ["Animation"]
    assert [candidate["movieId"] for candidate in candidates] == [1]


def test_duplicate_candidates_are_deduplicated_and_keep_all_matching_queries():
    first = _movie(10)
    catalog = FakeCatalogTool({
        "secret agency": [first, _movie(11)],
        "supernatural anomalies": [first, _movie(12)],
    })
    agent = RecommendationAgent(catalog_tool=catalog, llm=FakeLLM([]))

    candidates = agent._initial_retrieval(_context())
    candidate = next(item for item in candidates if item["movieId"] == 10)

    assert len({item["movieId"] for item in candidates}) == len(candidates)
    assert candidate["matched_queries"] == [
        "secret agency",
        "supernatural anomalies",
    ]


def test_explicit_numeric_and_year_filters_are_applied():
    context = _context(
        required_criteria=[],
        preferred_criteria=[],
        narrative_mechanisms=[],
    )
    context.filters = {
        "minimum_rating_count": 50,
        "minimum_rating": 3.5,
        "release_year_from": 2010,
        "release_year_to": 2020,
    }
    agent = RecommendationAgent(
        catalog_tool=FakeCatalogTool(),
        llm=FakeLLM([]),
    )

    assert agent._matches_explicit_filters(_movie(1), context)
    assert not agent._matches_explicit_filters(
        _movie(2, title="Old Movie (2009)"),
        context,
    )
    assert not agent._matches_explicit_filters(
        _movie(3, average_rating=3.0),
        context,
    )
    assert not agent._matches_explicit_filters(
        _movie(4, rating_count=10),
        context,
    )


def test_included_and_excluded_genres_are_enforced():
    context = _context(genres_include=["Sci-Fi", "Mystery"])
    agent = RecommendationAgent(
        catalog_tool=FakeCatalogTool(),
        llm=FakeLLM([]),
    )

    assert agent._matches_explicit_filters(_movie(1), context)
    assert not agent._matches_explicit_filters(
        _movie(2, genres="Horror|Mystery"),
        context,
    )
    assert agent._matches_explicit_filters(
        _movie(3, genres="Mystery"),
        context,
    )


def test_all_genre_match_requires_every_included_genre():
    context = _context(genres_include=["Sci-Fi", "Mystery"])
    context.movie_request.genres_match = "all"
    agent = RecommendationAgent(
        catalog_tool=FakeCatalogTool(),
        llm=FakeLLM([]),
    )

    assert agent._matches_explicit_filters(
        _movie(1, genres="Sci-Fi|Mystery"),
        context,
    )
    assert not agent._matches_explicit_filters(
        _movie(2, genres="Mystery"),
        context,
    )


def test_explicit_genre_filter_mode_overrides_analyzed_mode():
    context = _context(genres_include=["Sci-Fi", "Mystery"])
    context.movie_request.genres_match = "all"
    context.filters["genre_match"] = "any"
    agent = RecommendationAgent(
        catalog_tool=FakeCatalogTool(),
        llm=FakeLLM([]),
    )

    assert agent._matches_explicit_filters(
        _movie(1, genres="Mystery"),
        context,
    )


def test_watched_movies_do_not_use_candidate_slots():
    context = _context(
        required_criteria=["secret agency"],
        narrative_mechanisms=[],
    )
    context.taste_context = UserTasteContext(watched_movie_ids=list(range(1, 9)))
    catalog = FakeCatalogTool({
        "secret agency": [_movie(movie_id) for movie_id in range(1, 10)],
    })
    agent = RecommendationAgent(catalog_tool=catalog, llm=FakeLLM([]))

    candidates = agent._initial_retrieval(context)

    assert [candidate["movieId"] for candidate in candidates] == [9]


def test_title_exclusions_are_applied_without_treating_narrative_as_title():
    context = _context(
        required_criteria=[],
        preferred_criteria=[],
        narrative_mechanisms=[],
        exclude=["Catalog Movie 1", "avoid graphic violence"],
    )
    agent = RecommendationAgent(
        catalog_tool=FakeCatalogTool(),
        llm=FakeLLM([]),
    )

    assert not agent._matches_explicit_filters(_movie(1), context)
    assert agent._matches_explicit_filters(_movie(2), context)


def test_no_candidates_never_produce_a_model_invented_movie():
    catalog = FakeCatalogTool({
        "secret agency": [],
        "supernatural anomalies": [],
    })
    llm = FakeLLM([_response("No tool call."), _final_response(999)])
    agent = RecommendationAgent(catalog_tool=catalog, llm=llm)

    result = agent.run(_context())

    assert "Nu am găsit candidați" in result
    assert "Catalog Movie 999" not in result
    assert llm.messages == []
    assert llm.structured_prompts == []


def test_unknown_movie_ids_and_model_titles_are_not_accepted():
    catalog = FakeCatalogTool({
        "secret agency": [_movie(10)],
        "supernatural anomalies": [],
    })
    llm = FakeLLM([
        _response("No tool call."),
        _response(json.dumps({
            "recommendations": [
                {
                    "movieId": 999,
                    "title": "Invented Movie",
                    "reason": "A made-up result.",
                    "uncertainty": "Unknown.",
                },
                {
                    "movieId": 10,
                    "title": "Invented Movie",
                    "reason": "A possible match.",
                    "uncertainty": "The plot is not verified.",
                },
            ]
        })),
    ])
    agent = RecommendationAgent(catalog_tool=catalog, llm=llm)

    result = agent.run(_context())

    assert "Catalog Movie 10" in result
    assert "Invented Movie" not in result
    assert "A made-up result" not in result
    assert "A possible match" not in result
    assert len(llm.structured_prompts) == 1


def test_narrative_recommendations_always_show_catalog_limitations():
    catalog = FakeCatalogTool({
        "secret agency": [_movie(10)],
        "supernatural anomalies": [],
    })
    llm = FakeLLM([
        _response("No tool call."),
        _final_response(10, uncertainty=""),
    ])
    agent = RecommendationAgent(catalog_tool=catalog, llm=llm)

    result = agent.run(_context())

    assert "Catalog Movie 10" in result
    assert "Candidați posibili" in result
    assert "Nu pot confirma criteriile narative" in result


def test_narrative_requests_use_only_candidates_with_verified_synopses():
    catalog = FakeCatalogTool({
        "secret agency": [_movie(10), _movie(11)],
        "supernatural anomalies": [],
    })
    synopsis = {
        "plot_summary": (
            "A former prisoner tries to rebuild his life after release."
        ),
        "narrative_source": {
            "title": "Example film",
            "url": "https://en.wikipedia.org/wiki/Example_film",
            "revision_url": (
                "https://en.wikipedia.org/wiki/Example_film?oldid=123"
            ),
            "license_url": (
                "https://creativecommons.org/licenses/by-sa/4.0/deed.en"
            ),
        },
    }
    synopsis_tool = FakeSynopsisTool({10: synopsis})
    llm = FakeLLM([
        _response("No tool call."),
        _final_response(10),
    ])
    agent = RecommendationAgent(
        catalog_tool=catalog,
        llm=llm,
        synopsis_tool=synopsis_tool,
    )

    result = agent.run(_context())

    assert "Catalog Movie 10" in result
    assert "Catalog Movie 11" not in result
    assert "reintegration" in result
    assert "CC BY-SA 4.0" in result
    assert "oldid=123" in result
    final_payload = json.loads(llm.structured_prompts[-1])
    assert final_payload["validated_catalog_candidates"][0][
        "plot_summary"
    ] == synopsis["plot_summary"]
    assert "narrative_source" not in json.dumps(final_payload)


def test_narrative_requests_stop_when_no_verified_synopsis_exists():
    catalog = FakeCatalogTool({
        "secret agency": [_movie(10)],
        "supernatural anomalies": [],
    })
    llm = FakeLLM([])
    agent = RecommendationAgent(
        catalog_tool=catalog,
        llm=llm,
        synopsis_tool=FakeSynopsisTool(),
    )

    result = agent.run(_context())

    assert "sinopsis narativ verificabil" in result
    assert llm.messages == []
    assert llm.structured_prompts == []


def test_sourced_descriptions_are_limited_to_two_sentences():
    description = RecommendationAgent._limit_to_two_sentences(
        "First sentence. Second sentence! Third sentence."
    )

    assert description == "First sentence. Second sentence!"


def test_successful_tool_results_may_add_validated_candidates():
    initial = [_movie(movie_id) for movie_id in range(10, 17)]
    added = [
        _movie(20, title="Tool Result Movie (2018)"),
        _movie(21, title="Second Tool Result (2019)"),
        _movie(22, title="Third Tool Result (2021)"),
    ]
    catalog = FakeCatalogTool(
        {"secret agency": initial, "supernatural anomalies": []},
        search_results=added,
    )
    tool_call = SimpleNamespace(
        function=SimpleNamespace(
            name="search_movies",
            arguments={"query": "Tool Result Movie"},
        )
    )
    llm = FakeLLM([
        _response("", tool_calls=[tool_call]),
        _response("Additional search complete."),
        _final_response(20),
    ])
    agent = RecommendationAgent(catalog_tool=catalog, llm=llm)

    result = agent.run(_context())

    assert "Tool Result Movie (2018)" in result
    assert catalog.search_calls == [{"query": "Tool Result Movie"}]
    final_prompt = json.loads(llm.structured_prompts[-1])
    transmitted_candidates = final_prompt["validated_catalog_candidates"]
    assert len(transmitted_candidates) == agent.INITIAL_CANDIDATE_LIMIT
    transmitted_ids = [
        candidate["movieId"]
        for candidate in transmitted_candidates
    ]
    assert 20 in transmitted_ids
    assert 21 not in transmitted_ids
