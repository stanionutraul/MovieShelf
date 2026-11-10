from movieshelf.agents.request_analyzer import MovieRequestAnalyzer


class FakeLLM:
    def __init__(self, result):
        self.result = result
        self.prompt = None

    def generate_structured(self, prompt, schema):
        self.prompt = prompt
        return self.result


def _result(**overrides):
    result = {
        "intent_summary": "Find a movie matching the request.",
        "genres_include": [],
        "genres_exclude": [],
    }
    result.update(overrides)
    return result


def test_inferred_genres_not_mentioned_by_user_are_removed():
    llm = FakeLLM(_result(
        genres_include=["science fiction", "mystery"],
        genres_exclude=["horror"],
    ))
    analyzer = MovieRequestAnalyzer(llm=llm)

    request = analyzer.analyze(
        "Movies about a secret agency dealing with anomalies, like "
        "Men in Black. I am not looking for TV shows."
    )

    assert request.genres_include == []
    assert request.genres_exclude == []
    assert "explicitly asks for a genre" in llm.prompt


def test_explicit_included_and_excluded_genres_are_preserved():
    analyzer = MovieRequestAnalyzer(llm=FakeLLM(_result(
        genres_include=["science fiction", "Mystery"],
        genres_exclude=["horror"],
    )))

    request = analyzer.analyze(
        "I want science fiction and mystery movies, but no horror."
    )

    assert request.genres_include == ["science fiction", "Mystery"]
    assert request.genres_exclude == ["horror"]
    assert request.genres_match == "any"


def test_explicit_all_genre_mode_is_preserved():
    analyzer = MovieRequestAnalyzer(llm=FakeLLM(_result(
        genres_include=["Drama", "Fantasy"],
        genres_match="all",
    )))

    request = analyzer.analyze(
        "I need a movie that is both drama and fantasy."
    )

    assert request.genres_include == ["Drama", "Fantasy"]
    assert request.genres_match == "all"


def test_all_genre_mode_requires_explicit_all_language():
    analyzer = MovieRequestAnalyzer(llm=FakeLLM(_result(
        genres_include=["Drama", "Fantasy"],
        genres_match="all",
    )))

    request = analyzer.analyze("I want drama and fantasy films.")

    assert request.genres_match == "any"


def test_moods_and_viewing_preferences_do_not_become_genre_filters():
    analyzer = MovieRequestAnalyzer(llm=FakeLLM(_result(
        genres_include=["Drama"],
        genres_exclude=["Horror", "Sci-Fi", "Mystery"],
    )))

    request = analyzer.analyze(
        "I want cozy, easy-to-watch animated movies, but nothing too "
        "intense or emotional."
    )

    assert request.genres_include == []
    assert request.genres_exclude == []


def test_explicit_animation_and_horror_exclusion_are_kept():
    analyzer = MovieRequestAnalyzer(llm=FakeLLM(_result(
        genres_include=["Animation"],
        genres_exclude=["Horror"],
    )))

    request = analyzer.analyze(
        "I want animated films, but I do not want horror."
    )

    assert request.genres_include == ["Animation"]
    assert request.genres_exclude == ["Horror"]


def test_clear_content_avoidances_are_extracted_deterministically():
    analyzer = MovieRequestAnalyzer(llm=FakeLLM(_result(
        content_avoidances=["gore", "jump scares"],
    )))

    request = analyzer.analyze(
        "Recommend a dark thriller, but no gore or jump scares."
    )

    assert request.content_avoidances == ["gore", "jump scares"]


def test_soft_content_preferences_are_not_treated_as_hard_avoidances():
    analyzer = MovieRequestAnalyzer(llm=FakeLLM(_result(
        content_avoidances=["gore", "jump scares"],
    )))

    request = analyzer.analyze(
        "I'm not necessarily looking for cheap jump scares or gore "
        "just for the sake of it."
    )

    assert request.content_avoidances == []


def test_romanian_content_avoidances_are_recognized():
    analyzer = MovieRequestAnalyzer(llm=FakeLLM(_result()))

    request = analyzer.analyze(
        "Vreau un thriller fără gore sau jump scare-uri."
    )

    assert request.content_avoidances == ["gore", "jump scares"]


def test_explicit_content_avoidance_fails_closed_without_verified_data():
    from movieshelf.agents.recommendation_agent import RecommendationAgent
    from movieshelf.schemas.movie_request import MovieRequest
    from movieshelf.schemas.recommendation_context import RecommendationContext

    class EmptyCatalog:
        def search_movies(self, *args, **kwargs):
            return []

        def filter_movies(self, *args, **kwargs):
            return []

        def get_random_movies(self, *args, **kwargs):
            return []

        def get_movies_by_tags(self, *args, **kwargs):
            return []

        def get_movie_details(self, *args, **kwargs):
            return []

    llm = FakeLLM(_result())
    context = RecommendationContext(
        movie_request=MovieRequest(
            intent_summary="Find a thriller.",
            content_avoidances=["gore", "jump scares"],
        ),
    )
    agent = RecommendationAgent(
        catalog_tool=EmptyCatalog(),
        llm=llm,
    )

    result = agent.run(context)

    assert "nu conține avertismente de conținut verificate" in result
    assert llm.prompt is None
