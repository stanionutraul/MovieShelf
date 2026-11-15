from movieshelf.agents.context_builder import RecommendationContextBuilder
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


def test_explicit_decade_is_extracted_as_hard_release_year_range():
    analyzer = MovieRequestAnalyzer(llm=FakeLLM(_result()))

    request = analyzer.analyze("Looking for funny movies from the 80s.")

    assert request.release_year_from == 1980
    assert request.release_year_to == 1989


def test_explicit_four_digit_decade_and_year_range_are_extracted():
    analyzer = MovieRequestAnalyzer(llm=FakeLLM(_result()))

    decade = analyzer.analyze("Recommend a film from the 1980s.")
    year_range = analyzer.analyze("Films released between 1990 and 1999.")

    assert (decade.release_year_from, decade.release_year_to) == (1980, 1989)
    assert (year_range.release_year_from, year_range.release_year_to) == (
        1990,
        1999,
    )


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


def test_unsupported_viewing_effort_criteria_are_removed():
    analyzer = MovieRequestAnalyzer(llm=FakeLLM(_result(
        required_criteria=["new_movie"],
        preferred_criteria=[
            "cozy_vibe",
            "popcorn_compatibility",
            "no_active_mental_engagement",
        ],
    )))

    request = analyzer.analyze(
        "It's 3AM and I want a cozy movie that feels new and different."
    )

    assert request.required_criteria == ["new_movie"]
    assert request.preferred_criteria == ["cozy_vibe"]

def test_optional_romance_and_holiday_themes_are_not_search_criteria():
    analyzer = MovieRequestAnalyzer(llm=FakeLLM(_result(
        required_criteria=[
            "romantic",
            "holiday-themed",
            "warm and comforting",
        ],
        preferred_criteria=["romantic or holiday-themed"],
        genres_include=["Romance"],
        genres_exclude=["Romance"],
    )))

    request = analyzer.analyze(
        "It doesn't have to be romantic, holiday-themed, or any particular "
        "genre. I want something warm and comforting."
    )

    assert request.non_constraints == ["romantic", "holiday-themed"]
    assert request.required_criteria == ["warm and comforting"]
    assert request.preferred_criteria == []
    assert request.genres_include == []
    assert request.genres_exclude == []


def test_bonus_points_are_not_promoted_to_required_criteria():
    analyzer = MovieRequestAnalyzer(llm=FakeLLM(_result(
        required_criteria=["subtitles"],
        preferred_criteria=["available on an OTT platform"],
    )))

    request = analyzer.analyze(
        "Language is no bar and I am happy with subtitles. Bonus points "
        "if it is available on an OTT platform."
    )

    assert request.required_criteria == []
    assert request.preferred_criteria == [
        "available on an OTT platform"
    ]


def test_context_builder_preserves_the_original_user_request():
    original_request = (
        "It's 3AM and I want a cozy movie that feels new and different."
    )
    analyzer = MovieRequestAnalyzer(llm=FakeLLM(_result()))
    context = RecommendationContextBuilder(analyzer=analyzer).build(
        text=original_request,
    )

    assert context.original_request == original_request


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


def test_viewing_experience_is_preserved():
    result = {
        "intent_summary": "A cozy, comforting and lighthearted movie.",
        "viewing_experience": {
            "atmosphere": ["cozy", "comforting", "warm"],
            "tone": ["lighthearted", "feel-good"],
            "mental_effort": "low",
            "emotional_intensity": None,
            "pacing": None,
        },
    }

    analyzer = MovieRequestAnalyzer(llm=FakeLLM(result))

    request = analyzer.analyze(
        "I want something cozy, comforting, lighthearted and "
        "that doesn't require too much mental energy."
    )

    assert request.viewing_experience.atmosphere == [
        "cozy", "comforting", "warm"
    ]
    assert request.viewing_experience.tone == [
        "lighthearted", "feel-good"
    ]
    assert request.viewing_experience.mental_effort == "low"
    assert request.viewing_experience.emotional_intensity is None
    assert request.viewing_experience.pacing is None


def test_viewing_experience_does_not_infer_emotional_intensity():
    result = {
        "intent_summary": "A warm and feel-good movie.",
        "viewing_experience": {
            "atmosphere": ["warm"],
            "tone": ["feel-good"],
        },
    }

    analyzer = MovieRequestAnalyzer(llm=FakeLLM(result))

    request = analyzer.analyze("Something warm and feel-good.")

    assert request.viewing_experience.emotional_intensity is None
    assert request.viewing_experience.mental_effort is None
    assert request.viewing_experience.pacing is None