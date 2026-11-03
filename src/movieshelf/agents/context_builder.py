from movieshelf.agents.request_analyzer import MovieRequestAnalyzer
from movieshelf.schemas.recommendation_context import RecommendationContext
from movieshelf.schemas.user_taste import UserTasteContext


class RecommendationContextBuilder:
    """Builds a temporary RecommendationContext for a single request."""

    def __init__(
        self,
        analyzer: MovieRequestAnalyzer | None = None,
    ):
        self.analyzer = analyzer or MovieRequestAnalyzer()

    def build(
        self,
        text: str | None = None,
        filters: dict | None = None,
        taste_context: UserTasteContext | None = None,
        random_recommendation: bool = False,
    ) -> RecommendationContext:

        movie_request = None

        if text and text.strip():
            movie_request = self.analyzer.analyze(text)

        return RecommendationContext(
            movie_request=movie_request,
            taste_context=taste_context,
            filters=filters or {},
            random_recommendation=random_recommendation,
        )