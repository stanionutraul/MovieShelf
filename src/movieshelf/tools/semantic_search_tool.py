
import json

from movieshelf.semantic_search import SemanticMovieSearchEngine


class SemanticSearchTool:
    """Exposes semantic movie search as an agent-usable tool."""

    name = "semantic_movie_search"

    description = (
        "Search movies based on the meaning and narrative concepts "
        "of a natural-language query. Supports genre and rating filters."
    )

    def __init__(self, search_engine: SemanticMovieSearchEngine):
        self.search_engine = search_engine

    def execute(
        self,
        query: str,
        n: int = 10,
        include_genres: list[str] | None = None,
        exclude_genres: list[str] | None = None,
        min_rating_count: int = 0,
        min_average_rating: float | None = None,
    ) -> list[dict]:

        if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 100:
            raise ValueError("n must be an integer between 1 and 100.")

        results = self.search_engine.search(
            query=query,
            n=n,
            include_genres=include_genres,
            exclude_genres=exclude_genres,
            min_rating_count=min_rating_count,
            min_average_rating=min_average_rating,
        )

        return json.loads(results.to_json(orient="records"))
