
import json
import logging
import math
import re
from typing import Any, Protocol

from pydantic import BaseModel, Field

from movieshelf.llm import MovieShelfLLM
from movieshelf.schemas.recommendation_context import RecommendationContext
from movieshelf.schemas.user_taste import UserTasteContext
from movieshelf.tools.movie_catalog_tool import MovieCatalogTool


logger = logging.getLogger(__name__)


class MovieCatalogSearchEngine(Protocol):
    def search(self, query: str, n: int = 10) -> Any:
        ...


class CandidateEnricher(Protocol):
    def enrich_candidates(
        self,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ...


class MovieRecommendation(BaseModel):
    movieId: int
    description: str = Field(
        default="",
        description=(
            "One or two short sentences, grounded only in the supplied "
            "licensed movie synopsis."
        ),
    )


class MovieRecommendationResponse(BaseModel):
    recommendations: list[MovieRecommendation]


class RecommendationAgent:
    """Coordinates deterministic catalog retrieval and LLM-assisted evaluation."""

    MAX_TOOL_CALLS = 5
    INITIAL_CANDIDATE_LIMIT = 8
    RETRIEVAL_CANDIDATE_LIMIT = 100
    TAG_SEARCH_RESULT_MULTIPLIER = 4
    MAX_TAGS_PER_CANDIDATE = 5
    MAX_PLOT_PROMPT_CHARACTERS = 1200
    MAX_FINAL_PLOT_PROMPT_CHARACTERS = 600
    MAX_PROMPT_CANDIDATES = 4
    MAX_TASTE_REVIEWS_IN_PROMPT = 3
    MAX_TASTE_REVIEW_CHARACTERS = 300
    TAG_QUERY_EXPANSIONS = (
        (
            "cozy",
            (
                "warm",
                "heartwarming",
                "feel-good",
                "feel good",
                "lighthearted",
                "comforting",
            ),
        ),
        (
            "uplifting",
            ("heartwarming", "feel-good", "feel good", "lighthearted", "warm"),
        ),
        (
            "light mood",
            ("heartwarming", "feel-good", "feel good", "lighthearted", "warm"),
        ),
        (
            "mood-boosting",
            ("heartwarming", "feel-good", "feel good", "lighthearted", "warm"),
        ),
        (
            "destress",
            ("heartwarming", "feel-good", "feel good", "lighthearted", "warm"),
        ),
        (
            "minimal stress",
            ("feel-good", "feel good", "lighthearted", "silly", "funny"),
        ),
        ("vampire", ("vampire",)),
        ("predatory", ("blood",)),
        ("bloodthirsty", ("blood",)),
        ("hunt people", ("blood",)),
        ("psycholog", ("psychological",)),
        ("recontextual", ("plot twist",)),
        ("hidden information", ("mindfuck",)),
        ("unpredictable", ("unpredictable",)),
        ("shocking twist", ("twist",)),
        ("ambiguous ending", ("ambiguous",)),
    )
    NARRATIVE_MECHANISM_SUPPORT = {
        "foreshadowing": re.compile(
            r"\b(?:foreshadow\w*|clues?|hints?|predict\w*|twists?|"
            r"reveal\w*|hidden information)\b",
            re.IGNORECASE,
        ),
        "hidden information available to attentive viewers": re.compile(
            r"\b(?:hidden information|secrets?|clues?|hints?|"
            r"reveal\w*|twists?)\b",
            re.IGNORECASE,
        ),
        "recontextualization": re.compile(
            r"\b(?:recontextual\w*|reinterpret\w*|twists?|"
            r"reveal\w*|change the way you look)\b",
            re.IGNORECASE,
        ),
    }
    CANDIDATE_PROMPT_FIELDS = {
        "movieId",
        "movie_id",
        "title",
        "genres",
        "average_rating",
        "rating_count",
        "tags",
        "matched_queries",
        "retrieval_score",
        "request_relevance_score",
        "plot_summary",
        "taste_genre_match",
    }
    SUPPORTED_FILTERS = {
        "min_rating_count",
        "minimum_rating_count",
        "min_average_rating",
        "minimum_rating",
        "release_year_from",
        "release_year_to",
        "include_genres",
        "exclude_genres",
        "genre_match",
    }

    SYSTEM_PROMPT = """
You are MovieShelf, an AI movie discovery agent.

Your task is to find movies that match the user's request using the
catalog information provided by Python tools.

The recommendation_context in the user message is the user's request.
The catalog candidates are search results, not a replacement for that
request. Never infer the request from candidate metadata.

IMPORTANT RULES:

1. Use the retrieved catalog data as the source of truth.
2. Never invent movie titles, release years, ratings, genres, tags or IDs.
3. Do not claim that a movie satisfies a narrative requirement unless
   the retrieved information supports that conclusion.
4. MovieLens tags are user-contributed metadata. Treat them as clues,
   not as verified plot descriptions.
5. Distinguish mandatory criteria from preferences.
6. Do not treat assumptions or examples as mandatory criteria.
7. Do not interpret a preferred genre or style as an explicit genre filter.
8. Do not recommend movies listed in watched_movie_ids.
9. Disliked movies are information about user taste, not automatically
   titles that must be excluded.
10. Use the user's taste_summary, preferred_genres, and review_examples
    as preference signals, not as mandatory filters. The current request
    and its explicit filters take priority.
11. Separate the user's personal circumstances from the movie qualities
    they ask for. Do not treat hardship mentioned as context as a request
    for a movie about the same hardship.
12. Use rated_movie_examples only as examples of the user's taste. Their
    genres are catalog metadata; ratings do not prove why the user liked or
    disliked a film. Do not infer plot facts from these examples.
13. Use taste_genre_match only as a soft tie-breaker between candidates
    that are similarly relevant to the current request.
14. Treat review_examples as the user's opinions, not verified plot facts.
15. If the available data cannot verify a criterion, explicitly say so.
16. Respond in the same language as the user's request.
17. Do not reveal internal reasoning, tool plans, or implementation details.

TOOL USAGE:

- search_movies: search the catalog by movie title or text.
- filter_movies: apply explicit genre and rating-statistic criteria.
- get_random_movies: retrieve a random catalog sample when requested.
- get_movies_by_tags: find movies associated with relevant catalog tags.
- get_movie_details: retrieve the available information for a movie ID.

Python has already performed the initial catalog searches and applied
explicit, verifiable filters. Use the supplied candidates to evaluate the
request. Call catalog tools only when additional information is needed.
Do not repeat a search that has already returned the same information.
Treat MovieLens tags as user-contributed clues, never as verified plot facts.

FINAL RESPONSE:

Recommend only movies present in the retrieved results.
For each recommendation, include the title and year when available.
Explain briefly which user criteria are supported by the retrieved data.
Mention relevant limitations when the catalog does not contain enough
information to verify a narrative requirement.
"""


    def __init__(
        self,
        catalog_tool: MovieCatalogTool,
        llm: MovieShelfLLM | None = None,
        synopsis_tool: CandidateEnricher | None = None,
        search_engine: MovieCatalogSearchEngine | None = None,
    ):
        self.catalog_tool = catalog_tool
        self.llm = llm or MovieShelfLLM()
        self.synopsis_tool = synopsis_tool
        self.search_engine = search_engine

        self.tool_functions = {
            "search_movies": self.catalog_tool.search_movies,
            "filter_movies": self.catalog_tool.filter_movies,
            "get_random_movies": self.catalog_tool.get_random_movies,
            "get_movies_by_tags": self.catalog_tool.get_movies_by_tags,
            "get_movie_details": self.catalog_tool.get_movie_details,
        }

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_movies",
                    "description": (
                        "Search the movie catalog by movie title or text."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Movie title or search text.",
                            },
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 100,
                                "description": "Maximum number of results.",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "filter_movies",
                    "description": (
                        "Filter movies by genres and rating statistics."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "include_genres": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "genre_match": {
                                "type": "string",
                                "enum": ["any", "all"],
                                "description": (
                                    "Match any included genre (default) or "
                                    "require all included genres."
                                ),
                            },
                            "exclude_genres": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "min_rating_count": {
                                "type": "integer",
                                "minimum": 0,
                            },
                            "min_average_rating": {"type": "number"},
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 100,
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_random_movies",
                    "description": "Retrieve a random selection of movies.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 100,
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_movies_by_tags",
                    "description": (
                        "Find movies associated with relevant MovieLens tags."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Tag or concept to search for.",
                            },
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 100,
                                "description": "Maximum number of results.",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_movie_details",
                    "description": (
                        "Retrieve details and available tags for a movie."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "movie_id": {
                                "type": "integer",
                                "description": "MovieLens movie ID.",
                            },
                        },
                        "required": ["movie_id"],
                    },
                },
            },
        ]


    @staticmethod
    def _normalize_genre(genre: str) -> str:
        """Convert common genre names to MovieLens genre names."""

        normalized = genre.strip().lower()

        aliases = {
            "science fiction": "Sci-Fi",
            "sci fi": "Sci-Fi",
            "sci-fi": "Sci-Fi",
            "animated": "Animation",
            "cartoons": "Animation",
            "children's": "Children",
            "children": "Children",
            "romantic comedy": "Romance",
            "film noir": "Film-Noir",
            "noir": "Film-Noir",
        }

        if normalized in aliases:
            return aliases[normalized]

        return genre.strip().title()


    @classmethod
    def _normalize_genres(cls, genres: list[str] | None) -> list[str] | None:
        if not genres:
            return None

        return list(dict.fromkeys(
            cls._normalize_genre(genre)
            for genre in genres
            if isinstance(genre, str) and genre.strip()
        ))


    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        """Convert tool results to JSON-compatible values."""

        if hasattr(value, "to_dict"):
            return value.to_dict(orient="records")

        if hasattr(value, "model_dump"):
            return value.model_dump()

        if isinstance(value, list):
            return [
                RecommendationAgent._to_jsonable(item)
                for item in value
            ]

        if isinstance(value, dict):
            return {
                str(key): RecommendationAgent._to_jsonable(item)
                for key, item in value.items()
            }

        return value


    @staticmethod
    def _get_movie_id(movie: dict) -> int | None:
        """Read a MovieLens ID from common result field names."""

        for key in ("movie_id", "movieId", "id"):
            value = movie.get(key)

            if value is not None:
                if isinstance(value, bool):
                    return None

                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None

        return None


    def _remove_watched_movies(
        self,
        movies: Any,
        watched_movie_ids: list[int],
    ) -> Any:
        """Remove watched titles from a list of catalog results."""

        if not watched_movie_ids or not isinstance(movies, list):
            return movies

        watched_ids = set(watched_movie_ids)

        return [
            movie
            for movie in movies
            if not isinstance(movie, dict)
            or self._get_movie_id(movie) not in watched_ids
        ]


    def _initial_retrieval(
        self,
        context: RecommendationContext,
    ) -> list[dict]:
        """Retrieve and deterministically filter the initial candidates."""

        movie_request = context.movie_request
        taste_context = context.taste_context
        self._validated_filters(context.filters)

        if context.random_recommendation:
            candidates = self.catalog_tool.get_random_movies(
                limit=self.INITIAL_CANDIDATE_LIMIT
            )

        elif movie_request and (
            movie_request.narrative_mechanisms
            or movie_request.required_criteria
            or movie_request.preferred_criteria
            or movie_request.examples_interpretation
            or movie_request.intent_summary.strip()
        ):
            queries = self._tag_search_queries(
                movie_request,
                context.original_request or "",
            )
            query_results = []
            for query in queries:
                matches = self.catalog_tool.get_movies_by_tags(
                    query=query,
                    limit=(
                        self.INITIAL_CANDIDATE_LIMIT
                        * self.TAG_SEARCH_RESULT_MULTIPLIER
                    ),
                )

                normalized_matches = self._to_jsonable(matches)
                if not isinstance(normalized_matches, list):
                    raise TypeError(
                        "Tag search must return a list of movie records."
                    )

                query_results.append((query, normalized_matches))

            lexical_candidates = self._search_catalog_candidates(
                movie_request,
                context,
            )
            candidates = self._select_balanced_candidates(
                query_results,
                context,
                seed_candidates=lexical_candidates,
                allowed_movie_ids=(
                    self._available_narrative_movie_ids(context)
                ),
            )

            if (
                not candidates
                and movie_request
                and not (
                    movie_request.narrative_mechanisms
                    or movie_request.required_criteria
                    or movie_request.preferred_criteria
                    or movie_request.examples_interpretation
                )
                and (
                    movie_request.genres_include
                    or movie_request.genres_exclude
                    or context.filters
                )
            ):
                candidates = self._genre_filtered_catalog(context)

        elif context.filters:
            candidates = self._genre_filtered_catalog(context)

        elif movie_request and self.search_engine is not None:
            candidates = self._search_catalog_candidates(
                movie_request,
                context,
            )

        elif taste_context:
            preferred_genres = taste_context.preferred_genres
            if preferred_genres:
                candidates = self.catalog_tool.filter_movies(
                    include_genres=preferred_genres,
                    genre_match="any",
                    limit=self.INITIAL_CANDIDATE_LIMIT,
                )
            else:
                candidates = self.catalog_tool.get_random_movies(
                    limit=self.INITIAL_CANDIDATE_LIMIT
                )

        else:
            candidates = []

        candidates = self._to_jsonable(candidates)
        if not isinstance(candidates, list):
            raise TypeError(
                "Initial catalog retrieval must return a list of movies."
            )

        watched_ids = (
            taste_context.watched_movie_ids
            if taste_context
            else []
        )

        candidates = self._remove_watched_movies(
            candidates,
            watched_ids,
        )

        candidates = [
            candidate
            for candidate in candidates
            if isinstance(candidate, dict)
            and self._get_movie_id(candidate) is not None
            and isinstance(candidate.get("title"), str)
            and self._matches_explicit_filters(candidate, context)
        ]

        candidates = candidates[:self.INITIAL_CANDIDATE_LIMIT]
        return self._personalize_candidates(
            candidates,
            taste_context,
        )

    def _search_catalog_candidates(
        self,
        movie_request,
        context: RecommendationContext,
    ) -> list[dict]:
        if self.search_engine is None:
            return []

        query = self._catalog_search_query(movie_request)
        if not query:
            return []

        results = self.search_engine.search(
            query,
            n=self.RETRIEVAL_CANDIDATE_LIMIT,
        )
        movies = getattr(self.catalog_tool, "movies", None)
        if movies is None or not hasattr(movies, "to_dict"):
            return []

        catalog_records = {
            self._get_movie_id(record): record
            for record in movies.to_dict(orient="records")
            if isinstance(record, dict)
        }
        taste = context.taste_context
        watched_ids = set(taste.watched_movie_ids) if taste else set()
        candidates = []

        for result in results.to_dict(orient="records"):
            movie_id = self._get_movie_id(result)
            movie = catalog_records.get(movie_id)
            if (
                movie is None
                or movie_id in watched_ids
                or not self._matches_explicit_filters(movie, context)
            ):
                continue

            candidates.append({
                **movie,
                "retrieval_score": float(
                    result["semantic_score"]
                    if "semantic_score" in result
                    else result["final_score"]
                ),
            })

        return candidates


    def _available_narrative_movie_ids(
        self,
        context: RecommendationContext,
    ) -> set[int] | None:
        if (
            self.synopsis_tool is None
            or not self._requires_narrative_uncertainty(context)
        ):
            return None

        get_movie_ids = getattr(
            self.synopsis_tool,
            "movie_ids_with_status",
            None,
        )
        if not callable(get_movie_ids):
            return None

        return get_movie_ids("available")


    @classmethod
    def _catalog_search_query(cls, movie_request) -> str:
        unsupported_metadata_terms = {
            "available", "availability", "english", "different",
            "major", "new", "platform", "platforms", "region",
            "regions", "series", "show", "shows", "streaming",
            "subtitle", "subtitles", "tv",
        }
        phrases = [
            *movie_request.required_criteria,
            *movie_request.preferred_criteria,
            *movie_request.genres_include,
        ]
        if not phrases:
            phrases.append(movie_request.intent_summary)
        words = []
        seen = set()
        for phrase in phrases:
            for word in re.findall(r"[A-Za-z0-9'-]+", phrase):
                normalized = word.casefold()
                if (
                    normalized in unsupported_metadata_terms
                    or normalized in seen
                ):
                    continue
                seen.add(normalized)
                words.append(word.replace("_", " "))

        for year in (
            movie_request.release_year_from,
            movie_request.release_year_to,
        ):
            if year is not None and str(year) not in seen:
                words.append(str(year))
                seen.add(str(year))

        return " ".join(words[:32])


    @classmethod
    def _personalize_candidates(
        cls,
        candidates: list[dict],
        taste_context: UserTasteContext | None,
    ) -> list[dict]:
        """Use rated genre averages only to break ties in request relevance."""

        if not taste_context or not taste_context.genre_rating_summary:
            return candidates

        genre_ratings = {
            item.genre.casefold(): item
            for item in taste_context.genre_rating_summary
        }
        ranked_candidates = []
        for index, candidate in enumerate(candidates):
            movie_genres = {
                genre.strip().casefold()
                for genre in str(candidate.get("genres", "")).split("|")
                if genre.strip()
            }
            matches = [
                summary
                for genre, summary in genre_ratings.items()
                if genre in movie_genres
            ]
            if matches:
                weights = [
                    min(summary.rated_movie_count, 10)
                    for summary in matches
                ]
                score = sum(
                    summary.average_rating * weight
                    for summary, weight in zip(matches, weights)
                ) / sum(weights)
                candidate["taste_genre_match"] = {
                    "average_rating": round(score, 3),
                    "matched_genres": [
                        {
                            "genre": summary.genre,
                            "average_rating": round(
                                summary.average_rating,
                                3,
                            ),
                            "rated_movie_count": summary.rated_movie_count,
                        }
                        for summary in matches
                    ],
                }
            relevance_score = candidate.get("request_relevance_score")
            if relevance_score is None:
                matched_queries = candidate.get("matched_queries", [])
                relevance_score = sum(
                    cls._query_specificity(query)
                    for query in matched_queries
                    if isinstance(query, str)
                )
            ranked_candidates.append((
                -relevance_score,
                -candidate.get(
                    "taste_genre_match",
                    {"average_rating": 2.5},
                ).get("average_rating", -1.0),
                index,
                candidate,
            ))

        ranked_candidates.sort(key=lambda item: item[:3])
        return [item[3] for item in ranked_candidates]


    def _select_balanced_candidates(
        self,
        query_results: list[tuple[str, list[dict]]],
        context: RecommendationContext,
        seed_candidates: list[dict] | None = None,
        allowed_movie_ids: set[int] | None = None,
    ) -> list[dict]:
        """Rank full-text matches, with matching tags as additional evidence."""

        candidates_by_id = {}
        first_seen_order = {}
        for semantic_rank, record in enumerate(
            seed_candidates or [],
            start=1,
        ):
            movie_id = self._get_movie_id(record)
            if (
                movie_id is None
                or (
                    allowed_movie_ids is not None
                    and movie_id not in allowed_movie_ids
                )
            ):
                continue
            candidate = dict(record)
            candidate["semantic_rank"] = semantic_rank
            candidates_by_id[movie_id] = candidate
            first_seen_order[movie_id] = len(first_seen_order)

        for query, records in query_results:
            seen_in_query = set()

            for tag_rank, record in enumerate(records, start=1):
                if not isinstance(record, dict):
                    continue

                movie_id = self._get_movie_id(record)
                if (
                    movie_id is None
                    or (
                        allowed_movie_ids is not None
                        and movie_id not in allowed_movie_ids
                    )
                    or movie_id in seen_in_query
                    or (
                        context.taste_context
                        and movie_id
                        in context.taste_context.watched_movie_ids
                    )
                    or not isinstance(record.get("title"), str)
                    or not self._matches_explicit_filters(record, context)
                ):
                    continue

                seen_in_query.add(movie_id)
                if movie_id not in candidates_by_id:
                    candidates_by_id[movie_id] = dict(record)
                    first_seen_order[movie_id] = len(first_seen_order)
                candidate = candidates_by_id[movie_id]
                matched_queries = candidate.setdefault("matched_queries", [])
                if query not in matched_queries:
                    matched_queries.append(query)
                candidate.setdefault("tag_ranks", {})[query] = tag_rank

        for candidate in candidates_by_id.values():
            semantic_rank = candidate.get("semantic_rank")
            semantic_score = (
                2 / (60 + semantic_rank)
                if isinstance(semantic_rank, int)
                else 0.0
            )
            tag_score = sum(
                self._query_specificity(query) / (60 + rank)
                for query, rank in candidate.get("tag_ranks", {}).items()
            )
            candidate["request_relevance_score"] = (
                semantic_score + tag_score
            )

        selected_ids = sorted(
            candidates_by_id,
            key=lambda movie_id: (
                -candidates_by_id[movie_id]["request_relevance_score"],
                -len(candidates_by_id[movie_id].get("matched_queries", [])),
                first_seen_order[movie_id],
            ),
        )[:self.INITIAL_CANDIDATE_LIMIT]

        selected = [candidates_by_id[movie_id] for movie_id in selected_ids]
        self._attach_candidate_tags(selected)
        return selected


    @staticmethod
    def _query_specificity(query: str) -> float:
        """Give multi-word request concepts more weight than broad single tags."""

        word_count = len(re.findall(r"\b\w+\b", query))
        return min(2.0, 1.0 + 0.5 * max(0, word_count - 1))


    def _attach_candidate_tags(self, candidates: list[dict]) -> None:
        """Attach only the actual MovieLens tags matching searched concepts."""

        for candidate in candidates:
            movie_id = self._get_movie_id(candidate)
            details = self.catalog_tool.get_movie_details(movie_id)
            if not isinstance(details, dict):
                candidate["tags"] = []
                continue

            tag_rows = details.get("tags", [])
            queries = candidate.get("matched_queries", [])
            relevant_tags = []

            for tag_row in tag_rows:
                if not isinstance(tag_row, dict):
                    continue

                tag = tag_row.get("tag")
                if not isinstance(tag, str):
                    continue

                if any(query.casefold() in tag.casefold() for query in queries):
                    relevant_tags.append({
                        "tag": tag,
                        "tag_count": tag_row.get("tag_count"),
                    })

            relevant_tags.sort(
                key=lambda item: (
                    item.get("tag_count") or 0,
                    item["tag"].casefold(),
                ),
                reverse=True,
            )
            candidate["tags"] = relevant_tags[:self.MAX_TAGS_PER_CANDIDATE]


    def _genre_filtered_catalog(
        self,
        context: RecommendationContext,
    ) -> list[dict]:
        """Filter catalog rows directly, then sample deterministically by ID."""

        movies = getattr(self.catalog_tool, "movies", None)
        if movies is None or not hasattr(movies, "to_dict"):
            return []

        records = movies.to_dict(orient="records")
        eligible = [
            record
            for record in records
            if isinstance(record, dict)
            and self._get_movie_id(record) is not None
            and isinstance(record.get("title"), str)
            and not (
                context.taste_context
                and self._get_movie_id(record)
                in context.taste_context.watched_movie_ids
            )
            and self._matches_explicit_filters(record, context)
        ]
        eligible.sort(key=lambda movie: self._get_movie_id(movie))

        if len(eligible) <= self.INITIAL_CANDIDATE_LIMIT:
            return eligible

        last_index = len(eligible) - 1
        indexes = {
            round(
                index * last_index
                / (self.INITIAL_CANDIDATE_LIMIT - 1)
            )
            for index in range(self.INITIAL_CANDIDATE_LIMIT)
        }
        return [eligible[index] for index in sorted(indexes)]


    @classmethod
    def _tag_search_queries(
        cls,
        movie_request,
        original_request: str = "",
    ) -> list[str]:
        """Extract a few focused tag queries from the structured request."""

        supporting_text = " ".join([
            movie_request.intent_summary,
            original_request,
            *movie_request.required_criteria,
            *movie_request.preferred_criteria,
        ])
        phrases = [
            *(
                mechanism
                for mechanism in movie_request.narrative_mechanisms
                if (
                    (support := cls.NARRATIVE_MECHANISM_SUPPORT.get(
                        mechanism.casefold()
                    ))
                    is None
                    or support.search(supporting_text)
                )
            ),
            *movie_request.required_criteria,
            *movie_request.preferred_criteria,
            *(
                mechanism
                for example in movie_request.examples_interpretation
                for mechanism in example.relevant_mechanisms
            ),
            *movie_request.genres_include,
        ]

        source_text = " ".join([supporting_text, *phrases])
        if not phrases:
            phrases = [movie_request.intent_summary]

        ignored_words = {
            "about", "after", "also", "and", "deals", "from",
            "into", "looking", "movies", "movie", "that", "their",
            "these", "this", "those", "with", "without", "new",
            "different", "available", "availability", "english",
            "major", "platform", "platforms", "region", "regions",
            "series", "show", "shows", "streaming", "subtitle",
            "subtitles", "tv",
        }
        queries = []

        def add_query(value: str) -> None:
            query = value.strip(" .!?")

            if (
                len(query) > 2
                and query.casefold()
                not in {existing.casefold() for existing in queries}
            ):
                queries.append(query)

        for phrase in phrases:
            for part in re.split(
                r"[,;]|\b(?:and|or|but|without)\b",
                phrase,
                flags=re.IGNORECASE,
            ):
                words = [
                    word
                    for word in re.findall(r"[A-Za-z0-9'-]+", part)
                    if word.casefold() not in ignored_words
                ]
                add_query(" ".join(words))
                normalized_part = " ".join(words).casefold()
                for marker, expansions in cls.TAG_QUERY_EXPANSIONS:
                    if marker in normalized_part:
                        for expansion in expansions:
                            add_query(expansion)

        for marker, expansions in cls.TAG_QUERY_EXPANSIONS:
            if re.search(rf"\b{re.escape(marker)}\b", source_text, re.I):
                for expansion in expansions:
                    add_query(expansion)

        return queries


    @classmethod
    def _filter_genres(
        cls,
        context: RecommendationContext,
        key: str,
    ) -> list[str]:
        """Combine request genres with comma- or pipe-separated context filters."""

        request = context.movie_request
        request_values = (
            getattr(request, key, [])
            if request
            else []
        )
        filter_key = (
            "include_genres"
            if key == "genres_include"
            else "exclude_genres"
        )
        filter_value = context.filters.get(filter_key, "")
        filter_values = (
            re.split(r"[,|]", filter_value)
            if isinstance(filter_value, str)
            else []
        )
        return cls._normalize_genres(
            [*request_values, *filter_values]
        ) or []


    @classmethod
    def _validated_filters(
        cls,
        filters: dict[str, str | int | float | bool],
    ) -> dict[str, str | int | float]:
        unsupported = set(filters) - cls.SUPPORTED_FILTERS
        if unsupported:
            raise ValueError(
                "Unsupported recommendation filters: "
                + ", ".join(sorted(unsupported))
            )

        validated = {}
        for key, value in filters.items():
            if key in {"min_rating_count", "minimum_rating_count"}:
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                ):
                    raise ValueError(f"{key} must be a non-negative integer.")
            elif key in {"min_average_rating", "minimum_rating"}:
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not 0 <= value <= 5
                ):
                    raise ValueError(f"{key} must be between 0 and 5.")
            elif key in {"release_year_from", "release_year_to"}:
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 1870
                ):
                    raise ValueError(f"{key} must be a valid release year.")
            elif key in {"include_genres", "exclude_genres"}:
                if not isinstance(value, str):
                    raise ValueError(f"{key} must be a string.")
            elif key == "genre_match":
                if not isinstance(value, str) or value not in {"any", "all"}:
                    raise ValueError(
                        "genre_match must be either 'any' or 'all'."
                    )

            validated[key] = value

        return validated


    @classmethod
    def _matches_explicit_filters(
        cls,
        movie: dict,
        context: RecommendationContext,
    ) -> bool:
        """Apply only explicit filters supported by catalog fields."""

        filters = cls._validated_filters(context.filters)
        genres = {
            genre.strip().casefold()
            for genre in str(movie.get("genres", "")).split("|")
        }
        included = {
            genre.casefold()
            for genre in cls._filter_genres(context, "genres_include")
        }
        excluded = {
            genre.casefold()
            for genre in cls._filter_genres(context, "genres_exclude")
        }

        genre_match = filters.get("genre_match")
        if genre_match is None:
            genre_match = (
                context.movie_request.genres_match
                if context.movie_request
                else "any"
            )

        included_genres_match = (
            included.issubset(genres)
            if genre_match == "all"
            else not included.isdisjoint(genres)
        )
        if (
            (included and not included_genres_match)
            or excluded.intersection(genres)
        ):
            return False

        minimum_rating_count = max(
            filters.get("min_rating_count", 0),
            filters.get("minimum_rating_count", 0),
        )
        if minimum_rating_count:
            rating_count = movie.get("rating_count")
            if (
                not isinstance(rating_count, (int, float))
                or isinstance(rating_count, bool)
                or not math.isfinite(rating_count)
                or rating_count < minimum_rating_count
            ):
                return False

        minimum_rating = max(
            filters.get("min_average_rating", 0),
            filters.get("minimum_rating", 0),
        )
        if minimum_rating:
            average_rating = movie.get("average_rating")
            if (
                not isinstance(average_rating, (int, float))
                or isinstance(average_rating, bool)
                or not math.isfinite(average_rating)
                or average_rating < minimum_rating
            ):
                return False

        year_from = filters.get("release_year_from")
        year_to = filters.get("release_year_to")
        request = context.movie_request
        if request and request.release_year_from is not None:
            year_from = max(
                year_from or request.release_year_from,
                request.release_year_from,
            )
        if request and request.release_year_to is not None:
            year_to = min(
                year_to or request.release_year_to,
                request.release_year_to,
            )
        if (
            year_from is not None
            and year_to is not None
            and year_from > year_to
        ):
            return False
        if year_from is not None or year_to is not None:
            title = movie.get("title", "")
            year_match = re.search(r"\((\d{4})\)\s*$", title)
            if not year_match:
                return False

            year = int(year_match.group(1))
            if year_from is not None and year < year_from:
                return False
            if year_to is not None and year > year_to:
                return False

        return not cls._is_explicitly_excluded_title(movie, context)


    @staticmethod
    def _is_explicitly_excluded_title(
        movie: dict,
        context: RecommendationContext,
    ) -> bool:
        """Apply title exclusions only when the exclusion names a title."""

        request = context.movie_request
        if not request:
            return False

        def normalize_title(value: str) -> str:
            value = re.sub(r"\s*\(\d{4}\)\s*$", "", value.strip())
            return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()

        title = normalize_title(str(movie.get("title", "")))
        return any(
            normalize_title(exclusion) == title
            for exclusion in request.exclude
            if exclusion.strip()
        )

    @staticmethod
    def _requires_narrative_uncertainty(
        context: RecommendationContext,
    ) -> bool:
        request = context.movie_request
        return bool(
            request
            and (
                request.narrative_mechanisms
                or request.required_criteria
                or request.preferred_criteria
                or request.examples_interpretation
            )
        )


    def _build_user_message(
        self,
        context: RecommendationContext,
        initial_candidates: list[dict],
    ) -> str:
        """Build the context sent to Qwen."""

        context_data = self._prompt_context_data(context)
        compact_candidates = [
            self._candidate_prompt_record(
                candidate,
                max_plot_characters=800,
            )
            for candidate in self._rank_prompt_candidates(
                initial_candidates
            )
        ][:self.MAX_PROMPT_CANDIDATES]

        payload = {
            "recommendation_context": context_data,
            "initial_catalog_candidates": compact_candidates,
            "catalog_evidence_note": (
                "MovieLens tags are user-contributed clues, not verified "
                "plot descriptions. When present, plot_summary is the "
                "attributed plot section for the exact IMDb-linked title."
            ),
            "instructions": (
                "The recommendation_context contains the user's complete "
                "request. Evaluate only the supplied catalog candidates "
                "against mandatory criteria and preferences. Candidates with "
                "higher request_relevance_score have stronger full-text or "
                "tag evidence for the request. Do not select candidates just "
                "because of a weak, incidental tag match. Python has "
                "already performed the initial searches and verifiable "
                "filters; tool calls are optional for additional data. "
                "When a taste profile is present, use it to personalize "
                "selection without overriding the current request or "
                "explicit filters. Use rated_movie_examples and review "
                "examples only as taste preferences; they are not factual "
                "evidence about candidate plots. Use taste_genre_match only "
                "as a tie-breaker after relevance to the current request. "
                "Use plot_summary as the source for plot information and "
                "the supplied MovieLens tags only as clues. State when "
                "narrative criteria cannot be verified, and do not invent "
                "plot information or claim unsupported criteria are met."
            ),
        }

        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )

    @classmethod
    def _prompt_context_data(
        cls,
        context: RecommendationContext,
    ) -> dict:
        """Omit identifiers and excess history already handled by Python."""

        data = context.model_dump(mode="json")
        taste = data.get("taste_context")
        if not isinstance(taste, dict):
            return data

        for field in (
            "liked_movie_ids",
            "disliked_movie_ids",
            "watched_movie_ids",
        ):
            taste.pop(field, None)

        taste["rated_movie_examples"] = taste.get(
            "rated_movie_examples",
            [],
        )[:6]
        taste["review_examples"] = [
            {
                **review,
                "review": review.get("review", "")[
                    :cls.MAX_TASTE_REVIEW_CHARACTERS
                ],
            }
            for review in taste.get("review_examples", [])[
                :cls.MAX_TASTE_REVIEWS_IN_PROMPT
            ]
            if isinstance(review, dict)
        ]
        return data

    @staticmethod
    def _rank_prompt_candidates(candidates: list[dict]) -> list[dict]:
        return sorted(
            (
                candidate
                for candidate in candidates
                if isinstance(candidate, dict)
            ),
            key=lambda candidate: (
                -float(
                    candidate.get(
                        "request_relevance_score",
                        candidate.get("retrieval_score", 0.0),
                    )
                    or 0.0
                ),
                str(candidate.get("title", "")).casefold(),
            ),
        )

    @classmethod
    def _candidate_prompt_record(
        cls,
        candidate: dict,
        max_plot_characters: int | None = None,
    ) -> dict:
        """Keep prompt evidence compact and omit ranking-only fields."""

        compact = {
            key: value
            for key, value in candidate.items()
            if key in cls.CANDIDATE_PROMPT_FIELDS
            and key != "tags"
        }
        plot_summary = compact.get("plot_summary")
        plot_limit = (
            cls.MAX_PLOT_PROMPT_CHARACTERS
            if max_plot_characters is None
            else max_plot_characters
        )
        if (
            isinstance(plot_summary, str)
            and len(plot_summary) > plot_limit
        ):
            compact["plot_summary"] = (
                plot_summary[:plot_limit].rstrip()
                + "..."
            )
        tags = candidate.get("tags", [])
        if isinstance(tags, list):
            compact["tags"] = [
                {
                    key: tag[key]
                    for key in ("tag", "tag_count")
                    if key in tag
                }
                for tag in tags[:cls.MAX_TAGS_PER_CANDIDATE]
                if isinstance(tag, dict)
            ]
        return compact


    def _execute_tool(
        self,
        tool_name: str,
        arguments: dict,
    ) -> str:
        """Execute one registered catalog tool."""

        aliases = {
            "get_movies_by_tags": {
                "tag": "query",
            },
            "get_movie_details": {
                "movieId": "movie_id",
            },
        }

        normalized_arguments = dict(arguments)

        for old_name, new_name in aliases.get(
            tool_name, {}
        ).items():
            if old_name in normalized_arguments:
                normalized_arguments.setdefault(
                    new_name,
                    normalized_arguments.pop(old_name),
                )

        function = self.tool_functions.get(tool_name)

        if function is None:
            return json.dumps({
                "error": f"Unknown tool: {tool_name}"
            })

        try:
            result = function(**normalized_arguments)
            result = self._to_jsonable(result)

            return json.dumps(
                result,
                ensure_ascii=False,
                default=str,
            )

        except Exception as error:
            logger.exception(
                "Tool execution failed: %s",
                tool_name,
            )

            return json.dumps({
                "error": str(error),
                "tool": tool_name,
            })


    def _generate_final_answer(
        self,
        context: RecommendationContext,
        candidates: dict[int, dict],
    ) -> str:
        """Generate and validate a structured answer against catalog results."""

        if not candidates:
            return (
                "Nu am găsit candidați validați în catalog pentru această "
                "cerere."
            )

        if self.synopsis_tool:
            final_instructions = (
                "Choose only relevant candidates that satisfy the "
                "mandatory criteria as far as the supplied evidence allows. "
                "Candidates are ordered by combined semantic and "
                "catalog-tag relevance; prefer earlier candidates when "
                "their plot evidence supports the request. Separate the "
                "user's personal circumstances from the movie qualities "
                "they ask for; do not recommend a film about a hardship "
                "merely because the user mentioned that hardship. "
                "When a taste profile is supplied, use it as a soft "
                "personalization signal; explicit request criteria and "
                "filters take priority. Rated movie examples and reviews "
                "are taste signals only, not plot evidence. Use "
                "taste_genre_match only as a tie-breaker after request "
                "relevance. "
                "Use a candidate's plot_summary as the only source for plot "
                "details; MovieLens tags are clues, not verified facts. "
                "Use matching_search_terms only as weak retrieval signals; "
                "they do not prove the movie's tone or plot. "
                "For each recommendation, write a concise description in "
                "the user's language using one or two short sentences: "
                "briefly state the premise and connect it to the request "
                "only when the synopsis supports that connection. Avoid "
                "spoilers. Do not invent plot details or repeat long source "
                "passages. Return only movieId and description fields. "
                "Return no recommendations if none is sufficiently relevant."
            )
        else:
            final_instructions = (
                "Choose only relevant candidates that satisfy the "
                "mandatory criteria as far as the catalog evidence "
                "allows. Treat MovieLens tags as user-contributed clues, "
                "not verified plot facts. Do not invent titles, IDs, "
                "years, genres, or narrative facts. Do not write reasons "
                "or plot summaries; return only candidate movieId values. "
                "Do not rank by rating alone. Return no recommendations "
                "if none is sufficiently relevant."
            )

        prompt_candidates = self._rank_prompt_candidates(
            list(candidates.values())
        )[:self.MAX_PROMPT_CANDIDATES]
        prompt_candidate_map = {
            movie_id: candidate
            for candidate in prompt_candidates
            if (movie_id := self._get_movie_id(candidate)) is not None
        }
        prompt = json.dumps(
            {
                "recommendation_context": self._prompt_context_data(context),
                "validated_catalog_candidates": [
                    self._final_candidate_prompt_record(candidate)
                    for candidate in prompt_candidates
                ],
                "instructions": final_instructions,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

        try:
            parsed = self.llm.generate_structured(
                prompt,
                MovieRecommendationResponse,
            )
            recommendations = (
                parsed.get("recommendations")
                if isinstance(parsed, dict)
                else None
            )
            if not isinstance(recommendations, list):
                return "Nu am putut valida răspunsul final al modelului."

            return self._render_validated_recommendations(
                recommendations,
                prompt_candidate_map,
                context,
            )

        except Exception:
            logger.exception(
                "Failed to generate the final recommendation."
            )

            return "Nu am putut valida recomandările generate."


    def _render_validated_recommendations(
        self,
        recommendations: list[dict],
        candidates: dict[int, dict],
        context: RecommendationContext,
    ) -> str:
        """Render only candidate IDs that still satisfy verifiable filters."""

        valid = []
        seen_ids = set()

        for recommendation in recommendations:
            if not isinstance(recommendation, dict):
                continue

            movie_id = recommendation.get("movieId")
            if (
                not isinstance(movie_id, int)
                or isinstance(movie_id, bool)
                or movie_id not in candidates
                or movie_id in seen_ids
            ):
                continue

            movie = candidates[movie_id]
            if not self._matches_explicit_filters(movie, context):
                continue

            taste = context.taste_context
            if taste and movie_id in taste.watched_movie_ids:
                continue

            valid.append((movie, recommendation))
            seen_ids.add(movie_id)

        if not valid:
            return (
                "Nu am putut valida nicio recomandare care să respecte "
                "filtrele și criteriile verificabile."
            )

        if self.synopsis_tool is not None:
            return self._render_sourced_recommendations(
                valid,
                context,
            )

        rendered = []
        for movie, _ in valid:
            genres = movie.get("genres") or "Nedisponibil"
            details = [
                f"**{movie['title']}**",
                f"- Genuri din catalog: {genres}.",
            ]

            if movie.get("average_rating") is not None:
                details.append(
                    f"- Rating mediu: {movie['average_rating']} "
                    f"({movie.get('rating_count', 0)} evaluări)."
                )

            tags = movie.get("tags", [])
            if tags:
                tag_names = [
                    tag["tag"]
                    for tag in tags
                    if isinstance(tag, dict)
                    and isinstance(tag.get("tag"), str)
                ]
                if tag_names:
                    details.append(
                        "- Tag-uri MovieLens (indicii furnizate de "
                        "utilizatori): "
                        + ", ".join(tag_names)
                        + "."
                    )

            if self._requires_narrative_uncertainty(context):
                details.append(
                    "- Limitare: catalogul nu conține sinopsisuri, iar "
                    "tag-urile MovieLens sunt indicii ale utilizatorilor. "
                    "Nu pot confirma criteriile narative sau de atmosferă "
                    "din aceste date."
                )

            rendered.append("\n".join(details))

        response = "\n\n".join(rendered)
        if self._requires_narrative_uncertainty(context):
            return (
                "Candidați posibili; potrivirea narativă nu este "
                "confirmată de catalog:\n\n"
                + response
            )

        return response


    @classmethod
    def _final_candidate_prompt_record(cls, candidate: dict) -> dict:
        """Send only the evidence needed to select and describe a movie."""

        compact = {
            key: candidate[key]
            for key in ("movieId", "title", "genres")
            if key in candidate
        }
        plot_summary = candidate.get("plot_summary")
        if isinstance(plot_summary, str):
            compact["plot_summary"] = plot_summary[
                :cls.MAX_FINAL_PLOT_PROMPT_CHARACTERS
            ].rstrip()
            if len(plot_summary) > cls.MAX_FINAL_PLOT_PROMPT_CHARACTERS:
                compact["plot_summary"] += "..."
        matched_queries = candidate.get("matched_queries")
        if isinstance(matched_queries, list):
            compact["matching_search_terms"] = [
                query
                for query in matched_queries
                if isinstance(query, str)
            ]
        return compact

    @staticmethod
    def _limit_to_two_sentences(text: Any) -> str:
        if not isinstance(text, str):
            return ""

        normalized = re.sub(r"\s+", " ", text.strip())
        if not normalized:
            return ""

        sentence_ends = [
            match.end()
            for match in re.finditer(r"[.!?](?=\s|$)", normalized)
        ]
        if len(sentence_ends) > 2:
            normalized = normalized[:sentence_ends[1]]

        if len(normalized) > 400:
            normalized = normalized[:400].rsplit(" ", 1)[0].rstrip(" ,;:-")
            if not normalized.endswith((".", "!", "?", "…")):
                normalized += "…"

        return normalized


    def _render_sourced_recommendations(
        self,
        recommendations: list[tuple[dict, dict]],
        context: RecommendationContext,
    ) -> str:
        rendered = []
        for movie, recommendation in recommendations:
            source = movie.get("narrative_source")
            summary = movie.get("plot_summary")
            if (
                not isinstance(source, dict)
                or not isinstance(summary, str)
                or not summary.strip()
            ):
                if self._requires_narrative_uncertainty(context):
                    continue
                rendered.append(
                    f"**{movie['title']}** — Nu este disponibil un "
                    "sinopsis narativ licențiat și verificabil."
                )
                continue

            description = self._limit_to_two_sentences(
                recommendation.get("description")
            )
            if not description:
                continue

            source_title = re.sub(
                r"[\[\]]",
                "",
                str(source.get("title", "Wikipedia")),
            )
            revision_url = source.get("revision_url")
            license_url = source.get("license_url")
            if not isinstance(revision_url, str) or not isinstance(
                license_url,
                str,
            ):
                continue

            rendered.append(
                f"**{movie['title']}** — {description}\n"
                f"*Sursă: [Wikipedia: {source_title}]({revision_url}); "
                f"text adaptat sub "
                f"[CC BY-SA 4.0]({license_url}).*"
            )

        if not rendered:
            return (
                "Nu am găsit recomandări cu sinopsis verificabil "
                "și descriere validată."
            )

        return "\n\n".join(rendered)


    def run(
        self,
        context: RecommendationContext,
    ) -> str:
        """Run one independent movie recommendation request."""

        request = context.movie_request
        if request and request.content_avoidances:
            avoided = ", ".join(request.content_avoidances)
            return (
                "Nu pot recomanda filme în siguranță pentru această cerere: "
                f"catalogul actual nu conține avertismente de conținut "
                f"verificate pentru a confirma absența elementelor evitate "
                f"({avoided})."
            )

        try:
            initial_candidates = self._initial_retrieval(context)
            retrieved_candidate_count = len(initial_candidates)
            if self.synopsis_tool is not None:
                initial_candidates = self.synopsis_tool.enrich_candidates(
                    initial_candidates
                )

        except Exception:
            logger.exception(
                "Initial catalog or synopsis retrieval failed."
            )

            raise

        if (
            self.synopsis_tool is not None
            and self._requires_narrative_uncertainty(context)
        ):
            initial_candidates = [
                candidate
                for candidate in initial_candidates
                if (
                    isinstance(candidate.get("plot_summary"), str)
                    and candidate["plot_summary"].strip()
                    and isinstance(candidate.get("narrative_source"), dict)
                )
            ]

        if not initial_candidates:
            if not retrieved_candidate_count:
                return (
                    "Nu am găsit candidați potriviți în catalog pentru această "
                    "cerere. Catalogul nu conține seriale, disponibilitate "
                    "pe platforme sau informații verificate despre subtitrări."
                )
            if (
                self.synopsis_tool is not None
                and self._requires_narrative_uncertainty(context)
            ):
                return (
                    "Nu am găsit candidați cu un sinopsis narativ "
                    "verificabil și licențiat pentru această cerere."
                )
            return (
                "Nu am găsit candidați care să respecte filtrele "
                "verificabile din catalog."
            )

        messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": self._build_user_message(
                    context,
                    initial_candidates,
                ),
            },
        ]

        valid_candidates = {}
        for candidate in initial_candidates:
            movie_id = self._get_movie_id(candidate)
            if movie_id is not None:
                valid_candidates[movie_id] = dict(candidate)
        successful_tool_calls = 0

        available_narrative_movie_ids = (
            self._available_narrative_movie_ids(context)
        )
        tool_call_budget = (
            self.MAX_TOOL_CALLS
            if (
                len(valid_candidates) < self.INITIAL_CANDIDATE_LIMIT
                and available_narrative_movie_ids is None
            )
            else 0
        )
        for _ in range(tool_call_budget):
            try:
                response = self.llm.chat_with_tools(
                    messages=messages,
                    tools=self.tools,
                )

            except Exception:
                logger.exception(
                    "LLM tool-calling request failed."
                )

                break

            assistant_message = response.message

            tool_calls = assistant_message.tool_calls or []

            if not tool_calls:
                break

            messages.append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [
                    call.model_dump()
                    if hasattr(call, "model_dump")
                    else call
                    for call in tool_calls
                ],
            })

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                arguments = tool_call.function.arguments or {}

                if not isinstance(arguments, dict):
                    try:
                        arguments = json.loads(arguments)
                    except (TypeError, ValueError):
                        arguments = {}

                result = self._execute_tool(
                    function_name,
                    arguments,
                )

                try:
                    tool_result = json.loads(result)
                except json.JSONDecodeError:
                    logger.error(
                        "Tool %s returned invalid JSON.",
                        function_name,
                    )
                    tool_result = {"error": "Invalid JSON tool result."}

                if isinstance(tool_result, dict) and "error" in tool_result:
                    logger.warning(
                        "Tool %s failed: %s",
                        function_name,
                        tool_result["error"],
                    )
                else:
                    successful_tool_calls += 1
                    is_list_result = isinstance(tool_result, list)
                    tool_movies = (
                        tool_result
                        if is_list_result
                        else [tool_result]
                        if isinstance(tool_result, dict)
                        and self._get_movie_id(tool_result) is not None
                        else []
                    )
                    accepted_tool_movies = []

                    for tool_movie in tool_movies:
                        if not isinstance(tool_movie, dict):
                            continue

                        movie_id = self._get_movie_id(tool_movie)
                        if (
                            movie_id is None
                            or not isinstance(tool_movie.get("title"), str)
                        ):
                            continue

                        if (
                            movie_id not in valid_candidates
                            and len(valid_candidates)
                            >= self.INITIAL_CANDIDATE_LIMIT
                        ):
                            continue

                        candidate = {
                            **valid_candidates.get(movie_id, {}),
                            **tool_movie,
                        }
                        if not self._matches_explicit_filters(
                            candidate,
                            context,
                        ):
                            continue

                        taste = context.taste_context
                        if taste and movie_id in taste.watched_movie_ids:
                            continue

                        if function_name == "get_movies_by_tags":
                            query = arguments.get("query") or arguments.get("tag")
                            if isinstance(query, str):
                                matched_queries = candidate.setdefault(
                                    "matched_queries",
                                    [],
                                )
                                if query not in matched_queries:
                                    matched_queries.append(query)
                                self._attach_candidate_tags([candidate])

                        if isinstance(candidate.get("tags"), list):
                            candidate["tags"] = candidate["tags"][
                                :self.MAX_TAGS_PER_CANDIDATE
                            ]

                        valid_candidates[movie_id] = candidate
                        accepted_tool_movies.append(
                            self._candidate_prompt_record(candidate)
                        )

                    if is_list_result:
                        tool_result = accepted_tool_movies
                    elif accepted_tool_movies:
                        tool_result = accepted_tool_movies[0]
                    elif tool_movies:
                        tool_result = None

                    result = json.dumps(
                        tool_result,
                        ensure_ascii=False,
                        default=str,
                    )

                messages.append({
                    "role": "tool",
                    "tool_name": function_name,
                    "content": result,
                })

        if self.synopsis_tool is not None:
            try:
                enriched_candidates = self.synopsis_tool.enrich_candidates(
                    list(valid_candidates.values())
                )
            except Exception:
                logger.exception(
                    "Final candidate synopsis retrieval failed."
                )
                raise

            valid_candidates = {
                movie_id: candidate
                for candidate in enriched_candidates
                if (movie_id := self._get_movie_id(candidate)) is not None
            }

            if self._requires_narrative_uncertainty(context):
                valid_candidates = {
                    movie_id: candidate
                    for movie_id, candidate in valid_candidates.items()
                    if (
                        isinstance(candidate.get("plot_summary"), str)
                        and candidate["plot_summary"].strip()
                        and isinstance(candidate.get("narrative_source"), dict)
                    )
                }

        logger.info(
            "Recommendation completed. Initial candidates: %s, "
            "successful additional tool calls: %s",
            len(initial_candidates),
            successful_tool_calls,
        )

        return self._generate_final_answer(
            context,
            valid_candidates,
        )
