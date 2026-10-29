
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticMovieSearchEngine:
    """Rank movies by semantic relevance, rating confidence, and quality."""

    def __init__(
        self,
        movie_profiles: pd.DataFrame,
        movie_catalog: pd.DataFrame | None = None,
        model_name: str = "intfloat/multilingual-e5-small",
        global_mean: float = 3.5404,
        semantic_weight: float = 0.80,
        confidence_votes: int = 1_000,
    ) -> None:
        if not 0 <= semantic_weight <= 1:
            raise ValueError("semantic_weight must be between 0 and 1.")

        if not 0.5 <= global_mean <= 5:
            raise ValueError("global_mean must be between 0.5 and 5.")

        if confidence_votes <= 0:
            raise ValueError("confidence_votes must be positive.")

        required = {"movieId", "profile_text"}
        missing = required - set(movie_profiles.columns)

        if missing:
            raise ValueError(
                f"Missing profile columns: {sorted(missing)}"
            )

        profiles = movie_profiles.copy()
        profiles = profiles.dropna(subset=["profile_text"])
        profiles["profile_text"] = (
            profiles["profile_text"].astype(str).str.strip()
        )
        profiles = profiles[profiles["profile_text"] != ""]

        if profiles.empty:
            raise ValueError(
                "At least one non-empty movie profile is required."
            )

        if profiles["movieId"].duplicated().any():
            raise ValueError(
                "movie_profiles contains duplicate movieId values."
            )

        self.has_rating_data = movie_catalog is not None
        self.global_mean = global_mean
        self.semantic_weight = semantic_weight
        self.confidence_votes = confidence_votes

        self.has_genre_data = False

        if movie_catalog is not None:
            rating_columns = [
                "movieId",
                "rating_count",
                "average_rating",
                "weighted_rating",
            ]
            missing = set(rating_columns) - set(movie_catalog.columns)

            if missing:
                raise ValueError(
                    f"Missing catalog columns: {sorted(missing)}"
                )

            # Include genres and title when they are available.
            optional_columns = [
                column
                for column in ["title", "genres"]
                if column in movie_catalog.columns
            ]

            catalog_columns = rating_columns + [
                column
                for column in optional_columns
                if column not in rating_columns
            ]

            catalog = movie_catalog[catalog_columns].copy()

            if catalog["movieId"].duplicated().any():
                raise ValueError(
                    "movie_catalog contains duplicate movieId values."
                )

            self.has_genre_data = "genres" in catalog.columns

            profiles = profiles.merge(
                catalog,
                on="movieId",
                how="left",
                validate="one_to_one",
            )

        self.profiles = profiles.reset_index(drop=True)
        self.model = SentenceTransformer(model_name)

        passages = [
            f"passage: {profile}"
            for profile in self.profiles["profile_text"]
        ]

        self.profile_embeddings = self.model.encode(
            passages,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    def search(
        self,
        query: str,
        n: int = 10,
        exclude_genres: list[str] | None = None,
        include_genres: list[str] | None = None,
        min_rating_count: int = 0,
        min_average_rating: float | None = None,
    ) -> pd.DataFrame:
        """Search semantically, with optional rating and genre filters."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise ValueError("n must be a positive integer.")

        if (
            not isinstance(min_rating_count, int)
            or isinstance(min_rating_count, bool)
            or min_rating_count < 0
        ):
            raise ValueError("min_rating_count must be a non-negative integer.")

        if min_average_rating is not None:
            if not 0.5 <= min_average_rating <= 5:
                raise ValueError(
                    "min_average_rating must be between 0.5 and 5."
                )

        exclude_genres = exclude_genres or []
        include_genres = include_genres or []

        if (exclude_genres or include_genres) and not self.has_genre_data:
            raise ValueError(
                "Genre filters require a 'genres' column in movie_catalog."
            )

        query_embedding = self.model.encode(
            [f"query: {query.strip()}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        scores = cosine_similarity(
            query_embedding,
            self.profile_embeddings,
        ).ravel()

        results = self.profiles.copy()
        results["semantic_score"] = scores

        # Remove candidates with no positive semantic similarity.
        results = results[results["semantic_score"] > 0].copy()

        # Genre filters are hard constraints, not ranking preferences.
        if exclude_genres:
            excluded = {
                genre.strip().casefold()
                for genre in exclude_genres
                if genre.strip()
            }

            def has_excluded_genre(value: object) -> bool:
                if not isinstance(value, str):
                    return False

                movie_genres = {
                    genre.strip().casefold()
                    for genre in value.split("|")
                }
                return bool(movie_genres & excluded)

            results = results[
                ~results["genres"].apply(has_excluded_genre)
            ].copy()

        if include_genres:
            included = {
                genre.strip().casefold()
                for genre in include_genres
                if genre.strip()
            }

            def has_included_genre(value: object) -> bool:
                if not isinstance(value, str):
                    return False

                movie_genres = {
                    genre.strip().casefold()
                    for genre in value.split("|")
                }
                return bool(movie_genres & included)

            results = results[
                results["genres"].apply(has_included_genre)
            ].copy()

        if self.has_rating_data:
            # Missing rating counts are treated as zero.
            rating_count = results["rating_count"].fillna(0)
            average_rating = results["average_rating"]

            if min_rating_count > 0:
                results = results[
                    rating_count >= min_rating_count
                ].copy()

                rating_count = results["rating_count"].fillna(0)
                average_rating = results["average_rating"]

            if min_average_rating is not None:
                results = results[
                    average_rating >= min_average_rating
                ].copy()

                rating_count = results["rating_count"].fillna(0)
                average_rating = results["average_rating"]

            # Shrink unreliable averages toward the global mean.
            # This reduces the influence of films with very few votes.
            confidence = (
                rating_count
                / (rating_count + self.confidence_votes)
            )

            results["rating_confidence"] = confidence

            adjusted_rating = (
                confidence * average_rating.fillna(self.global_mean)
                + (1 - confidence) * self.global_mean
            )

            results["quality_score"] = (
                adjusted_rating / 5
            ).clip(0, 1)

            results["final_score"] = (
                self.semantic_weight * results["semantic_score"]
                + (1 - self.semantic_weight) * results["quality_score"]
            )
        else:
            results["final_score"] = results["semantic_score"]

        return (
            results.sort_values(
                by=["final_score", "semantic_score", "movieId"],
                ascending=[False, False, True],
            )
            .head(n)
            .reset_index(drop=True)
        )
