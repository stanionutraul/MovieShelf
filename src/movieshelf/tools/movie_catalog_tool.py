
import pandas as pd


class MovieCatalogTool:
    """Provides deterministic access to the MovieShelf movie catalog."""

    def __init__(self, catalog_path, tags_path):
        self.movies = pd.read_csv(catalog_path)
        self.tags = pd.read_csv(tags_path)

        self.movies["title_lower"] = (
            self.movies["title"].fillna("").str.lower()
        )
        self.tags["tag_lower"] = (
            self.tags["tag"].fillna("").str.lower()
        )

    @staticmethod
    def _records(dataframe):
        return dataframe.drop(
            columns=["title_lower", "tag_lower"],
            errors="ignore",
        ).to_dict(orient="records")

    def search_movies(self, query, limit=10):
        """Search movies by title."""
        if not query or limit < 1:
            return []

        results = self.movies[
            self.movies["title_lower"].str.contains(
                query.lower(), regex=False
            )
        ]

        return self._records(results.head(limit))

    def filter_movies(
        self,
        include_genres=None,
        exclude_genres=None,
        min_rating_count=0,
        min_average_rating=None,
        limit=20,
    ):
        """Filter movies by genres and rating statistics."""
        results = self.movies.copy()

        for genre in include_genres or []:
            results = results[
                results["genres"].str.contains(
                    genre, case=False, regex=False, na=False
                )
            ]

        for genre in exclude_genres or []:
            results = results[
                ~results["genres"].str.contains(
                    genre, case=False, regex=False, na=False
                )
            ]

        results = results[
            results["rating_count"] >= min_rating_count
        ]

        if min_average_rating is not None:
            results = results[
                results["average_rating"] >= min_average_rating
            ]

        results = results.sort_values(
            "weighted_rating", ascending=False
        )

        return self._records(results.head(limit))

    
    def get_movie_details(self, movie_id):
        """Get catalog information and MovieLens tags for a movie."""

        results = self.movies[
            self.movies["movieId"] == movie_id
        ]

        if results.empty:
            return None

        movie = self._records(results)[0]

        movie_tags = self.tags[
            self.tags["movieId"] == movie_id
        ].sort_values(
            "tag_count",
            ascending=False,
        )

        movie["tags"] = self._records(movie_tags)

        return movie


    def get_random_movies(self, limit=10):
        """Return random movies from the catalog."""
        results = self.movies.sample(
            n=min(limit, len(self.movies))
        )

        return self._records(results)

   
    def get_movies_by_tags(self, query=None, limit=20, tag=None):
        """Find movies using MovieLens tags."""

        # Accept both argument names because the LLM may send either one.
        query = query or tag

        if not query or limit < 1:
            return []

        matching_tags = self.tags[
            self.tags["tag_lower"].str.contains(
                query.lower(), regex=False
            )
        ]

        movie_ids = (
            matching_tags
            .groupby("movieId")["tag_count"]
            .sum()
            .sort_values(ascending=False)
            .head(limit)
            .index
        )

        results = self.movies[
            self.movies["movieId"].isin(movie_ids)
        ].copy()

        tag_counts = (
            matching_tags.groupby("movieId")["tag_count"].sum()
        )

        results["matching_tag_count"] = (
            results["movieId"].map(tag_counts)
        )

        results = results.sort_values(
            "matching_tag_count", ascending=False
        )

        return self._records(results)

