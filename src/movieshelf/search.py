
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieSearchEngine:
    """Search movie profiles using TF-IDF and cosine similarity."""

    def __init__(self, movie_profiles: pd.DataFrame) -> None:
        required_columns = {"movieId", "profile_text"}
        missing_columns = required_columns - set(movie_profiles.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        profiles = movie_profiles[["movieId", "profile_text"]].copy()

        profiles = profiles.dropna(subset=["profile_text"])
        profiles["profile_text"] = profiles["profile_text"].astype(str).str.strip()
        profiles = profiles[profiles["profile_text"] != ""]

        if profiles.empty:
            raise ValueError("At least one non-empty movie profile is required.")

        self.profiles = profiles.reset_index(drop=True)

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.profile_matrix = self.vectorizer.fit_transform(
            self.profiles["profile_text"]
        )

    def search(self, query: str, n: int = 10) -> pd.DataFrame:
        """Return the top matching movies, ordered by similarity."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise ValueError("n must be a positive integer.")

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(
            query_vector,
            self.profile_matrix,
        ).ravel()

        results = self.profiles.copy()
        results["similarity_score"] = scores

        results = results[results["similarity_score"] > 0]

        results = results.sort_values(
            by=["similarity_score", "movieId"],
            ascending=[False, True],
        )

        return results.head(n).reset_index(drop=True)
