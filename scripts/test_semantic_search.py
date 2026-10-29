
from pathlib import Path
import re

import pandas as pd

from movieshelf.semantic_search import SemanticMovieSearchEngine


ROOT = Path(__file__).resolve().parents[1]

PROFILES_PATH = ROOT / "data" / "processed" / "movie_profiles.csv"
RANKED_CATALOG_PATH = (
    ROOT / "data" / "processed" / "movie_catalog_ranked.csv"
)

SAMPLE_SIZE = 500
RESULTS_PER_QUERY = 5


QUERIES = [
    (
        "I need movie recommendations for a high school film analysis "
        "class. I need films that portray human struggles, challenge "
        "existing culture, tackle current events, create icons and stars, "
        "and broke records. I want critically acclaimed films and at least "
        "one movie that is more than 20 years old."
    ),
    "Critically acclaimed films that portray human struggles",
    "Films that challenge existing culture and social norms",
    "Films that tackle important contemporary issues and current events",
    "Movies that created iconic characters and made actors into stars",
    "Movies that broke box office records",
    (
        "I need a good apocalyptic or doomsday film with an eerie, "
        "unsettling atmosphere that makes me feel scared, but it should "
        "not be a horror movie."
    ),
    "Apocalyptic films about the end of the world with an eerie atmosphere",
    "Non-horror films about a looming apocalypse that create suspense and dread",
]


REFERENCE_TITLES = [
    "The Hurt Locker",
    "Small Soldiers",
    "Black Hawk Down",
    "Alien: Covenant",
    "American Psycho",
    "Before I Go to Sleep",
    "Saving Private Ryan",
    "Finding Nemo",
]


def load_test_profiles() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load profiles and catalog, including reference films."""

    if not PROFILES_PATH.exists():
        raise FileNotFoundError(
            f"Movie profiles not found: {PROFILES_PATH}"
        )

    if not RANKED_CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Ranked movie catalog not found: {RANKED_CATALOG_PATH}"
        )

    profiles = pd.read_csv(PROFILES_PATH)
    catalog = pd.read_csv(RANKED_CATALOG_PATH)

    required_profile_columns = {"movieId", "profile_text"}
    missing_profiles = required_profile_columns - set(profiles.columns)

    if missing_profiles:
        raise ValueError(
            f"Missing profile columns: {sorted(missing_profiles)}"
        )

    required_catalog_columns = {
        "movieId",
        "title",
        "genres",
        "rating_count",
        "average_rating",
        "weighted_rating",
    }
    missing_catalog = required_catalog_columns - set(catalog.columns)

    if missing_catalog:
        raise ValueError(
            f"Missing ranked catalog columns: {sorted(missing_catalog)}"
        )

    sample = profiles.sample(
        n=min(SAMPLE_SIZE, len(profiles)),
        random_state=42,
    )

    # Locate reference titles in profile text.
    reference_pattern = "|".join(
        re.escape(title) for title in REFERENCE_TITLES
    )

    reference_mask = profiles["profile_text"].str.contains(
        reference_pattern,
        case=False,
        regex=True,
        na=False,
    )

    profiles_to_test = (
        pd.concat([sample, profiles[reference_mask]])
        .drop_duplicates(subset=["movieId"])
        .reset_index(drop=True)
    )

    return profiles_to_test, catalog


def print_results(
    results: pd.DataFrame,
    query: str,
) -> None:
    """Print ranked recommendations and their scoring details."""

    print(f"\n{'=' * 100}")
    print(f"CERERE: {query}\n")

    if results.empty:
        print("Nu au fost găsite rezultate pentru aceste criterii.")
        return

    for _, movie in results.iterrows():
        title = movie.get("title")

        if pd.isna(title) or not title:
            title = movie["profile_text"].split(". Genres:")[0]

        average_rating = movie.get("average_rating", float("nan"))
        rating_count = movie.get("rating_count", float("nan"))
        weighted_rating = movie.get("weighted_rating", float("nan"))
        confidence = movie.get("rating_confidence", float("nan"))
        genres = movie.get("genres", "N/A")

        print(
            f"Final: {movie['final_score']:.4f} | "
            f"Semantic: {movie['semantic_score']:.4f} | "
            f"Rating: {average_rating:.2f} | "
            f"Voturi: {rating_count:,.0f} | "
            f"Rating ponderat inițial: {weighted_rating:.3f} | "
            f"Încredere rating: {confidence:.1%} | "
            f"Genuri: {genres} | "
            f"{title}"
        )


def main() -> None:
    profiles, catalog = load_test_profiles()

    print(f"Filme în catalogul de profiluri: {len(pd.read_csv(PROFILES_PATH)):,}")
    print(f"Filme în catalogul cu ratinguri: {len(catalog):,}")
    print(f"Filme pentru test: {len(profiles):,}")
    print("Generez embeddings pentru eșantion...")

    engine = SemanticMovieSearchEngine(
        profiles,
        movie_catalog=catalog,
    )

    # Run ordinary semantic searches.
    for query in QUERIES:
        results = engine.search(
            query,
            n=RESULTS_PER_QUERY,
        )
        print_results(results, query)

    # Run a separate test with a hard genre exclusion.
    apocalypse_query = (
        "Apocalyptic films with an eerie, unsettling atmosphere "
        "that create suspense and dread."
    )

    try:
        filtered_results = engine.search(
            apocalypse_query,
            n=RESULTS_PER_QUERY,
            exclude_genres=["Horror"],
        )
        print_results(
            filtered_results,
            apocalypse_query + " [Fără genul Horror]",
        )
    except ValueError as error:
        print(f"\nFiltrul de gen nu a putut fi aplicat: {error}")


if __name__ == "__main__":
    main()
