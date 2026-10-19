
from pathlib import Path

from movieshelf.data_loader import load_movies, load_ratings
from movieshelf.data_processing import build_movie_statistics
from movieshelf.ranking import calculate_weighted_rating


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ml-32m"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "movie_catalog.csv"
RANKING_OUTPUT_FILE = OUTPUT_DIR / "movie_catalog_ranked.csv"


def main() -> None:
    """Build and save the movie catalog."""

    print("Loading movies...")
    movies = load_movies(DATA_DIR)
    print(f"Loaded {len(movies):,} movies.")

    print("Loading ratings...")
    ratings = load_ratings(DATA_DIR)
    print(f"Loaded {len(ratings):,} ratings.")

    print("Building movie statistics...")
    catalog = build_movie_statistics(movies, ratings)
    print(f"Built catalog with {len(catalog):,} movies.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    catalog.to_csv(OUTPUT_FILE, index=False)
    print(f"Base catalog saved to: {OUTPUT_FILE}")

    print("Calculating global mean rating...")
    global_mean = ratings["rating"].mean()
    print(f"Global mean rating: {global_mean:.4f}")

    print("Calculating weighted movie ratings...")
    ranked_catalog = calculate_weighted_rating(
        catalog,
        global_mean=global_mean,
        minimum_votes=1_000,
    )

    ranked_catalog.to_csv(RANKING_OUTPUT_FILE, index=False)
    print(f"Ranked catalog saved to: {RANKING_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
