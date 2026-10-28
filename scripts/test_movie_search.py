
from pathlib import Path

import pandas as pd

from movieshelf.search import MovieSearchEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILES_FILE = (
    PROJECT_ROOT / "data" / "processed" / "movie_profiles.csv"
)


def main() -> None:
    print("Loading movie profiles...")

    profiles = pd.read_csv(PROFILES_FILE)

    print(f"Loaded {len(profiles):,} movie profiles.")

    print("Building search engine...")
    engine = MovieSearchEngine(profiles)

    queries = [
        "Iraq War bomb defusal realistic action",
        "military soldiers rescue mission",
        "space exploration and alien life",
        "psychological thriller with an unreliable narrator",
    ]

    for query in queries:
        print(f"\nQuery: {query}")

        results = engine.search(query, n=5)

        if results.empty:
            print("No matching movies found.")
            continue

       
        for _, movie in results.iterrows():
            print(
                f"- movieId={movie['movieId']} | "
                f"similarity={movie['similarity_score']:.4f} | "
                f"coverage={movie['concept_coverage']:.2f} | "
                f"{movie['profile_text']}"
            )



if __name__ == "__main__":
    main()
