import argparse
import csv
import io
import re
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from movieshelf.schemas.user_taste import (
    GenreTasteSummary,
    TasteMovieRating,
    TasteReviewExample,
    UserTasteContext,
)


_EXPORT_FILES = {
    "watched.csv": "watched",
    "diary.csv": "diary",
    "ratings.csv": "ratings",
    "reviews.csv": "reviews",
}
_REVIEW_LIMIT = 400
_MAX_REVIEW_EXAMPLES = 6
_MAX_RATED_MOVIE_EXAMPLES_PER_POLARITY = 3


def _normalized_title_variants(value: str) -> set[str]:
    normalized = _normalized_title(value)
    variants = {normalized} if normalized else set()
    trailing_article = re.fullmatch(r"(.+),\s*(the|a|an)", value.strip(), re.I)
    if trailing_article:
        reordered = f"{trailing_article.group(2)} {trailing_article.group(1)}"
        variants.add(_normalized_title(reordered))
    else:
        leading_article = re.fullmatch(r"(the|a|an)\s+(.+)", value.strip(), re.I)
        if leading_article:
            reordered = f"{leading_article.group(2)}, {leading_article.group(1)}"
            variants.add(_normalized_title(reordered))
    return variants


def _catalog_title_variants(value: str) -> set[str]:
    title_match = re.match(r"^(.*?)\s*\((\d{4})\)\s*$", value)
    title_name = title_match.group(1) if title_match else value
    primary_title = re.split(r"\s+\(", title_name, maxsplit=1)[0].strip()
    variants = _normalized_title_variants(primary_title)

    for parenthetical in re.findall(r"\(([^()]*)\)", title_name):
        alias_match = re.match(
            r"^(?:a\.?\s*k\.?\s*a\.?|aka)\s*(.+)$",
            parenthetical.strip(),
            re.IGNORECASE,
        )
        if alias_match:
            variants.update(
                _normalized_title_variants(alias_match.group(1).strip())
            )

    return variants


class LetterboxdImportResult(BaseModel):
    """Matched viewing history and compact taste profile from an export."""

    taste_context: UserTasteContext
    imported_entries: int
    matched_entries: int
    unmatched_titles: list[str] = Field(default_factory=list)


def _normalized_title(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        character
        for character in value
        if not unicodedata.combining(character)
    )
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _year(value: Any) -> int | None:
    if value is None:
        return None

    match = re.search(r"\b(\d{4})\b", str(value))
    return int(match.group(1)) if match else None


def _rating(value: Any, source: str, row_number: int) -> float | None:
    if value is None or not str(value).strip():
        return None

    try:
        rating = float(str(value).strip())
    except ValueError as error:
        raise ValueError(
            f"Invalid Letterboxd rating in {source}, row {row_number}."
        ) from error

    if not 0 <= rating <= 5:
        raise ValueError(
            f"Letterboxd rating in {source}, row {row_number} "
            "must be between 0 and 5."
        )

    return rating


def _row_value(row: dict[str, str], *names: str) -> str:
    values = {
        key.strip().casefold(): value.strip()
        for key, value in row.items()
        if key and value
    }
    for name in names:
        value = values.get(name.casefold())
        if value:
            return value
    return ""


def _entry_key(uri: str, name: str, year: int | None) -> tuple[str, ...]:
    if uri:
        return ("uri", uri.rstrip("/").casefold())
    return ("title", _normalized_title(name), str(year or ""))


def _catalog_lookup(
    catalog: pd.DataFrame,
) -> tuple[dict[tuple[str, int | None], list[int]], dict[str, list[int]]]:
    required = {"movieId", "title"}
    missing = required - set(catalog.columns)
    if missing:
        raise ValueError(
            "Movie catalog is missing required columns: "
            + ", ".join(sorted(missing))
        )

    by_title_year: dict[tuple[str, int | None], list[int]] = defaultdict(list)
    by_title: dict[str, list[int]] = defaultdict(list)
    for record in catalog[["movieId", "title"]].to_dict(orient="records"):
        movie_id = record["movieId"]
        title = record["title"]
        if pd.isna(movie_id) or not isinstance(title, str):
            continue

        try:
            movie_id = int(movie_id)
        except (TypeError, ValueError):
            continue

        title_match = re.match(r"^(.*?)\s*\((\d{4})\)\s*$", title)
        title_year = int(title_match.group(2)) if title_match else None
        for normalized in _catalog_title_variants(title):
            if not normalized:
                continue
            if movie_id not in by_title[normalized]:
                by_title[normalized].append(movie_id)
            matches = by_title_year[(normalized, title_year)]
            if movie_id not in matches:
                matches.append(movie_id)

    return by_title_year, by_title


def _match_movie(
    name: str,
    year: int | None,
    by_title_year: dict[tuple[str, int | None], list[int]],
    by_title: dict[str, list[int]],
) -> int | None:
    matches = set()
    for normalized in _normalized_title_variants(name):
        if year is None:
            matches.update(by_title.get(normalized, []))
        else:
            matches.update(by_title_year.get((normalized, year), []))
            matches.update(by_title_year.get((normalized, None), []))
    return next(iter(matches)) if len(matches) == 1 else None



def _read_export_files(source: Path) -> dict[str, list[dict[str, str]]]:
    files: dict[str, bytes] = {}

    if source.is_dir():
        for filename in _EXPORT_FILES:
            candidates = [
                path
                for path in source.rglob(filename)
                if path.is_file()
                and path.parent == source
            ]

            if candidates:
                files[filename] = candidates[0].read_bytes()

    elif source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for filename in _EXPORT_FILES:
                # Only accept CSV files located directly in the ZIP root.
                root_path = filename

                if root_path in archive.namelist():
                    files[filename] = archive.read(root_path)

    else:
        raise ValueError(
            "Letterboxd export must be a ZIP file or an extracted directory."
        )

    if not files:
        raise ValueError(
            "No watched.csv, diary.csv, ratings.csv, or reviews.csv "
            "was found in the Letterboxd export root."
        )

    parsed: dict[str, list[dict[str, str]]] = {}

    for filename, content in files.items():
        try:
            text = content.decode("utf-8-sig")
            reader = csv.DictReader(io.StringIO(text))

            if not reader.fieldnames:
                raise ValueError("CSV has no header row.")

            if not any(
                field.strip().casefold() in {"name", "title"}
                for field in reader.fieldnames
                if field
            ):
                raise ValueError("CSV has no Name or Title column.")

            parsed[filename] = list(reader)

        except (UnicodeDecodeError, csv.Error, ValueError) as error:
            raise ValueError(
                f"Could not read Letterboxd file {filename}: {error}"
            ) from error

    return parsed


def load_letterboxd_export(
    source: str | Path,
    catalog: pd.DataFrame,
) -> LetterboxdImportResult:
    """Read an official Letterboxd CSV export and map entries to the catalog.

    Only exact normalized title matches are accepted; when a year is present,
    it must also match. Ambiguous or absent matches are returned as unmatched.
    """
    source_path = Path(source)
    export_files = _read_export_files(source_path)
    by_title_year, by_title = _catalog_lookup(catalog)
    entries: dict[tuple[str, ...], dict[str, Any]] = {}

    for filename, rows in export_files.items():
        category = _EXPORT_FILES[filename]
        for row_number, row in enumerate(rows, start=2):
            name = _row_value(row, "Name", "Title")
            if not name:
                continue

            year = _year(_row_value(row, "Year"))
            uri = _row_value(row, "Letterboxd URI", "URI")
            key = _entry_key(uri, name, year)
            entry = entries.setdefault(
                key,
                {
                    "name": name,
                    "year": year,
                    "uri": uri,
                    "watched": False,
                    "rating": None,
                    "review": "",
                    "date": "",
                },
            )
            entry["watched"] = True
            entry["name"] = name or entry["name"]
            entry["year"] = year or entry["year"]
            entry["uri"] = uri or entry["uri"]

            rating = _rating(
                _row_value(row, "Rating"),
                filename,
                row_number,
            )
            if rating is not None and (
                category == "ratings"
                or entry["rating"] is None
            ):
                entry["rating"] = rating

            review = _row_value(row, "Review", "Review Text")
            if review:
                entry["review"] = review

            date = _row_value(row, "Date", "Watched Date")
            if date:
                entry["date"] = date

    mapped_entries = []
    unmatched_titles = []
    for entry in entries.values():
        movie_id = _match_movie(
            entry["name"],
            entry["year"],
            by_title_year,
            by_title,
        )
        if movie_id is None:
            label = entry["name"]
            if entry["year"] is not None:
                label += f" ({entry['year']})"
            unmatched_titles.append(label)
            continue

        mapped_entries.append({**entry, "movie_id": movie_id})

    watched_ids = sorted({
        entry["movie_id"]
        for entry in mapped_entries
        if entry["watched"]
    })
    rated_entries = [
        entry
        for entry in mapped_entries
        if entry["rating"] is not None
    ]
    liked_ids = sorted({
        entry["movie_id"]
        for entry in rated_entries
        if entry["rating"] >= 4
    })
    disliked_ids = sorted({
        entry["movie_id"]
        for entry in rated_entries
        if entry["rating"] <= 2
    })

    genres_by_movie: dict[int, set[str]] = defaultdict(set)
    if "genres" in catalog.columns:
        for record in catalog[["movieId", "genres"]].to_dict(
            orient="records"
        ):
            if pd.isna(record["movieId"]) or pd.isna(record["genres"]):
                continue
            try:
                movie_id = int(record["movieId"])
            except (TypeError, ValueError):
                continue
            genres_by_movie[movie_id] = {
                genre.strip()
                for genre in str(record["genres"]).split("|")
                if genre.strip()
            }

    positively_rated_examples = sorted(
        (
            entry
            for entry in rated_entries
            if entry["rating"] >= 4
        ),
        key=lambda entry: (-entry["rating"], entry["name"].casefold()),
    )[:_MAX_RATED_MOVIE_EXAMPLES_PER_POLARITY]
    negatively_rated_examples = sorted(
        (
            entry
            for entry in rated_entries
            if entry["rating"] <= 2
        ),
        key=lambda entry: (entry["rating"], entry["name"].casefold()),
    )[:_MAX_RATED_MOVIE_EXAMPLES_PER_POLARITY]
    rated_movie_examples = [
        TasteMovieRating(
            title=entry["name"],
            rating=entry["rating"],
            genres=sorted(
                genres_by_movie[entry["movie_id"]],
                key=str.casefold,
            ),
        )
        for entry in [
            *positively_rated_examples,
            *negatively_rated_examples,
        ]
    ]

    genre_ratings: dict[str, list[float]] = defaultdict(list)
    for entry in rated_entries:
        for genre in genres_by_movie[entry["movie_id"]]:
            genre_ratings[genre].append(entry["rating"])

    genre_averages = {
        genre: sum(ratings) / len(ratings)
        for genre, ratings in genre_ratings.items()
        if len(ratings) >= 2
    }
    genre_rating_summary = [
        GenreTasteSummary(
            genre=genre,
            average_rating=genre_averages[genre],
            rated_movie_count=len(genre_ratings[genre]),
        )
        for genre in sorted(genre_averages, key=str.casefold)
    ]
    preferred_genres = [
        genre
        for genre, average in sorted(
            genre_averages.items(),
            key=lambda item: (-item[1], item[0].casefold()),
        )
        if average >= 3.5
    ][:3]
    less_preferred_genres = [
        genre
        for genre, average in sorted(
            genre_averages.items(),
            key=lambda item: (item[1], item[0].casefold()),
        )
        if average <= 2.5
    ][:3]

    review_entries = [
        entry
        for entry in mapped_entries
        if entry["review"].strip()
    ]
    positive_examples = sorted(
        (
            entry
            for entry in review_entries
            if entry["rating"] is not None and entry["rating"] >= 4
        ),
        key=lambda entry: (-entry["rating"], entry["name"].casefold()),
    )[:3]
    negative_examples = sorted(
        (
            entry
            for entry in review_entries
            if entry["rating"] is not None and entry["rating"] <= 2
        ),
        key=lambda entry: (entry["rating"], entry["name"].casefold()),
    )[:3]
    selected_examples = [*positive_examples, *negative_examples]
    selected_ids = {
        entry["movie_id"]
        for entry in selected_examples
    }
    remaining_examples = [
        entry
        for entry in review_entries
        if entry["movie_id"] not in selected_ids
    ]
    selected_examples.extend(
        remaining_examples[:_MAX_REVIEW_EXAMPLES - len(selected_examples)]
    )
    review_examples = [
        TasteReviewExample(
            title=entry["name"],
            rating=entry["rating"],
            review=entry["review"][:_REVIEW_LIMIT],
        )
        for entry in selected_examples
    ]

    taste_summary = None
    summary_parts = []
    if preferred_genres:
        summary_parts.append(
            "Genres rated highly: " + ", ".join(preferred_genres)
        )
    if less_preferred_genres:
        summary_parts.append(
            "Genres rated lower: " + ", ".join(less_preferred_genres)
        )
    if summary_parts:
        taste_summary = ". ".join(summary_parts) + "."

    taste_context = UserTasteContext(
        liked_movie_ids=liked_ids,
        disliked_movie_ids=disliked_ids,
        watched_movie_ids=watched_ids,
        preferred_genres=preferred_genres,
        less_preferred_genres=less_preferred_genres,
        review_examples=review_examples,
        rated_movie_examples=rated_movie_examples,
        genre_rating_summary=genre_rating_summary,
        taste_summary=taste_summary,
    )
    return LetterboxdImportResult(
        taste_context=taste_context,
        imported_entries=len(entries),
        matched_entries=len(mapped_entries),
        unmatched_titles=sorted(set(unmatched_titles), key=str.casefold),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a Letterboxd export against the MovieShelf catalog."
    )
    parser.add_argument(
        "export",
        type=Path,
        help="Path to a Letterboxd export ZIP or extracted export folder.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data") / "processed" / "movie_catalog_ranked.csv",
        help="Path to the MovieShelf ranked movie catalog.",
    )
    arguments = parser.parse_args()

    if not arguments.catalog.is_file():
        raise FileNotFoundError(
            f"Movie catalog not found: {arguments.catalog}"
        )

    result = load_letterboxd_export(
        arguments.export,
        pd.read_csv(arguments.catalog),
    )
    print(f"Entries imported: {result.imported_entries}")
    print(f"Entries matched to catalog: {result.matched_entries}")
    print(f"Watched movies matched: {len(result.taste_context.watched_movie_ids)}")
    print(f"Movies rated highly: {len(result.taste_context.liked_movie_ids)}")
    print(f"Movies rated poorly: {len(result.taste_context.disliked_movie_ids)}")
    print(
        "Rated title examples: "
        f"{len(result.taste_context.rated_movie_examples)}"
    )
    print(f"Unmatched titles: {len(result.unmatched_titles)}")
    if result.taste_context.taste_summary:
        print(f"Taste profile: {result.taste_context.taste_summary}")
    for title in result.unmatched_titles[:20]:
        print(f"Unmatched: {title}")


if __name__ == "__main__":
    main()
