import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

from movieshelf.tools.wikipedia_synopsis_tool import (
    WikipediaSynopsisError,
    WikipediaSynopsisTool,
)


logger = logging.getLogger(__name__)


class WikipediaPlotStore:
    """Persist attributed Wikipedia plot sections for local recommendation."""

    MAX_PROFILE_TAGS = 15
    COMPLETED_STATUSES = (
        "available",
        "unavailable",
        "no_article",
        "unlinked",
    )

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS movie_plots (
                    movie_id INTEGER PRIMARY KEY,
                    imdb_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    plot_text TEXT,
                    page_title TEXT,
                    page_url TEXT,
                    revision INTEGER,
                    license_title TEXT,
                    license_url TEXT,
                    error TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def completed_movie_ids(self) -> set[int]:
        statuses = ",".join("?" for _ in self.COMPLETED_STATUSES)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT movie_id FROM movie_plots "
                f"WHERE status IN ({statuses})",
                self.COMPLETED_STATUSES,
            )
            return {int(row["movie_id"]) for row in rows}

    def movie_ids_with_status(self, status: str) -> set[int]:
        if status not in {*self.COMPLETED_STATUSES, "error"}:
            raise ValueError(f"Unsupported plot-import status: {status}.")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT movie_id FROM movie_plots WHERE status = ?",
                (status,),
            )
            return {int(row["movie_id"]) for row in rows}

    def save(
        self,
        movie_id: int,
        imdb_id: str,
        status: str,
        plot: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        allowed_statuses = {
            *self.COMPLETED_STATUSES,
            "error",
        }
        if status not in allowed_statuses:
            raise ValueError(f"Unsupported plot-import status: {status}.")

        source = plot.get("narrative_source", {}) if plot else {}
        if status == "available":
            if (
                not isinstance(plot, dict)
                or not isinstance(plot.get("plot_summary"), str)
                or not plot["plot_summary"].strip()
                or not isinstance(source, dict)
                or not isinstance(source.get("title"), str)
                or not isinstance(source.get("url"), str)
                or not isinstance(source.get("revision"), int)
                or isinstance(source.get("revision"), bool)
                or not isinstance(source.get("license_title"), str)
                or not isinstance(source.get("license_url"), str)
                or not source["license_url"].startswith(
                    "https://creativecommons.org/licenses/by-sa/4.0"
                )
            ):
                raise ValueError(
                    "Available plots require text, revision, source, "
                    "and CC BY-SA 4.0 attribution."
                )

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO movie_plots (
                    movie_id, imdb_id, status, plot_text, page_title,
                    page_url, revision, license_title, license_url, error,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(movie_id) DO UPDATE SET
                    imdb_id = excluded.imdb_id,
                    status = excluded.status,
                    plot_text = excluded.plot_text,
                    page_title = excluded.page_title,
                    page_url = excluded.page_url,
                    revision = excluded.revision,
                    license_title = excluded.license_title,
                    license_url = excluded.license_url,
                    error = excluded.error,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    movie_id,
                    imdb_id,
                    status,
                    plot.get("plot_summary") if plot else None,
                    source.get("title"),
                    source.get("url"),
                    source.get("revision"),
                    source.get("license_title"),
                    source.get("license_url"),
                    error,
                ),
            )

    def enrich_candidates(
        self,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Attach locally stored plot evidence and its attribution."""
        movie_ids = {
            movie_id
            for candidate in candidates
            if (movie_id := candidate.get("movieId")) is not None
        }
        if not movie_ids:
            return [dict(candidate) for candidate in candidates]

        movie_id_list = list(movie_ids)
        plots = {}
        with self._connect() as connection:
            for batch_start in range(0, len(movie_id_list), 900):
                batch = movie_id_list[batch_start:batch_start + 900]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    "SELECT * FROM movie_plots "
                    f"WHERE status = 'available' AND movie_id IN "
                    f"({placeholders})",
                    batch,
                )
                plots.update({
                    int(row["movie_id"]): row
                    for row in rows
                })

        enriched = []
        for candidate in candidates:
            result = dict(candidate)
            row = plots.get(candidate.get("movieId"))
            if row is not None:
                result["plot_summary"] = row["plot_text"]
                result["narrative_source"] = {
                    "title": row["page_title"],
                    "url": row["page_url"],
                    "revision": row["revision"],
                    "revision_url": (
                        f"{row['page_url']}?oldid={row['revision']}"
                    ),
                    "license_title": row["license_title"],
                    "license_url": row["license_url"],
                }
            enriched.append(result)
        return enriched

    def add_plots_to_profiles(
        self,
        profiles: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add plot evidence while retaining a bounded set of catalog tags."""
        required = {"movieId", "profile_text"}
        missing = required - set(profiles.columns)
        if missing:
            raise ValueError(
                f"Movie profiles are missing columns: {sorted(missing)}"
            )

        with self._connect() as connection:
            plots = pd.read_sql_query(
                "SELECT movie_id, plot_text FROM movie_plots "
                "WHERE status = 'available'",
                connection,
            )

        result = profiles.copy()
        if plots.empty:
            return result

        result = result.merge(
            plots,
            left_on="movieId",
            right_on="movie_id",
            how="left",
            validate="one_to_one",
        )
        has_plot = result["plot_text"].notna()
        source_profiles = (
            result.loc[has_plot, "profile_text"].astype(str)
        )
        profile_parts = source_profiles.str.partition(
            " Descriptive tags:"
        )
        catalog_facts = profile_parts[0].str.rstrip()
        descriptive_tags = profile_parts[2].map(
            lambda value: ", ".join(
                tag.strip()
                for tag in value.rstrip(".").split(",")
                if tag.strip()
            )
        )
        descriptive_tags = descriptive_tags.map(
            lambda value: ", ".join(
                value.split(", ")[:self.MAX_PROFILE_TAGS]
            )
        )
        tagged_profiles = (
            catalog_facts
            + " Plot: "
            + result.loc[has_plot, "plot_text"].astype(str).str.rstrip(" .")
            + ". Descriptive tags: "
            + descriptive_tags
            + "."
        )
        result.loc[has_plot, "profile_text"] = tagged_profiles
        return result.drop(columns=["movie_id", "plot_text"])

    def coverage(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count "
                "FROM movie_plots GROUP BY status"
            )
            return {row["status"]: int(row["count"]) for row in rows}


class WikipediaPlotImporter:
    """Fetch and resume a licensed Wikipedia plot-section import."""

    def __init__(
        self,
        store: WikipediaPlotStore,
        wikipedia: WikipediaSynopsisTool | None = None,
        request_delay: float = 0.2,
        sleep=time.sleep,
    ):
        if request_delay < 0:
            raise ValueError("request_delay must be non-negative.")
        self.store = store
        self.wikipedia = wikipedia or WikipediaSynopsisTool()
        self.request_delay = request_delay
        self.sleep = sleep

    def import_movies(
        self,
        movies: pd.DataFrame | Iterable[dict[str, Any]],
        limit: int | None = None,
        batch_size: int = 100,
        retry_unavailable: bool = False,
    ) -> dict[str, int]:
        if batch_size < 1 or batch_size > 100:
            raise ValueError("batch_size must be between 1 and 100.")
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive.")

        records = (
            movies.to_dict(orient="records")
            if isinstance(movies, pd.DataFrame)
            else list(movies)
        )
        if retry_unavailable:
            retry_ids = self.store.movie_ids_with_status("unavailable")
            pending = [
                record
                for record in records
                if record.get("movieId") in retry_ids
            ]
        else:
            completed = self.store.completed_movie_ids()
            pending = [
                record
                for record in records
                if record.get("movieId") not in completed
            ]
        if limit is not None:
            pending = pending[:limit]

        totals = {
            "available": 0,
            "unavailable": 0,
            "no_article": 0,
            "unlinked": 0,
            "error": 0,
        }
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            imdb_to_record = {}
            for record in batch:
                raw_imdb_id = record.get("imdbId")
                imdb_id = self.wikipedia.normalize_imdb_id(raw_imdb_id)
                movie_id = int(record["movieId"])
                if imdb_id is None:
                    self.store.save(movie_id, "", "unlinked")
                    totals["unlinked"] += 1
                else:
                    imdb_to_record[imdb_id] = record

            articles = (
                self.wikipedia.find_wikipedia_articles(
                    set(imdb_to_record)
                )
                if imdb_to_record
                else {}
            )
            if imdb_to_record and self.request_delay:
                self.sleep(self.request_delay)

            for imdb_id, record in imdb_to_record.items():
                movie_id = int(record["movieId"])
                article = articles.get(imdb_id)
                if article is None:
                    self.store.save(movie_id, imdb_id, "no_article")
                    totals["no_article"] += 1
                    continue

                try:
                    plot = self.wikipedia.fetch_plot_section(*article)
                except WikipediaSynopsisError as error:
                    self.store.save(
                        movie_id,
                        imdb_id,
                        "error",
                        error=str(error),
                    )
                    totals["error"] += 1
                    logger.warning(
                        "Plot import failed for movieId %s: %s",
                        movie_id,
                        error,
                    )
                else:
                    status = "available" if plot else "unavailable"
                    self.store.save(movie_id, imdb_id, status, plot=plot)
                    totals[status] += 1

                if self.request_delay:
                    self.sleep(self.request_delay)

            logger.info(
                "Processed plot-import batch %s/%s; totals=%s",
                offset // batch_size + 1,
                (len(pending) + batch_size - 1) // batch_size,
                totals,
            )

        return totals
