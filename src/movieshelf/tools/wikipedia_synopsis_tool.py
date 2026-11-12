import json
import logging
import re
import ssl
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlencode, urlparse
from urllib.request import Request, urlopen

import truststore


logger = logging.getLogger(__name__)


class WikipediaSynopsisError(RuntimeError):
    """Raised when a Wikimedia metadata request fails or is malformed."""


class WikipediaSynopsisTool:
    """Resolve IMDb IDs to licensed English Wikipedia lead summaries."""

    WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
    WIKIPEDIA_SUMMARY_ENDPOINT = (
        "https://en.wikipedia.org/api/rest_v1/page/summary/"
    )
    WIKIPEDIA_PAGE_ENDPOINT = "https://en.wikipedia.org/w/rest.php/v1/page/"
    LICENSE_PATH_PREFIX = "/licenses/by-sa/4.0"
    USER_AGENT = (
        "MovieShelf/0.1 "
        "(https://github.com/stanionutraul/MovieShelf)"
    )
    MAX_SUMMARY_CHARACTERS = 2000
    MAX_PLOT_CHARACTERS = 12000

    def __init__(self, timeout: float = 10.0):
        if timeout <= 0:
            raise ValueError("timeout must be positive.")

        self.timeout = timeout
        self._ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self._cache: dict[str, dict[str, Any] | None] = {}

    @staticmethod
    def _normalize_imdb_id(value: Any) -> str | None:
        if value is None or isinstance(value, bool):
            return None

        imdb_id = str(value).strip()
        if not imdb_id or imdb_id.casefold() in {"0", "<na>"}:
            return None

        if imdb_id.startswith("tt"):
            digits = imdb_id[2:]
        else:
            digits = imdb_id

        if not digits.isdigit() or len(digits) > 12:
            return None

        return "tt" + digits.zfill(7)

    @classmethod
    def normalize_imdb_id(cls, value: Any) -> str | None:
        """Normalize MovieLens IMDb identifier formats."""
        return cls._normalize_imdb_id(value)

    def enrich_candidates(
        self,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Attach an attributed synopsis when a verified page is available."""
        enriched = [dict(candidate) for candidate in candidates]
        imdb_ids = {
            imdb_id
            for candidate in enriched
            if (
                imdb_id := self._normalize_imdb_id(
                    candidate.get("imdbId", candidate.get("imdb_id"))
                )
            )
        }
        uncached_ids = imdb_ids - self._cache.keys()

        articles = self._find_wikipedia_articles(uncached_ids)
        for imdb_id in uncached_ids:
            article = articles.get(imdb_id)
            self._cache[imdb_id] = (
                self._load_synopsis(*article)
                if article is not None
                else None
            )

        for candidate in enriched:
            imdb_id = self._normalize_imdb_id(
                candidate.get("imdbId", candidate.get("imdb_id"))
            )
            synopsis = self._cache.get(imdb_id) if imdb_id else None
            if synopsis:
                candidate.update(synopsis)

        return enriched

    def _find_wikipedia_articles(
        self,
        imdb_ids: set[str],
    ) -> dict[str, tuple[str, str]]:
        if not imdb_ids:
            return {}

        values = " ".join(
            f'"{imdb_id}"'
            for imdb_id in sorted(imdb_ids)
        )
        query = (
            "SELECT ?imdbId ?item ?article WHERE { "
            f"VALUES ?imdbId {{ {values} }} "
            "?item wdt:P345 ?imdbId . "
            "?article schema:about ?item ; "
            "schema:isPartOf <https://en.wikipedia.org/> . "
            "}"
        )
        url = (
            self.WIKIDATA_ENDPOINT
            + "?"
            + urlencode({"query": query, "format": "json"})
        )
        payload = self._request_json(
            url,
            accept="application/sparql-results+json",
        )
        bindings = payload.get("results", {}).get("bindings")
        if not isinstance(bindings, list):
            raise WikipediaSynopsisError(
                "Wikidata returned a response without query bindings."
            )

        articles = {}
        for binding in bindings:
            imdb_id = binding.get("imdbId", {}).get("value")
            item_uri = binding.get("item", {}).get("value")
            article_uri = binding.get("article", {}).get("value")
            if (
                imdb_id not in imdb_ids
                or not isinstance(item_uri, str)
                or not isinstance(article_uri, str)
            ):
                continue

            item_match = re.fullmatch(
                r"https?://www\.wikidata\.org/entity/(Q\d+)",
                item_uri,
            )
            article = urlparse(article_uri)
            if (
                item_match is None
                or article.scheme != "https"
                or article.netloc != "en.wikipedia.org"
                or not article.path.startswith("/wiki/")
            ):
                continue

            page_title = unquote(
                article.path.removeprefix("/wiki/")
            )
            if page_title:
                articles[imdb_id] = (
                    item_match.group(1),
                    page_title,
                )

        return articles

    def find_wikipedia_articles(
        self,
        imdb_ids: set[str],
    ) -> dict[str, tuple[str, str]]:
        """Resolve IMDb IDs to English Wikipedia articles through Wikidata."""
        return self._find_wikipedia_articles(imdb_ids)

    def fetch_plot_section(
        self,
        wikidata_item: str,
        page_title: str,
    ) -> dict[str, Any] | None:
        """Fetch a plot section with a verified page identity and license."""
        params = urlencode({
            "action": "query",
            "format": "json",
            "prop": "extracts|info|pageprops",
            "inprop": "url",
            "titles": page_title.replace("_", " "),
            "explaintext": "1",
            "exsectionformat": "wiki",
            "redirects": "1",
        })
        payload = self._request_json(
            "https://en.wikipedia.org/w/api.php?" + params,
        )
        pages = payload.get("query", {}).get("pages")
        if not isinstance(pages, dict) or len(pages) != 1:
            raise WikipediaSynopsisError(
                f"Wikipedia returned an invalid page response for {page_title}."
            )

        page = next(iter(pages.values()))
        if not isinstance(page, dict) or "missing" in page:
            return None

        pageprops = page.get("pageprops", {})
        if (
            not isinstance(pageprops, dict)
            or pageprops.get("wikibase_item") != wikidata_item
        ):
            logger.info(
                "Wikipedia page %s did not resolve to Wikidata item %s.",
                page_title,
                wikidata_item,
            )
            return None

        extract = page.get("extract")
        plot = (
            self._extract_plot_section(extract)
            if isinstance(extract, str)
            else None
        )
        if not plot:
            return None

        title = page.get("title")
        page_url = page.get("fullurl")
        revision = page.get("lastrevid")
        if (
            not isinstance(title, str)
            or not isinstance(page_url, str)
            or not isinstance(revision, int)
            or isinstance(revision, bool)
        ):
            raise WikipediaSynopsisError(
                f"Wikipedia returned incomplete plot metadata for {page_title}."
            )

        parsed_page_url = urlparse(page_url)
        if (
            parsed_page_url.scheme != "https"
            or parsed_page_url.netloc != "en.wikipedia.org"
        ):
            raise WikipediaSynopsisError(
                f"Wikipedia returned an invalid source URL for {page_title}."
            )

        license_page = self._request_json(
            self.WIKIPEDIA_PAGE_ENDPOINT
            + quote(title, safe="()_'")
        )
        license_data = license_page.get("license", {})
        license_url = license_data.get("url")
        license_title = license_data.get("title")
        parsed_license_url = (
            urlparse(license_url)
            if isinstance(license_url, str)
            else None
        )
        if (
            not isinstance(license_title, str)
            or parsed_license_url is None
            or parsed_license_url.netloc != "creativecommons.org"
            or not parsed_license_url.path.startswith(
                self.LICENSE_PATH_PREFIX
            )
        ):
            logger.warning(
                "Skipping Wikipedia page %s with unsupported license %s.",
                title,
                license_url,
            )
            return None

        return {
            "plot_summary": plot[:self.MAX_PLOT_CHARACTERS],
            "narrative_source": {
                "title": title,
                "url": page_url,
                "revision": revision,
                "revision_url": f"{page_url}?oldid={revision}",
                "license_title": license_title,
                "license_url": license_url,
            },
        }

    @staticmethod
    def _extract_plot_section(extract: str) -> str | None:
        heading = re.search(
            r"(?im)^==\s*Plot\s*==\s*$",
            extract,
        )
        if heading is None:
            return None

        remaining = extract[heading.end():]
        next_section = re.search(r"(?m)^==[^=].*?==\s*$", remaining)
        plot = remaining[:next_section.start()] if next_section else remaining
        normalized = re.sub(r"\s+", " ", plot).strip()
        return normalized or None

    def _load_synopsis(
        self,
        wikidata_item: str,
        page_title: str,
    ) -> dict[str, Any] | None:
        encoded_title = quote(page_title, safe="()_'")
        summary = self._request_json(
            self.WIKIPEDIA_SUMMARY_ENDPOINT + encoded_title,
            allow_not_found=True,
        )
        page = self._request_json(
            self.WIKIPEDIA_PAGE_ENDPOINT + encoded_title,
            allow_not_found=True,
        )
        if summary is None or page is None:
            return None

        if (
            summary.get("type") != "standard"
            or summary.get("wikibase_item") != wikidata_item
        ):
            logger.info(
                "Wikipedia page %s did not resolve to the expected film.",
                page_title,
            )
            return None

        extract = summary.get("extract")
        revision_value = summary.get("revision")
        if (
            isinstance(revision_value, str)
            and revision_value.isdecimal()
        ):
            revision = int(revision_value)
        else:
            revision = revision_value
        content_urls = summary.get("content_urls", {})
        desktop_urls = content_urls.get("desktop", {})
        page_url = desktop_urls.get("page")
        license_data = page.get("license", {})
        license_url = license_data.get("url")
        license_title = license_data.get("title")

        if (
            not isinstance(extract, str)
            or not extract.strip()
            or not isinstance(revision, int)
            or isinstance(revision, bool)
            or not isinstance(page_url, str)
            or not isinstance(license_url, str)
            or not isinstance(license_title, str)
        ):
            raise WikipediaSynopsisError(
                f"Wikipedia returned incomplete metadata for {page_title}."
            )

        parsed_page_url = urlparse(page_url)
        if (
            parsed_page_url.scheme != "https"
            or parsed_page_url.netloc != "en.wikipedia.org"
        ):
            raise WikipediaSynopsisError(
                f"Wikipedia returned an invalid source URL for {page_title}."
            )

        parsed_license_url = urlparse(license_url)
        if (
            parsed_license_url.netloc != "creativecommons.org"
            or not parsed_license_url.path.startswith(
                self.LICENSE_PATH_PREFIX
            )
        ):
            logger.warning(
                "Skipping Wikipedia page %s with unsupported license %s.",
                page_title,
                license_url,
            )
            return None

        return {
            "plot_summary": extract[:self.MAX_SUMMARY_CHARACTERS],
            "narrative_source": {
                "title": summary.get("title", page_title.replace("_", " ")),
                "url": page_url,
                "revision": revision,
                "revision_url": f"{page_url}?oldid={revision}",
                "license_title": license_title,
                "license_url": license_url,
            },
        }

    def _request_json(
        self,
        url: str,
        *,
        accept: str = "application/json",
        allow_not_found: bool = False,
    ) -> dict[str, Any] | None:
        request = Request(
            url,
            headers={
                "Accept": accept,
                "User-Agent": self.USER_AGENT,
            },
        )

        try:
            with urlopen(
                request,
                timeout=self.timeout,
                context=self._ssl_context,
            ) as response:
                raw_response = response.read()
        except HTTPError as error:
            if allow_not_found and error.code == 404:
                return None

            retry_after = error.headers.get("Retry-After")
            retry_message = (
                f" Retry after {retry_after} seconds."
                if retry_after
                else ""
            )
            raise WikipediaSynopsisError(
                f"Wikimedia request failed with HTTP {error.code}."
                f"{retry_message}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise WikipediaSynopsisError(
                f"Wikimedia request failed: {error}."
            ) from error

        try:
            payload = json.loads(raw_response.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise WikipediaSynopsisError(
                "Wikimedia returned invalid JSON."
            ) from error

        if not isinstance(payload, dict):
            raise WikipediaSynopsisError(
                "Wikimedia returned an unexpected JSON response."
            )

        return payload
