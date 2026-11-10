import hashlib
import logging
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from movieshelf.agents.context_builder import RecommendationContextBuilder
from movieshelf.agents.recommendation_agent import RecommendationAgent
from movieshelf.data_loader import load_links
from movieshelf.letterboxd import (
    LetterboxdImportResult,
    load_letterboxd_export,
)
from movieshelf.tools.movie_catalog_tool import MovieCatalogTool
from movieshelf.tools.wikipedia_synopsis_tool import WikipediaSynopsisTool


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "movie_catalog_ranked.csv"
TAGS_PATH = PROJECT_ROOT / "data" / "processed" / "movie_tags.csv"
LINKS_PATH = PROJECT_ROOT / "data" / "raw" / "ml-32m"

st.set_page_config(
    page_title="MovieShelf - find your next film",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading the MovieShelf catalog...")
def load_application_services() -> tuple[
    MovieCatalogTool,
    RecommendationContextBuilder,
    RecommendationAgent,
]:
    catalog = MovieCatalogTool(
        CATALOG_PATH,
        TAGS_PATH,
        links=load_links(LINKS_PATH),
    )
    context_builder = RecommendationContextBuilder()
    recommendation_agent = RecommendationAgent(
        catalog_tool=catalog,
        synopsis_tool=WikipediaSynopsisTool(),
    )
    return catalog, context_builder, recommendation_agent


def load_letterboxd_upload(
    content: bytes,
    catalog: pd.DataFrame,
) -> LetterboxdImportResult:
    """Import an uploaded ZIP without retaining it after the request."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".zip",
            prefix="movieshelf-letterboxd-",
            delete=False,
        ) as temporary_file:
            temporary_file.write(content)
            temporary_path = Path(temporary_file.name)

        return load_letterboxd_export(temporary_path, catalog)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def build_recommendation_filters(
    include_genres: list[str],
    genre_match: str,
    exclude_genres: list[str],
    minimum_rating_count: int,
    minimum_rating: float,
    release_years: tuple[int, int],
) -> dict[str, str | int | float]:
    filters: dict[str, str | int | float] = {
        "release_year_from": release_years[0],
        "release_year_to": release_years[1],
    }
    if include_genres:
        filters["include_genres"] = "|".join(include_genres)
        filters["genre_match"] = genre_match
    if exclude_genres:
        filters["exclude_genres"] = "|".join(exclude_genres)
    if minimum_rating_count > 0:
        filters["minimum_rating_count"] = minimum_rating_count
    if minimum_rating > 0:
        filters["minimum_rating"] = minimum_rating
    return filters


def _catalog_genres(catalog: MovieCatalogTool) -> list[str]:
    genres = {
        genre.strip()
        for value in catalog.movies["genres"].dropna()
        for genre in str(value).split("|")
        if genre.strip() and genre.strip() != "(no genres listed)"
    }
    return sorted(genres)


def _catalog_year_range(catalog: MovieCatalogTool) -> tuple[int, int]:
    years = (
        catalog.movies["title"]
        .astype(str)
        .str.extract(r"\((\d{4})\)\s*$", expand=False)
        .dropna()
        .astype(int)
    )
    if years.empty:
        return 1900, 2026
    return int(years.min()), max(int(years.max()), 2026)


def _load_uploaded_letterboxd(
    uploaded_file,
    catalog: MovieCatalogTool,
) -> None:
    content = uploaded_file.getvalue()
    upload_key = hashlib.sha256(content).hexdigest()
    if st.session_state.get("letterboxd_upload_key") == upload_key:
        return

    st.session_state["letterboxd_upload_key"] = upload_key
    st.session_state.pop("letterboxd_profile", None)
    st.session_state.pop("letterboxd_import_result", None)
    try:
        result = load_letterboxd_upload(content, catalog.movies)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        logger.exception("Could not import the uploaded Letterboxd export.")
        st.session_state["letterboxd_import_error"] = str(error)
        return

    st.session_state.pop("letterboxd_import_error", None)
    st.session_state["letterboxd_profile"] = result.taste_context
    st.session_state["letterboxd_import_result"] = result


def _render_profile_sidebar(catalog: MovieCatalogTool) -> None:
    st.sidebar.markdown("## Your Letterboxd")
    st.sidebar.caption(
        "Upload your official export ZIP to personalize recommendations. "
        "No Letterboxd login is needed."
    )
    uploaded_file = st.sidebar.file_uploader(
        "Letterboxd export",
        type=["zip"],
        help=(
            "Choose the ZIP downloaded from Letterboxd > Settings > Export."
        ),
        key="letterboxd_export",
    )

    if uploaded_file is None:
        st.session_state.pop("letterboxd_upload_key", None)
        st.session_state.pop("letterboxd_profile", None)
        st.session_state.pop("letterboxd_import_result", None)
        st.session_state.pop("letterboxd_import_error", None)
        st.sidebar.info(
            "You can also get recommendations without a Letterboxd profile."
        )
        return

    _load_uploaded_letterboxd(uploaded_file, catalog)
    if error := st.session_state.get("letterboxd_import_error"):
        st.sidebar.error(f"Import failed: {error}")
        return

    result: LetterboxdImportResult | None = st.session_state.get(
        "letterboxd_import_result"
    )
    if result is None:
        return

    taste = result.taste_context
    st.sidebar.success("Letterboxd profile ready")
    metric_columns = st.sidebar.columns(2)
    metric_columns[0].metric("Matched", result.matched_entries)
    metric_columns[1].metric(
        "Watched",
        len(taste.watched_movie_ids),
    )
    st.sidebar.caption(
        f"{result.imported_entries} entries read | "
        f"{len(taste.review_examples)} reviews available"
    )
    if result.unmatched_titles:
        with st.sidebar.expander(
            f"{len(result.unmatched_titles)} unmatched titles"
        ):
            st.write("\n".join(
                f"- {title}"
                for title in result.unmatched_titles[:30]
            ))
            if len(result.unmatched_titles) > 30:
                st.caption("Showing the first 30 unmatched titles.")


def _render_header() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">YOUR PERSONAL MOVIE DISCOVERY</div>
          <h1>Find a film that <em>feels right.</em></h1>
          <p>Tell us what you're in the mood for. MovieShelf combines your
          words, your Letterboxd taste, and the movie catalog to find a
          thoughtful next watch.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #101413; color: #f3f0e7; }
        [data-testid="stSidebar"] { background: #171d1b; border-right: 1px solid #29312d; }
        .block-container { max-width: 1120px; padding-top: 2rem; }
        .hero { padding: 2.2rem 0 1.8rem; border-bottom: 1px solid #303833; margin-bottom: 1.6rem; }
        .hero .eyebrow { color: #a4c47a; letter-spacing: .16em; font-size: .72rem; font-weight: 700; }
        .hero h1 { font-family: Georgia, serif; font-size: clamp(2.4rem, 5vw, 4.1rem); color: #f3f0e7; margin: .55rem 0; }
        .hero h1 em { color: #c1d69b; }
        .hero p { max-width: 650px; color: #b6c0b9; font-size: 1.05rem; line-height: 1.65; }
        h2, h3 { font-family: Georgia, serif !important; color: #f3f0e7 !important; }
        div[data-testid="stForm"] { background: #171d1b; border: 1px solid #303833; border-radius: 16px; padding: 1.2rem 1.35rem; }
        .stTextArea textarea { background: #111715; color: #f3f0e7; border-color: #39443d; }
        .stButton button[kind="primary"], button[kind="primaryFormSubmit"] { background: #c1d69b; color: #152017; border: 0; border-radius: 8px; font-weight: 700; }
        .stButton button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover { background: #d3e6af; color: #152017; }
        [data-testid="stMetric"] { background: #202824; border: 1px solid #303833; padding: .8rem; border-radius: 10px; }
        [data-testid="stMetricValue"] { color: #c1d69b; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        catalog, context_builder, recommendation_agent = (
            load_application_services()
        )
    except Exception as error:
        logger.exception("Could not initialize MovieShelf.")
        st.error(
            "MovieShelf could not start. Check that the processed catalog "
            "and MovieLens links data exist, then reload the page. "
            f"Details: {error}"
        )
        st.stop()

    _render_profile_sidebar(catalog)
    _render_header()

    minimum_year, maximum_year = _catalog_year_range(catalog)
    genre_options = _catalog_genres(catalog)
    with st.form("recommendation_form"):
        st.subheader("What are you looking for?")
        request = st.text_area(
            "Describe the kind of film you want",
            placeholder=(
                "A psychological thriller with an unpredictable ending, "
                "something thought-provoking but not gory..."
            ),
            height=150,
            label_visibility="collapsed",
        )
        random_recommendation = st.checkbox(
            "Surprise me with a random pick",
            help="Use this when you want a discovery without a text request.",
        )

        with st.expander("Fine-tune your search"):
            genre_columns = st.columns(2)
            included_genres = genre_columns[0].multiselect(
                "Include genres",
                genre_options,
                help=(
                    'With "All", a movie must have every selected genre.'
                ),
            )
            genre_match_label = genre_columns[0].selectbox(
                "Match included genres",
                ["Any", "All"],
                disabled=not included_genres,
            )
            excluded_genres = genre_columns[1].multiselect(
                "Exclude genres",
                genre_options,
            )

            year_columns = st.columns(2)
            release_years = year_columns[0].slider(
                "Release years",
                min_value=minimum_year,
                max_value=maximum_year,
                value=(minimum_year, maximum_year),
            )
            rating_columns = st.columns(2)
            minimum_rating = rating_columns[0].slider(
                "Minimum community rating",
                min_value=0.0,
                max_value=5.0,
                value=0.0,
                step=0.5,
                help="Set to 0 to leave this filter off.",
            )
            minimum_rating_count = rating_columns[1].number_input(
                "Minimum rating count",
                min_value=0,
                max_value=100_000,
                value=0,
                step=100,
                help="Set to 0 to leave this filter off.",
            )

        submitted = st.form_submit_button(
            "Find my next film",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not request.strip() and not random_recommendation:
            st.warning(
                "Describe the film you want, or select the random-pick option."
            )
        else:
            filters = build_recommendation_filters(
                include_genres=included_genres,
                genre_match=(
                    "all" if genre_match_label == "All" else "any"
                ),
                exclude_genres=excluded_genres,
                minimum_rating_count=int(minimum_rating_count),
                minimum_rating=float(minimum_rating),
                release_years=release_years,
            )
            try:
                with st.spinner(
                    "Reading your request and finding films that fit..."
                ):
                    context = context_builder.build(
                        text=request.strip() or None,
                        filters=filters,
                        taste_context=st.session_state.get(
                            "letterboxd_profile"
                        ),
                        random_recommendation=random_recommendation,
                    )
                    st.session_state["recommendation_result"] = (
                        recommendation_agent.run(context)
                    )
                st.session_state.pop("recommendation_error", None)
            except Exception as error:
                logger.exception("Movie recommendation request failed.")
                st.session_state.pop("recommendation_result", None)
                st.session_state["recommendation_error"] = str(error)

    if error := st.session_state.get("recommendation_error"):
        st.error(
            "MovieShelf could not complete this recommendation. Check that "
            "Ollama is running with the configured model and try again. "
            f"Details: {error}"
        )

    if result := st.session_state.get("recommendation_result"):
        st.markdown("---")
        st.subheader("Your movie shortlist")
        st.markdown(result)

    st.markdown("---")
    st.caption(
        "Your Letterboxd export is processed locally and is not saved by "
        "MovieShelf. Recommendations use your local Ollama model; licensed "
        "synopsis lookup may contact Wikidata and Wikipedia."
    )


if __name__ == "__main__":
    main()
