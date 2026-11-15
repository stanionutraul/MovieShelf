# Letterboxd import

MovieShelf can read a Letterboxd export ZIP or an extracted export folder
locally. It reads `watched.csv`, `diary.csv`, `ratings.csv`, and `reviews.csv`
when present. It does not log in to Letterboxd, scrape profiles, or save the
imported history.

Run an import check from the repository root:

```powershell
.\.venv\Scripts\python.exe -m movieshelf.letterboxd "C:\path\to\letterboxd-export.zip"
```

The command uses `data\processed\movie_catalog_ranked.csv` by default. Pass
`--catalog` to use another catalog CSV. It prints match counts, a compact taste
summary, and up to 20 unmatched titles; it does not print reviews.

An application can attach the imported profile to a recommendation request:

```python
from pathlib import Path

from movieshelf.agents.context_builder import RecommendationContextBuilder
from movieshelf.agents.recommendation_agent import RecommendationAgent
from movieshelf.data_loader import load_links
from movieshelf.letterboxd import load_letterboxd_export
from movieshelf.tools.movie_catalog_tool import MovieCatalogTool
from movieshelf.tools.wikipedia_synopsis_tool import WikipediaSynopsisTool

root = Path.cwd()
catalog = MovieCatalogTool(
    root / "data" / "processed" / "movie_catalog_ranked.csv",
    root / "data" / "processed" / "movie_tags.csv",
    links=load_links(root / "data" / "raw" / "ml-32m"),
)
profile = load_letterboxd_export(
    r"C:\path\to\letterboxd-export.zip",
    catalog.movies,
    movie_tags=catalog.tags,
).taste_context
context = RecommendationContextBuilder().build(
    text="Recommend a thoughtful film, but nothing too bleak.",
    taste_context=profile,
)
agent = RecommendationAgent(
    catalog_tool=catalog,
    synopsis_tool=WikipediaSynopsisTool(),
)
print(agent.run(context))
```

Movie matching uses normalized exact title matches and, when available, the
release year. Ambiguous and unmatched titles are reported rather than guessed.
Ratings of 4 stars or more are treated as positive examples; ratings of 2 stars
or less are treated as negative examples. Genre preferences are summarized
only when at least two matched ratings exist for that genre. At most three
positive and three negative review excerpts (up to 400 characters each) are
included in the profile; other available reviews, including unrated reviews,
can fill unused slots.
The profile also includes up to three highly rated and three low-rated titles
with their catalog genres, so the model sees representative examples instead
of opaque MovieLens IDs. These examples do not assert why a user rated a film
and are passed as soft preferences, not as plot evidence.
Mean ratings by genre and MovieLens tag include the number of rated films
behind each average. Tags present on at least two rated films provide a richer
taste signal than genre alone. The agent combines request relevance and
Letterboxd taste with equally weighted reciprocal-rank fusion, then applies
confidence-adjusted community rating as a separate soft ranking signal.
This quality signal uses the catalog's weighted rating and vote count rather
than a hard popularity cutoff, so lower-rated-confidence titles remain
discoverable. Candidate discovery also includes films matching the user's
rated tag preferences. The final model receives community rating evidence and
candidate-specific Letterboxd tag matches when choosing from the shortlist.

The profile remains a preference signal, not a hard filter. Explicit request
criteria and filters remain mandatory, and movies already watched are
excluded. Profile-only requests can seed discovery from tags on highly rated
films, falling back to preferred genres when tag history is insufficient.
For requests that also contain text, profile-tag matches can personalize the
ordering only after a movie has entered the request-retrieved candidate pool
and has a direct request-tag match or matches an explicitly included genre.
A profile-only match cannot introduce a new movie into that pool or boost a
semantic-only match. This keeps personalization from replacing the current
request, although semantic-only candidates can still be weak matches and are
measured by the local recommendation benchmark.
When used with the local Ollama integration, the compact profile and selected
review excerpts are included in the local model context.

The current MovieLens catalog does not contain verified content-warning data
for gore or jump scares. If the user explicitly asks to avoid either, the
agent currently returns a clear limitation instead of making an unverifiable
recommendation. Soft wording such as “not necessarily looking for cheap jump
scares or gore just for the sake of it” is not treated as a strict exclusion.
