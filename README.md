# MovieShelf

MovieShelf is a local movie discovery app that combines a natural-language
request, optional filters, and an optional Letterboxd export with the
MovieLens catalog.

## Run the app on Windows

Prerequisites:

- Python 3.13 or newer and the repository virtual environment.
- The processed files `data\processed\movie_catalog_ranked.csv` and
  `data\processed\movie_tags.csv`, `data\processed\movie_profiles.csv`, and
  the imported `data\processed\wikipedia_plots.sqlite`.
- MovieLens `links.csv` at `data\raw\ml-32m\links.csv`.
- Ollama running locally with the configured `qwen3:4b` model available.

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\movieshelf\app.py
```

The app opens in a browser. Enter a movie request, optionally add filters, and
choose **Find my next film**. For personalized recommendations, upload the ZIP
created by Letterboxd's **Settings → Export**. The app does not require a
Letterboxd account login and does not save the uploaded export. When supplied,
Letterboxd ratings and their MovieLens tags personalize discovery alongside
the current request; movies already watched are excluded.

The request analysis and recommendation model run through local Ollama.
Semantic discovery indexes the full MovieLens movie-profile catalog, not just
the titles with imported Wikipedia plots. The initial embedding build covers
about 87,000 movies and may take 20 minutes or more depending on hardware;
subsequent starts reuse the local cache. Imported, attributed plots add
narrative evidence where available. Only requests that require verified plot
or character facts are limited to titles with those plots. Narrative evidence
is read locally; recommendation prompts are not sent to Wikipedia. The final
model sees up to eight highest-ranked candidates, including their community
ratings and vote counts, and cannot recommend a title outside that shortlist.
It may use established general knowledge only to assess a known film's
subjective tone when no verified plot is available; it must not present
unsourced plot details.
For text requests, Letterboxd can reorder request-retrieved candidates only
when they also have a direct request-tag match or match an explicitly
included genre; profile-only matches and semantic-only candidates cannot
receive a taste boost. Within that pool, request relevance and Letterboxd
taste remain equally weighted, with confidence-adjusted community rating as
a separate soft ranking signal.
When a general mood request has no verified plot for a selected title, the
app shows its catalog genres, rating evidence, and matching user tags instead
of presenting an empty synopsis placeholder.

## Import licensed plot sections

MovieLens titles can be enriched with attributed English Wikipedia plot
sections. This optional import uses Wikidata to match IMDb identifiers,
stores only pages with a verified CC BY-SA license, and records the page
revision for attribution. Requests are made sequentially and the SQLite
database is resumable.

Start with a bounded 1980s sample:

```powershell
.\.venv\Scripts\python.exe scripts\import_wikipedia_plots.py --years 1980-1989 --limit 50
```

That sample is only useful for checking the importer. To evaluate
recommendations across the full decade, import all 1980s movies; to cover
other decades, import the entire catalog:

```powershell
.\.venv\Scripts\python.exe scripts\import_wikipedia_plots.py --years 1980-1989
.\.venv\Scripts\python.exe scripts\import_wikipedia_plots.py
```

Plot import is optional for general mood, genre, and style discovery. A
decade-only import limits only requests requiring verified narrative evidence
to that imported set. Import more years to broaden narrative matching. Restart
the app after an import so its local index and embedding cache include new
plot evidence.

Retry titles that previously had no usable plot section with
`--retry-unavailable`.

Imported plot text and generated embeddings are local files under
`data\processed` and are excluded from Git. Wikipedia text is provided under
CC BY-SA and displayed with source and revision attribution. MovieLens data
has separate usage conditions; review `data\raw\ml-32m\README.txt` before
redistributing or using the project commercially.

## Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

See [Letterboxd import details](docs/letterboxd-import.md) for matching
behavior and profile limitations.

See [recommendation benchmark](docs/recommendation-benchmark.md) for a
reproducible local prompt/profile evaluation and its diagnostic output.
