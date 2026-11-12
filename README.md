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
Letterboxd account login and does not save the uploaded export.

The request analysis and recommendation model run through local Ollama.
Recommendations use a local semantic index containing only movies with an
available, verified Wikipedia plot. The index grows as plot data is imported;
its first build may take several minutes depending on its size and hardware.
Narrative evidence is read locally; recommendation prompts are not sent to
Wikipedia. Catalog tag searches retain mood words from the original request,
while the final model sees only the four highest-ranked candidates and cannot
recommend a title outside that shortlist.

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

Import the plot data before starting MovieShelf. The app searches only titles
with imported licensed plots, so a decade-only import limits recommendations
to that decade. For general requests, import the entire catalog. Restart the
app after an import so its local index and embedding cache include the new
plots.

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
