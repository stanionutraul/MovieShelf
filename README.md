# MovieShelf

MovieShelf is a local movie discovery app that combines a natural-language
request, optional filters, and an optional Letterboxd export with the
MovieLens catalog.

## Run the app on Windows

Prerequisites:

- Python 3.13 or newer and the repository virtual environment.
- The processed files `data\processed\movie_catalog_ranked.csv` and
  `data\processed\movie_tags.csv`.
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
Synopsis lookup uses Wikidata and English Wikipedia, so that part of a
recommendation may make external requests.

## Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

See [Letterboxd import details](docs/letterboxd-import.md) for matching
behavior and profile limitations.
