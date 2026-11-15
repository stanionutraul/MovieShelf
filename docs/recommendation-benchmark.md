# Recommendation benchmark

`scripts\benchmark_recommendations.py` runs fixed user requests through the
local MovieShelf pipeline with no Letterboxd profile and each of the three
synthetic profiles. The script uses the local Ollama model and the cached
semantic index; it does not rebuild embeddings or import additional plots.

Run the cozy prompt across all profiles:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_recommendations.py `
  --prompt cozy `
  --profile all
```

Run the full prompt/profile matrix:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_recommendations.py
```

The default output is
`data\processed\recommendation_benchmark.json`, which is ignored by Git.
Each record includes the original prompt, structured request analysis,
explicit filters, profile-size summary, retrieval-pool counts, the highest
request-ranked candidates with their semantic/tag/Letterboxd/quality scores,
the final candidate shortlist, selected MovieLens IDs, tool-call count, and
rendered response. The benchmark is diagnostic rather than an automatic
quality score: human review is still required to judge subjective mood fit.

The test profiles are synthetic and must not be interpreted as real
Letterboxd users. To compare runs, preserve each output JSON before running
the benchmark again. The file may contain user prompts and candidate metadata;
keep it local and do not upload it without reviewing its contents.
