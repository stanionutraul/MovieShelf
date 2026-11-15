import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from movieshelf.app import load_application_services, load_letterboxd_upload


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_FILES = {
    "cozy": PROJECT_ROOT
    / "data"
    / "raw"
    / "synthetic_letterboxd"
    / "01_cozy_comfort.zip",
    "crime": PROJECT_ROOT
    / "data"
    / "raw"
    / "synthetic_letterboxd"
    / "02_grounded_crime.zip",
    "animation": PROJECT_ROOT
    / "data"
    / "raw"
    / "synthetic_letterboxd"
    / "03_animation_adventure.zip",
}
PROMPTS = {
    "cozy": (
        "I’m looking for movies that are cozy, comforting, lighthearted, "
        "and don’t require too much mental energy. Something I can curl up "
        "with during winter, switch my brain off, and just feel good.\n\n"
        "Recently watched Valentine’s Day (rewatch) and the Swedish "
        "Netflix movie The Christmas Switch (2026), and both fit the bill "
        "perfectly. Doesn’t have to be romantic, holiday-themed, or any "
        "particular genre. Just something that feels like a warm hug.\n\n"
        "Language is no bar! I’m happy watching anything with subtitles. "
        "Bonus points if it’s available on an OTT platform."
    ),
    "grounded_crime": (
        "Please recommend a movie focused on the actions of 1 or maybe 2 "
        "characters, so not really big crime organizations like Goodfellas, "
        "Casino... I’d also like the movie to be as realistic as possible "
        "in the violence and the setting. Something like Good Time, "
        "Buffalo 66, Hell or High Water, Bad Lieutenant would be perfect."
    ),
    "destress": (
        "Life has not been so great for me lately, ever since I graduated, "
        "I’ve been stuck applying for jobs, and with my sickness and "
        "everything else going wrong, I just want to watch a movie to "
        "destress and lighten my mood. Please suggest a film that fits."
    ),
    "true_crime_without_sa": (
        "I’ve gotten really into true crime recently, but serial killers, "
        "especially sexual crimes, creep me out. General murders are fine "
        "and cults are okay; I want to avoid stories about a stranger "
        "randomly killing or sexually assaulting someone."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed MovieShelf prompts against local Letterboxd test "
            "profiles and save retrieval/ranking diagnostics as JSON."
        )
    )
    parser.add_argument(
        "--prompt",
        choices=["all", *PROMPTS],
        default="all",
        help="Prompt to run (default: all).",
    )
    parser.add_argument(
        "--profile",
        choices=["all", "none", *PROFILE_FILES],
        default="all",
        help="Letterboxd profile to use (default: all).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "recommendation_benchmark.json"
        ),
        help="JSON output path (default: data/processed/recommendation_benchmark.json).",
    )
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if not isinstance(value, (str, int, float, bool, type(None))):
        if pd.isna(value):
            return None
        if hasattr(value, "item"):
            return _json_safe(value.item())
        return str(value)
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    catalog, context_builder, agent = load_application_services()
    agent.collect_diagnostics = True
    prompt_ids = list(PROMPTS) if args.prompt == "all" else [args.prompt]
    profile_ids = (
        ["none", *PROFILE_FILES]
        if args.profile == "all"
        else [args.profile]
    )
    records = []

    for prompt_id in prompt_ids:
        for profile_id in profile_ids:
            taste_context = None
            if profile_id != "none":
                profile_path = PROFILE_FILES[profile_id]
                result = load_letterboxd_upload(
                    profile_path.read_bytes(),
                    catalog.movies,
                    catalog.tags,
                )
                taste_context = result.taste_context

            context = context_builder.build(
                text=PROMPTS[prompt_id],
                taste_context=taste_context,
            )
            answer = agent.run(context)
            records.append({
                "prompt_id": prompt_id,
                "profile_id": profile_id,
                "diagnostics": _json_safe(agent.last_run_diagnostics),
                "answer": answer,
            })
            selected = agent.last_run_diagnostics.get(
                "final_selected_movie_ids",
                [],
            )
            print(
                f"{prompt_id} / {profile_id}: "
                f"{len(selected)} selected; "
                f"tool calls="
                f"{agent.last_run_diagnostics.get('successful_tool_calls', 0)}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "benchmark_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved benchmark results to {args.output}")


if __name__ == "__main__":
    main()
