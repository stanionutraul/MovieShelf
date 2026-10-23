
from pathlib import Path

import pandas as pd

from movieshelf.data_processing import repair_mojibake


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TAGS_FILE = PROJECT_ROOT / "data" / "raw" / "ml-32m" / "tags.csv"

print("Loading tags...")
tags = pd.read_csv(TAGS_FILE, usecols=["tag"])

print(f"Total rows: {len(tags):,}")

# Remove missing values and inspect each distinct tag only once.
unique_tags = tags["tag"].dropna().drop_duplicates()

print(f"Distinct non-missing tags: {len(unique_tags):,}")
print("Checking possible repairs...")

repaired_examples = []
changed_count = 0
failed_examples = []

for original in unique_tags:
    repaired = repair_mojibake(original)

    if repaired != original:
        changed_count += 1

        if len(repaired_examples) < 20:
            repaired_examples.append((original, repaired))

    elif any(marker in original for marker in ("Ã", "Â", "â€", "ï¿½")):
        if len(failed_examples) < 20:
            failed_examples.append(original)

print(f"\nDistinct tags repaired: {changed_count:,}")

print("\nExamples repaired:")
for original, repaired in repaired_examples:
    print(f"{original!r} -> {repaired!r}")

print("\nSuspicious examples not repaired:")
for text in failed_examples:
    print(repr(text))
