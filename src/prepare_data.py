"""Clean TinyStories and write one text file per split.

Fixes found while inspecting the raw data (see PROJECT_HANDOFF.md, "Data"):
  1. ~130K stories contain mojibake: UTF-8 punctuation decoded as cp1252,
     e.g. "donâ€™t" instead of "don't". Some closing quotes lost
     their last byte entirely ("â€" + space), which ftfy cannot repair,
     so we fix the known sequences by hand first and let ftfy handle the rest.
  2. 230 stories are empty.
  3. ~6.6K validation stories (30%) also appear verbatim in train; they are
     removed from validation.
  4. Everything is folded to plain ASCII so the byte-level BPE tokenizer does not
     spend vocabulary on rare Unicode.

Exact duplicates (~320K stories appear twice) are deliberately KEPT: they are
part of the natural training data, and removing them is one of the privacy
defenses we evaluate later (src/defense.py).

Output (gitignored):
  data/clean/train.txt, data/clean/validation.txt
    stories separated by a line containing only <|endoftext|>
  data/clean/stats.json
    what was changed, for the report

Usage:  python src/prepare_data.py
"""
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import ftfy
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "clean"
EOT = "<|endoftext|>"

# Known mojibake sequences -> intended ASCII. Order matters: the 3-char
# sequences must be replaced before the bare 2-char "â€" fallback.
MOJIBAKE = [
    ("â€œ", '"'),   # left double quote
    ("â€\u009d", '"'),   # right double quote (when the byte survived)
    ("â€™", "'"),   # right single quote / apostrophe
    ("â€˜", "'"),   # left single quote
    ("â€“", "-"),   # en dash
    ("â€”", "-"),   # em dash
    ("â€¦", "..."), # ellipsis
    ("â€", '"'),         # right double quote whose 3rd byte was lost
]

# Correctly-encoded Unicode punctuation -> ASCII.
PUNCT = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "―": "-", "−": "-",
    "…": "...", " ": " ", "­": "",
    "´": "'", "`": "'",
})


def to_ascii(text: str) -> tuple[str, int]:
    """Fold text to ASCII. Returns (text, number of characters dropped)."""
    text = text.translate(PUNCT)
    if text.isascii():
        return text, 0
    # Strip accents: e.g. "café" -> "cafe".
    decomposed = unicodedata.normalize("NFKD", text)
    out = "".join(c for c in decomposed if c.isascii())
    dropped = sum(1 for c in decomposed if not c.isascii() and not unicodedata.combining(c))
    return out, dropped


def clean_story(text: str, stats: Counter) -> str | None:
    """Return the cleaned story, or None if it should be dropped."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.isascii():
        if "â€" in text:
            stats["mojibake_fixed"] += 1
            for bad, good in MOJIBAKE:
                text = text.replace(bad, good)
        text = ftfy.fix_text(text, normalization="NFC")
        text, dropped = to_ascii(text)
        if dropped:
            stats["stories_with_dropped_chars"] += 1
            stats["chars_dropped"] += dropped
    # Normalize whitespace: trailing spaces, 3+ newlines -> paragraph break.
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    if not text:
        stats["dropped_empty"] += 1
        return None
    if EOT in text:  # would collide with our separator; never happens in practice
        stats["dropped_contains_eot"] += 1
        return None
    return text


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("roneneldan/TinyStories", cache_dir=str(DATA_DIR / "hf_cache"))
    all_stats = {}
    train_set = set()
    for split in ("train", "validation"):
        stats = Counter()
        texts = list(ds[split]["text"])
        stats["stories_in"] = len(texts)
        cleaned = []
        for t in texts:
            c = clean_story(t, stats)
            if c is not None:
                cleaned.append(c)
        if split == "train":
            train_set = set(cleaned)
        else:
            # ~30% of the official validation stories also appear verbatim in
            # train. Remove them so (a) validation loss is honest and (b) the
            # validation set can serve as true NON-members for membership
            # inference.
            before = len(cleaned)
            cleaned = [c for c in cleaned if c not in train_set]
            stats["dropped_overlap_with_train"] = before - len(cleaned)
        stats["stories_out"] = len(cleaned)
        counts = Counter(cleaned)
        stats["unique_stories_out"] = len(counts)
        stats["duplicate_copies_kept"] = len(cleaned) - len(counts)
        stats["chars_out"] = sum(len(c) for c in cleaned)
        assert all(c.isascii() for c in cleaned)

        out_path = OUT_DIR / f"{split}.txt"
        with open(out_path, "w", encoding="ascii", newline="\n") as f:
            f.write(f"\n{EOT}\n".join(cleaned))
            f.write(f"\n{EOT}\n")
        all_stats[split] = dict(stats)
        print(f"{split}: {dict(stats)}  -> {out_path}")

    with open(OUT_DIR / "stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)


def read_stories(path) -> list[str]:
    """Read a cleaned split file back into a list of stories."""
    with open(path, encoding="ascii") as f:
        text = f.read()
    return [s.strip("\n") for s in text.split(f"\n{EOT}\n") if s.strip()]


if __name__ == "__main__":
    main()
