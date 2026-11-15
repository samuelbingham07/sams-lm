"""Download TinyStories from Hugging Face into data/ (gitignored)."""
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def main():
    ds = load_dataset("roneneldan/TinyStories", cache_dir=str(DATA_DIR / "hf_cache"))
    for split in ds:
        out = DATA_DIR / f"tinystories_{split}.txt"
        with open(out, "w", encoding="utf-8") as f:
            for row in ds[split]:
                f.write(row["text"].strip() + "\n<|endoftext|>\n")
        print(f"{split}: {len(ds[split]):,} stories -> {out}")

if __name__ == "__main__":
    main()
