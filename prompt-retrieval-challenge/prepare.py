import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
MASK = "[MASK]"
EMAIL = re.compile(r"[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
TEST_FRACTION = 0.2
MAX_TEST_AUTHOR_PROMPTS = 40  # keep prolific authors in train so no single style dominates test
QUERY = "query_id"
DOC = "prompt_id"
TARGET_COL = "prompt_ids"
# The original "awesome-chatgpt-prompts" list (widely mirrored online and likely seen by
# pre-trained models) uses this opening; those prompts only go to train.
FAMOUS_OPENING = "i want you to act as"


def _find_csv(raw: Path) -> Path:
    files = sorted(p for p in raw.rglob("*.csv") if not p.name.lower().startswith("sample_submission"))
    for p in files:
        if {"act", "prompt"} <= set(pd.read_csv(p, nrows=0).columns):
            return p
    raise FileNotFoundError(f"No CSV with 'act' and 'prompt' columns in {raw}")


def _mask_title_words(title: str, prompt: str) -> str:
    """Hide every word of the title (3+ characters) inside the prompt, including other forms
    of the same word: words of 4+ characters are matched on their first 4-5 letters
    (review -> reviewer, reviewing; code -> coder, codebase), 3-letter words exactly or with a
    plural ending."""
    words = {w for w in re.findall(r"\w+", title.lower()) if len(w) >= 3}
    for w in sorted(words, key=len, reverse=True):
        pattern = rf"(?i)\b{re.escape(w[:5])}\w*" if len(w) >= 4 else rf"(?i)\b{re.escape(w)}(?:s|es)?\b"
        prompt = re.sub(pattern, MASK, prompt)
    return prompt


def _split_authors(df: pd.DataFrame, rng: np.random.Generator) -> set:
    """Pick whole authors for test until ~TEST_FRACTION of the prompts are covered."""
    famous = df["prompt"].str.lower().str.contains(FAMOUS_OPENING, regex=False)
    counts = df.groupby("contributor").size()
    blocked = set(df.loc[famous, "contributor"]) | set(counts[counts > MAX_TEST_AUTHOR_PROMPTS].index)
    pool = sorted(set(counts.index) - blocked)
    target, chosen, total = TEST_FRACTION * len(df), set(), 0
    for author in rng.permutation(pool):
        if total >= target:
            break
        chosen.add(author)
        total += counts[author]
    return chosen


def prepare(raw: Path, public: Path, private: Path) -> None:
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(_find_csv(raw))
    df = df.dropna(subset=["act", "prompt"])
    df["act"] = df["act"].str.strip()
    norm_prompt = df["prompt"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[~norm_prompt.duplicated() & ~df["act"].str.lower().duplicated()].reset_index(drop=True)

    # E-mail addresses inside prompt texts are placeholders, but are removed anyway.
    df["prompt_masked"] = [EMAIL.sub("[EMAIL]", _mask_title_words(a, p)) for a, p in zip(df["act"], df["prompt"])]
    # Contributor names include e-mail addresses, so authors are replaced by opaque IDs.
    authors = sorted(df["contributor"].unique())
    df["author_id"] = df["contributor"].map({a: f"a{i:04d}" for i, a in enumerate(authors)})

    test_authors = _split_authors(df, rng)
    is_test = df["contributor"].isin(test_authors).to_numpy()

    for split, mask in (("train", ~is_test), ("test", is_test)):
        part = df[mask].reset_index(drop=True)
        n = len(part)
        # Independent random numbering for queries and prompts, so IDs carry no pairing signal.
        q_order, d_order = rng.permutation(n), rng.permutation(n)
        part[QUERY] = [f"{split}_q{i:04d}" for i in np.argsort(q_order)]
        part[DOC] = [f"{split}_p{i:04d}" for i in np.argsort(d_order)]

        queries = part[[QUERY, "act"]].rename(columns={"act": "title"}).sort_values(QUERY)
        docs = part[[DOC, "prompt_masked", "type", "for_devs", "author_id"]]
        docs = docs.rename(columns={"prompt_masked": "prompt"}).sort_values(DOC)
        docs.to_csv(public / f"{split}_prompts.csv", index=False)

        truth = part[[QUERY, DOC]].rename(columns={DOC: TARGET_COL}).sort_values(QUERY)
        if split == "train":
            queries.merge(truth, on=QUERY).to_csv(public / "train.csv", index=False)
        else:
            queries.to_csv(public / "test.csv", index=False)
            truth.to_csv(private / "answers.csv", index=False)
            ids = docs[DOC].to_numpy()
            sample = queries[[QUERY]].copy()
            sample[TARGET_COL] = [" ".join(rng.choice(ids, size=10, replace=False)) for _ in range(len(sample))]
            sample.to_csv(public / "sample_submission.csv", index=False)
