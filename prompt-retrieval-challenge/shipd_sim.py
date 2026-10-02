"""Replays every Shipd pre-submission rule seen so far on the prepared output.
Usage: python shipd_sim.py <dir containing praw/prompts.csv>"""
import re, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import prepare
from grade import grade

S = Path(sys.argv[1]); raw = S / "praw"
h = lambda d: {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*.csv"))}
for r in ["Q1", "Q2"]:
    prepare(raw, S / r / "public", S / r / "private")
pub, priv = S / "Q1" / "public", S / "Q1" / "private"
F = {f.name: pd.read_csv(f) for f in sorted(pub.iterdir())}
ans = pd.read_csv(priv / "answers.csv")
for n, d in F.items():
    print(f"   {n:22s} {str(d.shape):12s} {list(d.columns)}")
print(f"   {'answers.csv':22s} {str(ans.shape):12s} {list(ans.columns)}")
tr, te, ss = F["train.csv"], F["test.csv"], F["sample_submission.csv"]
trp, tep = F["train_prompts.csv"], F["test_prompts.csv"]
idc = "query_id"; targets = [c for c in ans.columns if c != idc]
pub_test_cols = set(te.columns) | set(tep.columns)
raw_df = pd.read_csv(raw / "prompts.csv")
title_of = dict(zip(ans[idc], te.set_index(idc).loc[ans[idc]].title))
prompt_of = dict(zip(tep.prompt_id, tep.prompt))
leftover = sum(any(re.search(rf"(?i)\b{re.escape(w)}\b", prompt_of[p]) for w in re.findall(r"\w+", title_of[q].lower()) if len(w) >= 3)
               for q, p in zip(ans[idc], ans.prompt_ids))
public_text = " ".join(pd.concat([d.astype(str).agg(" ".join, axis=1) for d in F.values()]))
checks = {
    "reproducible (two runs identical)": h(S / "Q1") == h(S / "Q2"),
    "test.csv IDs unique": te[idc].is_unique,
    "train.csv IDs unique": tr[idc].is_unique,
    "train/test feature columns match (train = test + target)": set(tr.columns) - set(targets) == set(te.columns) and set(targets) <= set(tr.columns),
    "no answer column in public test (exact)": not (set(targets) & pub_test_cols),
    "no answer column in public test (substring)": not [c for c in pub_test_cols for t in targets if t in c],
    "train/test IDs disjoint (queries and prompts)": not (set(tr[idc]) & set(te[idc])) and not (set(trp.prompt_id) & set(tep.prompt_id)),
    "test/sample/answers IDs identical": set(te[idc]) == set(ss[idc]) == set(ans[idc]),
    "sample columns == answers columns": list(ss.columns) == list(ans.columns),
    "no NaN in answers / train.csv / test.csv": not (ans.isna().any().any() or tr.isna().any().any() or te.isna().any().any()),
    "no duplicate train/test rows on shared features": tr.merge(te, on=[c for c in te.columns if c != idc]).empty,
    "prompt files have identical columns": list(trp.columns) == list(tep.columns),
    "no author in both train and test": not (set(trp.author_id) & set(tep.author_id)),
    "no e-mail addresses in public files": not re.search(r"[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}", public_text),
    "no 'I want you to act as' prompts in test": not tep.prompt.str.lower().str.contains("i want you to act as").any(),
    "no title word left in its own test prompt": leftover == 0,
    "answer IDs are test prompt IDs": set(ans.prompt_ids) <= set(tep.prompt_id),
}
rng = np.random.default_rng(0)
av = ans.assign(visibility=np.where(rng.random(len(ans)) < 0.5, "public", "private"))
checks["known answer = 1.0 on full/public/private"] = [grade(ans, av), grade(ans, av[av.visibility == "public"]), grade(ans, av[av.visibility == "private"])] == [1.0, 1.0, 1.0]
for name, ok in checks.items():
    print(("PASS " if ok else "FAIL ") + name)
ids = tep.prompt_id.to_numpy()
rand = pd.DataFrame({idc: te[idc], "prompt_ids": [" ".join(rng.choice(ids, 10, replace=False)) for _ in range(len(te))]})
second = ans.assign(prompt_ids=[f"{ids[0] if ids[0] != p else ids[1]} {p}" for p in ans.prompt_ids])
print(f"scores: sample {grade(ss, ans):.4f} | random {grade(rand, ans):.4f} | correct at rank 2 {grade(second, ans):.4f} | known answer {grade(ans, ans):.4f}")
