"""Replays every Shipd pre-submission rule seen so far on the prepared output.
Usage: python shipd_sim.py <dir containing realraw/>"""
import sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import prepare
import grade as G

S = Path(sys.argv[1]); raw = S / "realraw"
h = lambda d: {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*.csv"))}
for r in ["W1", "W2"]:
    prepare(raw, S / r / "public", S / r / "private")
pub, priv = S / "W1" / "public", S / "W1" / "private"
F = {f.name: pd.read_csv(f) for f in sorted(pub.iterdir())}
ans = pd.read_csv(priv / "answers.csv")
for n, d in F.items():
    print(f"   {n:22s} {str(d.shape):12s} {list(d.columns)}")
print(f"   {'answers.csv':22s} {str(ans.shape):12s} {list(ans.columns)}")
tr, te, ss = F["train.csv"], F["test.csv"], F["sample_submission.csv"]
trr, ter = F["train_readings.csv"], F["test_readings.csv"]
idc = "query_id"; targets = [c for c in ans.columns if c != idc]
site_hours = ter.groupby("location_id").inspection_hours.agg(["first", "nunique"])
checks = {
    "reproducible (two runs identical)": h(S / "W1") == h(S / "W2"),
    "test.csv IDs unique": te[idc].is_unique,
    "train.csv IDs unique": tr[idc].is_unique,
    "train/test feature columns match (train = test + target)": set(tr.columns) - set(targets) == set(te.columns) and set(targets) <= set(tr.columns),
    "no answer column in public test (exact)": not (set(targets) & (set(te.columns) | set(ter.columns))),
    "no answer column in public test (substring)": not [c for c in set(te.columns) | set(ter.columns) for t in targets if t in c],
    "train/test IDs disjoint": not (set(tr[idc]) & set(te[idc])),
    "test/sample/answers IDs identical": set(te[idc]) == set(ss[idc]) == set(ans[idc]),
    "sample columns == answers columns": list(ss.columns) == list(ans.columns),
    "no NaN anywhere in answers / train.csv / test.csv": not (ans.isna().any().any() or tr.isna().any().any() or te.isna().any().any()),
    "no duplicate train/test rows on shared features": tr.merge(te, on=[c for c in te.columns if c != idc]).empty,
    "readings files have identical columns": list(trr.columns) == list(ter.columns),
    "no label column in readings": "landslide_risk_48h" not in trr.columns,
    "no shared sites / record_ids between train and test": not (set(trr.location_id) & set(ter.location_id)) and not (set(trr.record_id) & set(ter.record_id)),
    "grade.py costs == inspection_hours in test_readings": (site_hours["nunique"] == 1).all() and {int(k): float(v) for k, v in site_hours["first"].items()} == G.INSPECTION_HOURS,
}
rng = np.random.default_rng(0)
av = ans.assign(visibility=np.where(rng.random(len(ans)) < 0.5, "public", "private"))
checks["known answer = 1.0 on full/public/private"] = [G.grade(ans, av), G.grade(ans, av[av.visibility == "public"]), G.grade(ans, av[av.visibility == "private"])] == [1.0, 1.0, 1.0]
for name, ok in checks.items():
    print(("PASS " if ok else "FAIL ") + name)

lab = pd.read_csv(raw / "train (1).csv")[["record_id", "landslide_risk_48h"]]
rd = ter.merge(lab, on="record_id")
rd["date"] = rd.timestamp.str[:10]
rd = rd.sort_values(["date", "location_id"])
def plan(mask, order=None):
    sel = rd[mask] if order is None else rd[mask].sort_values(order, ascending=False)
    toks = (sel.location_id.astype(str) + "@" + sel.date).groupby(sel[idc]).agg(" ".join)
    return pd.DataFrame({idc: te[idc], "site_days": te[idc].map(toks).fillna("")})
first_per_day = rd.groupby("date").cumcount() == 0
print("scores: " + " | ".join(f"{k} {G.grade(v, ans):.4f}" for k, v in {
    "sample": ss, "empty plan": plan(rd.record_id < 0), "inspect everything": plan(rd.record_id >= 0),
    "first site every day": plan(first_per_day),
    "relevant only, every other": plan((rd.landslide_risk_48h == 1) & (rd.record_id % 2 == 0)),
    "relevant + 1 decoy/day": plan((rd.landslide_risk_48h == 1) | first_per_day),
}.items()))
print("weeks:", len(te), "| min relevant tokens per answer row:", ans.site_days.str.split().str.len().min())
trr2 = trr.merge(lab, on="record_id")
rel = {(int(t.split("@")[0]), t.split("@")[1]) for cell in tr["site_days"] for t in cell.split()}
derived = np.array([int((s, d) in rel) for s, d in zip(trr2.location_id, trr2.timestamp.str[:10])])
print("train labels recoverable from train.csv:", bool((derived == trr2.landslide_risk_48h.to_numpy()).all()))
