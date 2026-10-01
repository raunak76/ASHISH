"""Re-creates every Shipd pre-submission rule seen so far, on a prepared public/private dir."""
import sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import prepare
from grade import grade

S = Path(sys.argv[1]); raw = S / "realraw"  # usage: python shipd_sim.py <dir containing realraw/>
h = lambda d: {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*.csv"))}
for r in ["V1", "V2"]:
    prepare(raw, S / r / "public", S / r / "private")
pub, priv = S / "V1" / "public", S / "V1" / "private"
F = {f.name: pd.read_csv(f) for f in sorted(pub.iterdir())}
ans = pd.read_csv(priv / "answers.csv")
for n, d in F.items():
    print(f"   {n:22s} {str(d.shape):12s} {list(d.columns)}")
print(f"   {'answers.csv':22s} {str(ans.shape):12s} {list(ans.columns)}")
tr, te, ss = F["train.csv"], F["test.csv"], F["sample_submission.csv"]
idc = "query_id"; targets = [c for c in ans.columns if c != idc]
checks = {
    "reproducible (two runs identical)": h(S / "V1") == h(S / "V2"),
    "test.csv IDs unique": te[idc].is_unique,
    "train.csv IDs unique": tr[idc].is_unique,
    "train/test feature columns match (train = test + target)": set(tr.columns) - set(targets) == set(te.columns) and set(targets) <= set(tr.columns),
    "no answer column in public test (exact)": not (set(targets) & set(te.columns)),
    "no answer column in public test (substring)": not [c for c in te.columns for t in targets if t in c],
    "train/test IDs disjoint": not (set(tr[idc]) & set(te[idc])),
    "test/sample/answers IDs identical": set(te[idc]) == set(ss[idc]) == set(ans[idc]),
    "sample columns == answers columns": list(ss.columns) == list(ans.columns),
    "no NaN in answers": not ans.isna().any().any(),
    "no NaN target in train.csv": not tr[targets].isna().any().any(),
    "no duplicate train/test rows on shared features": tr.merge(te, on=[c for c in te.columns if c != idc]).empty,
    "readings files have identical columns": list(F["train_readings.csv"].columns) == list(F["test_readings.csv"].columns),
    "no label column in readings": "landslide_risk_48h" not in F["train_readings.csv"].columns,
    "no shared sites between train and test readings": not (set(F["train_readings.csv"].location_id) & set(F["test_readings.csv"].location_id)),
    "no shared record_ids": not (set(F["train_readings.csv"].record_id) & set(F["test_readings.csv"].record_id)),
}
rng = np.random.default_rng(0)
av = ans.assign(visibility=np.where(rng.random(len(ans)) < 0.5, "public", "private"))
checks["known answer = 1.0 on full/public/private"] = [grade(ans, av), grade(ans, av[av.visibility == "public"]), grade(ans, av[av.visibility == "private"])] == [1.0, 1.0, 1.0]
for name, ok in checks.items():
    print(("PASS " if ok else "FAIL ") + name)
lab = pd.read_csv(raw / "train (1).csv")[["record_id", "landslide_risk_48h"]]
rd = F["test_readings.csv"].merge(lab, on="record_id"); rd = rd[rd[idc].isin(te[idc])]
def build(col, asc=False):
    return pd.DataFrame([[q, " ".join(map(str, g.sort_values(col, ascending=asc).location_id.tolist()[:3]))] for q, g in rd.groupby(idc)], columns=[idc, "location_ids"])
rd["r"] = rng.random(len(rd)); rd["rain"] = rd.rainfall_72h_mm.fillna(0)
print(f"scores: sample {grade(ss, ans):.4f} | random {grade(build('r'), ans):.4f} | rain72 {grade(build('rain'), ans):.4f} | worst {grade(build(['landslide_risk_48h', 'location_id'], [True, True]), ans):.4f}")
# train labels derived from train.csv must equal the raw labels
trr = F["train_readings.csv"].merge(lab, on="record_id")
rel = {(q, int(s)) for q, sites in zip(tr[idc], tr["location_ids"].astype(str)) for s in sites.split()}
derived = np.array([int((q, s) in rel) for q, s in zip(trr[idc], trr.location_id)])
print("train labels recoverable from train.csv:", bool((derived == trr.landslide_risk_48h.to_numpy()).all()), "| train positive rate", round(derived.mean(), 4))
