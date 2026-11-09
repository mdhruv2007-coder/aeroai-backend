"""Script G - build a leak-free Cd split: de-duplicate, then split BY AIRFOIL. (never overwrites anything)
Run from the project root:   python AEROAI_self_audit_scripts/G_make_clean_split.py
Why: the original lock split by row. 37% of train rows are exact duplicates and 73% of val rows also sit
in train, so R2 0.92 was inflated. Writes data/processed/clean_v2/{cd_train,cd_val}.csv + manifest."""
import hashlib, json
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path.cwd()
P = ROOT / "data" / "processed"
OUT = P / "clean_v2"
if OUT.exists() and any(OUT.iterdir()):
    raise SystemExit(f"{OUT} already has files - refusing to overwrite. Delete it deliberately to re-run.")
OUT.mkdir(exist_ok=True)

key = ["naca", "aoa", "re", "cd"]
full = pd.concat([pd.read_csv(P / "cd_train_locked.csv"), pd.read_csv(P / "cd_val_locked.csv")], ignore_index=True)
n0 = len(full)
full = full.drop_duplicates(key).reset_index(drop=True)
print(f"rows {n0:,} -> {len(full):,} after dropping exact duplicates ({n0 - len(full):,} removed)")
print(f"unique airfoils: {full.naca.nunique()}")

gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
tr_idx, va_idx = next(gss.split(full, groups=full.naca))
tr, va = full.iloc[tr_idx].reset_index(drop=True), full.iloc[va_idx].reset_index(drop=True)

assert not set(tr.naca) & set(va.naca), "airfoil leak"
assert not set(map(tuple, tr[key].round(6).values.tolist())) & set(map(tuple, va[key].round(6).values.tolist())), "row leak"
tr.to_csv(OUT / "cd_train.csv", index=False)
va.to_csv(OUT / "cd_val.csv", index=False)

sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
json.dump({
    "created": datetime.now().isoformat(), "seed": 42, "val_fraction_of_airfoils": 0.15,
    "reason": "Re-lock: original split was by row; 73.3% of val rows also appeared in train (exact duplicates), "
              "so reported R2 was inflated. This split removes exact duplicates and holds out whole airfoils.",
    "train": {"rows": len(tr), "airfoils": int(tr.naca.nunique()), "sha256": sha(OUT / "cd_train.csv")},
    "val": {"rows": len(va), "airfoils": int(va.naca.nunique()), "sha256": sha(OUT / "cd_val.csv")},
    "golden_rule": "Val airfoils never appear in train. Report R2 on this val set only.",
}, open(OUT / "manifest.json", "w"), indent=2)

print(f"train {len(tr):,} rows / {tr.naca.nunique()} airfoils | val {len(va):,} rows / {va.naca.nunique()} airfoils")
for n, d in (("train", tr), ("val", va)):
    print(f"  {n}: AoA {d.aoa.min():.1f}..{d.aoa.max():.1f} | Re {d.re.min():.0f}..{d.re.max():.0f} | "
          f"median Cd {d.cd.median():.4f} | symmetric airfoils (naca<1000) {(d.naca < 1000).mean()*100:.1f}%")
print(f"written to {OUT}")
