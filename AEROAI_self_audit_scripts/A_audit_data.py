"""Script A - read-only audit of the Cd training data.

Usage:   python A_audit_data.py path/to/cd_train_locked.csv
Needs:   pandas
Answers: where do the rows come from, how many have names that cannot be parsed,
         and what do the drag labels look like?
"""
import re
import sys

import pandas as pd


def clean_name(name):
    return str(name).upper().replace("NACA", "").strip()


def parse_p_like_training(name):
    """Mimics the training script: p is the 2nd digit of the name, else None."""
    try:
        return float(clean_name(name)[1])
    except (IndexError, ValueError):
        return None


def is_four_digit(name):
    return re.fullmatch(r"\d{4}", clean_name(name)) is not None


def bands(series, edges, labels):
    cut = pd.cut(series, bins=edges, labels=labels, right=False)
    return cut.value_counts().reindex(labels)


def main(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"rows: {len(df):,}")
    print("columns:", ", ".join(df.columns))

    if "data_source" in df.columns:
        print("\nrows by data_source (1 = XFoil in your code):")
        print(df["data_source"].value_counts(dropna=False).to_string())

    if "naca" in df.columns:
        names = df["naca"].astype(str)
        not4 = ~names.map(is_four_digit)
        fallback = names.map(parse_p_like_training).isna()
        print(f"\nunique airfoil names: {names.nunique():,}")
        print("most common names:")
        print(names.value_counts().head(8).to_string())
        print(f"rows whose name is not a 4-digit NACA code: {not4.sum():,}"
              f" ({not4.mean():.1%})")
        print(f"rows where p silently becomes 4.0: {fallback.sum():,}"
              f" ({fallback.mean():.1%})")
        print(f"rows with 'kana' in the name: "
              f"{names.str.lower().str.contains('kana').sum():,}")

    if "re" in df.columns:
        print("\nReynolds number quantiles:")
        print(df["re"].quantile([0, .05, .25, .5, .75, .95, 1]).round(0).to_string())
        print(bands(df["re"], [0, 1e6, 1.5e6, 1e12],
                    ["<1.0M", "1.0-1.5M", ">=1.5M"]).to_string())

    if "aoa" in df.columns:
        edges = [-100, 0, 4, 8, 12, 16, 100]
        labels = ["<0", "0-4", "4-8", "8-12", "12-16", ">=16"]
        print("\nrows by angle of attack band:")
        print(bands(df["aoa"], edges, labels).to_string())

    if "cd" in df.columns:
        cd = df["cd"]
        print("\nCd summary:")
        print(cd.describe(percentiles=[.05, .25, .5, .75, .95]).round(5).to_string())
        print(f"share below 0.004: {(cd < 0.004).mean():.1%}"
              f" | above 0.18: {(cd > 0.18).mean():.1%}"
              f" | inside 0.006-0.025: {cd.between(0.006, 0.025).mean():.1%}")
        if "aoa" in df.columns:
            edges = [-100, 0, 4, 8, 12, 16, 100]
            labels = ["<0", "0-4", "4-8", "8-12", "12-16", ">=16"]
            band = pd.cut(df["aoa"], bins=edges, labels=labels, right=False)
            print("\nmean Cd by angle band:")
            print(cd.groupby(band, observed=False).mean().round(4).to_string())
        med = cd.median()
        print(f"\nmedian Cd = {med:.4f}. A clean, unstalled airfoil near Re = 1M"
              " is roughly 0.006-0.025.")
        if med > 0.03:
            print("-> Drag labels are dominated by high-drag rows. Find out why"
                  " before you quote absolute Cd.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
