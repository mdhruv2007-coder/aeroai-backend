"""
BigFoil Airfoil Polar Parser
============================
Parses BigFoil.com Excel (.xlsx) exports into clean CSV for stall margin model training.

Usage:
    python bigfoil_parser.py                          # parse all .xlsx in raw/bigfoil/
    python bigfoil_parser.py --file e387.xlsx         # parse single file
    python bigfoil_parser.py --re-min 100000          # filter by Re range
    python bigfoil_parser.py --re-max 2000000
    python bigfoil_parser.py --mach-max 0.10          # filter Mach range
"""

import os
import re
import sys
import logging
import argparse
import traceback
from pathlib import Path

import pandas as pd
import numpy as np
import openpyxl

# ── Paths (WSL-compatible) ────────────────────────────────────────────────────
RAW_DIR  = Path("/mnt/d/Projects/Personal project 1-AERO AI/MVP Prototype-1/data/raw/bigfoil")
PROC_DIR = Path("/mnt/d/Projects/Personal project 1-AERO AI/MVP Prototype-1/data/processed")

OUTPUT_POLARS = PROC_DIR / "bigfoil_polars.csv"
OUTPUT_STALL  = PROC_DIR / "bigfoil_stall_margins.csv"
LOG_FILE      = PROC_DIR / "parse_log.txt"

PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ── Sheet name parser ─────────────────────────────────────────────────────────
_RE_SHEET = re.compile(
    r"(?P<idx>\d+)\.\s*Re\s*(?P<re>[\d\.]+)(?P<re_unit>[Mk]?)\s*Mach\s*(?P<mach>[\d\.]+)",
    re.IGNORECASE,
)

def _parse_re_value(value_str: str, unit: str) -> float:
    v = float(value_str)
    u = unit.upper()
    if u == "M":
        return v * 1_000_000
    elif u == "K":
        return v * 1_000
    return v

def parse_sheet_name(name: str):
    m = _RE_SHEET.match(name.strip())
    if not m:
        return None
    try:
        idx  = int(m.group("idx"))
        re_v = _parse_re_value(m.group("re"), m.group("re_unit"))
        mach = float(m.group("mach"))
        return idx, re_v, mach
    except ValueError:
        return None


# ── Index sheet reader ────────────────────────────────────────────────────────
def read_index(wb):
    mapping = {}
    if "index" not in wb.sheetnames:
        return mapping
    ws = wb["index"]
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        try:
            idx = int(row[0])
            src = str(row[3]).strip() if row[3] else "Unknown"
            mapping[idx] = src
        except (ValueError, TypeError, IndexError):
            continue
    return mapping


# ── Single sheet parser ───────────────────────────────────────────────────────
def parse_data_sheet(ws):
    rows = list(ws.iter_rows(values_only=True))

    header_row_idx = None
    for i, row in enumerate(rows):
        row_str = " ".join(str(c) for c in row if c is not None).lower()
        if "alpha" in row_str and "cl" in row_str:
            header_row_idx = i
            break

    if header_row_idx is None:
        return None

    data_rows = rows[header_row_idx + 1:]
    records = []
    for row in data_rows:
        vals = row[:5]
        if any(v is None for v in vals[:3]):
            continue
        try:
            alpha = float(vals[0])
            cl    = float(vals[1])
            cd    = float(vals[2])
            cm    = float(vals[3]) if vals[3] is not None else np.nan
            clcd  = float(vals[4]) if vals[4] is not None else (cl / cd if cd != 0 else np.nan)
            records.append({
                "alpha_deg": round(alpha, 4),
                "Cl":        round(cl,    6),
                "Cd":        round(cd,    6),
                "Cm":        round(cm,    6),
                "Cl_Cd":     round(clcd,  4),
            })
        except (TypeError, ValueError):
            continue

    if not records:
        return None
    return pd.DataFrame(records)


# ── Metadata extractor ────────────────────────────────────────────────────────
def extract_airfoil_name(wb, sheet_name: str, fallback: str) -> str:
    try:
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(min_row=1, max_row=6, values_only=True))
        short = str(rows[1][0]).strip() if rows[1][0] else ""
        full  = str(rows[2][0]).strip() if rows[2][0] else ""
        if full and full.lower() not in ("none", "nan", ""):
            return full
        if short and short.lower() not in ("none", "nan", ""):
            return short
    except Exception:
        pass
    return fallback


# ── Stall margin computation ──────────────────────────────────────────────────
def compute_stall_margins(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for keys, grp in df.groupby(["airfoil_name", "Re", "Mach", "datasource"], sort=False):
        grp = grp.copy()
        stall_idx = grp["Cl"].idxmax()
        stall_aoa = grp.loc[stall_idx, "alpha_deg"]
        stall_cl  = grp.loc[stall_idx, "Cl"]
        grp["stall_aoa"]        = round(stall_aoa, 4)
        grp["stall_Cl"]         = round(stall_cl, 6)
        grp["stall_margin_deg"] = (grp["stall_aoa"] - grp["alpha_deg"]).round(4)
        results.append(grp)
    if not results:
        return df
    return pd.concat(results, ignore_index=True)


# ── Single file parser ────────────────────────────────────────────────────────
def parse_excel_file(filepath: Path, re_min=0, re_max=float("inf"), mach_max=float("inf")):
    log.info(f"Parsing: {filepath.name}")
    try:
        wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
    except Exception as e:
        log.error(f"  Cannot open workbook: {e}")
        return pd.DataFrame()

    index_map    = read_index(wb)
    airfoil_name = None
    all_frames   = []
    skipped      = 0

    for sheet_name in wb.sheetnames:
        parsed = parse_sheet_name(sheet_name)
        if parsed is None:
            continue

        sheet_idx, re_val, mach_val = parsed

        if not (re_min <= re_val <= re_max):
            skipped += 1
            continue
        if mach_val > mach_max:
            skipped += 1
            continue

        if airfoil_name is None:
            airfoil_name = extract_airfoil_name(wb, sheet_name, filepath.stem.upper())

        ws         = wb[sheet_name]
        datasource = index_map.get(sheet_idx, "Unknown")
        df_sheet   = parse_data_sheet(ws)

        if df_sheet is None or df_sheet.empty:
            log.warning(f"  Sheet '{sheet_name}': no data extracted")
            continue

        df_sheet.insert(0, "sheet_index",  sheet_idx)
        df_sheet.insert(0, "datasource",   datasource)
        df_sheet.insert(0, "Mach",         mach_val)
        df_sheet.insert(0, "Re",           re_val)
        df_sheet.insert(0, "airfoil_name", airfoil_name)
        df_sheet.insert(0, "source_file",  filepath.name)
        all_frames.append(df_sheet)

    wb.close()

    if not all_frames:
        log.warning(f"  No data extracted from {filepath.name} (skipped {skipped} sheets)")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    log.info(f"  {len(all_frames)} sheets parsed | {len(combined)} rows | {skipped} sheets skipped")
    return combined


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(df_polars, df_stall):
    sep = "=" * 72
    log.info(sep)
    log.info("DATASET SUMMARY")
    log.info(sep)
    log.info(f"Total polar rows     : {len(df_polars):,}")
    log.info(f"Total stall rows     : {len(df_stall):,}")
    log.info(f"Unique airfoils      : {df_polars['airfoil_name'].nunique()}")
    log.info(f"Unique Re values     : {sorted(df_polars['Re'].unique().tolist())}")
    log.info(f"Unique Mach values   : {sorted(df_polars['Mach'].unique().tolist())}")
    log.info(f"Alpha range          : {df_polars['alpha_deg'].min():.1f}° → {df_polars['alpha_deg'].max():.1f}°")
    log.info(f"Cl range             : {df_polars['Cl'].min():.4f} → {df_polars['Cl'].max():.4f}")
    log.info(f"Cd range             : {df_polars['Cd'].min():.6f} → {df_polars['Cd'].max():.6f}")
    log.info("")
    log.info("Airfoils parsed:")
    for name in sorted(df_polars["airfoil_name"].unique()):
        n_rows = len(df_polars[df_polars["airfoil_name"] == name])
        n_cond = df_polars[df_polars["airfoil_name"] == name][["Re", "Mach"]].drop_duplicates().shape[0]
        log.info(f"  {name:<30s}  {n_rows:>6,} rows  |  {n_cond:>3} Re/Mach conditions")
    log.info("")
    log.info("Data sources:")
    for src, cnt in df_polars["datasource"].value_counts().items():
        log.info(f"  {str(src)[:55]:<55s}  {cnt:>6,} rows")
    if not df_stall.empty:
        log.info("")
        log.info("Stall margin statistics (deg):")
        sm = df_stall["stall_margin_deg"].describe()
        log.info(f"  mean={sm['mean']:.3f}  std={sm['std']:.3f}  min={sm['min']:.3f}  max={sm['max']:.3f}")
    log.info(sep)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",     type=str,   default=None)
    parser.add_argument("--re-min",   type=float, default=0)
    parser.add_argument("--re-max",   type=float, default=float("inf"))
    parser.add_argument("--mach-max", type=float, default=float("inf"))
    parser.add_argument("--no-stall", action="store_true")
    args = parser.parse_args()

    log.info("BigFoil Parser started")
    log.info(f"  RAW_DIR  : {RAW_DIR}")
    log.info(f"  PROC_DIR : {PROC_DIR}")
    log.info(f"  Re range : {args.re_min:.0f} → {args.re_max:.0f}")
    log.info(f"  Mach max : {args.mach_max}")
    log.info("")

    if not RAW_DIR.exists():
        log.error(f"RAW_DIR does not exist: {RAW_DIR}")
        log.error("Check that the path above matches where your .xlsx files are saved.")
        sys.exit(1)

    if args.file:
        targets = [RAW_DIR / args.file]
        if not targets[0].exists():
            log.error(f"File not found: {targets[0]}")
            sys.exit(1)
    else:
        targets = sorted(RAW_DIR.glob("*.xlsx"))
        if not targets:
            log.error(f"No .xlsx files found in {RAW_DIR}")
            sys.exit(1)
        log.info(f"Found {len(targets)} .xlsx file(s): {[t.name for t in targets]}")

    all_polars = []
    failed     = []

    for fp in targets:
        try:
            df = parse_excel_file(fp, args.re_min, args.re_max, args.mach_max)
            if not df.empty:
                all_polars.append(df)
        except Exception:
            log.error(f"  FAILED: {fp.name}\n{traceback.format_exc()}")
            failed.append(fp.name)

    if not all_polars:
        log.error("No data extracted from any file. Exiting.")
        sys.exit(1)

    df_polars = pd.concat(all_polars, ignore_index=True)
    df_polars.drop_duplicates(
        subset=["airfoil_name", "Re", "Mach", "alpha_deg", "datasource"],
        inplace=True,
    )
    df_polars.reset_index(drop=True, inplace=True)

    df_polars.to_csv(OUTPUT_POLARS, index=False, float_format="%.6f")
    log.info(f"\nPolars CSV saved: {OUTPUT_POLARS}  ({len(df_polars):,} rows)")

    df_stall = pd.DataFrame()
    if not args.no_stall:
        log.info("Computing stall margins...")
        df_stall = compute_stall_margins(df_polars)
        df_stall.to_csv(OUTPUT_STALL, index=False, float_format="%.6f")
        log.info(f"Stall margins CSV saved: {OUTPUT_STALL}  ({len(df_stall):,} rows)")

    print_summary(df_polars, df_stall)

    if failed:
        log.warning(f"\n{len(failed)} file(s) FAILED: {failed}")

    log.info(f"\nDone. Log: {LOG_FILE}")


if __name__ == "__main__":
    main()
