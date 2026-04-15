"""
Harmonise gmina TERYT codes across the 1999–2023 panel.

Source crosswalk: IBS (Institute for Structural Research) KTS-6 file
https://ibs.org.pl/en/resources/crosswalks-for-polish-counties-and-municipalities-kts-5-kts-6-1999-2024/

The crosswalk maps each historical TERYT code to a canonical (stable) unit ID,
flagging mergers, splits, and reclassifications.

Usage:
    python -m src.data.harmonize_teryt
    python -m src.data.harmonize_teryt --force
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CROSSWALK_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

CROSSWALK_URL = (
    "https://ibs.org.pl/app/uploads/2024/01/"
    "kts6_1999_2024.xlsx"
)
CROSSWALK_LOCAL = RAW_CROSSWALK_DIR / "ibs_kts6_crosswalk.xlsx"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PANEL_YEAR_FROM = 1999
PANEL_YEAR_TO = 2023


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_crosswalk(force: bool = False) -> Path:
    """Download the IBS KTS-6 crosswalk XLSX unless already cached.

    Parameters
    ----------
    force:
        Re-download even if the local file exists.

    Returns
    -------
    Path
        Local path to the downloaded XLSX file.
    """
    if CROSSWALK_LOCAL.exists() and not force:
        log.info("Crosswalk already cached: %s", CROSSWALK_LOCAL.name)
        return CROSSWALK_LOCAL

    log.info("Downloading crosswalk from %s", CROSSWALK_URL)
    RAW_CROSSWALK_DIR.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(CROSSWALK_URL, timeout=120)
        r.raise_for_status()
        CROSSWALK_LOCAL.write_bytes(r.content)
        log.info("Saved crosswalk: %s  (%d KB)", CROSSWALK_LOCAL.name, len(r.content) // 1024)
    except requests.RequestException as exc:
        log.error("Download failed: %s", exc)
        log.info("Manual download instructions:")
        log.info("  1. Visit https://ibs.org.pl/en/resources/crosswalks-for-polish-counties-and-municipalities-kts-5-kts-6-1999-2024/")
        log.info("  2. Download the KTS-6 (gmina level) XLSX file")
        log.info("  3. Save as: %s", CROSSWALK_LOCAL)
        raise
    return CROSSWALK_LOCAL


# ---------------------------------------------------------------------------
# Crosswalk loading
# ---------------------------------------------------------------------------

def _normalize_teryt(code) -> str:
    """Zero-pad a TERYT code to 6 or 7 characters (string-safe).

    Parameters
    ----------
    code:
        Raw code as int, float, or str.

    Returns
    -------
    str
        Zero-padded TERYT string.
    """
    s = re.sub(r"\D", "", str(code).split(".")[0].strip())
    if len(s) in (6, 7):
        return s
    if len(s) < 6:
        return s.zfill(6)
    return s[:7]


def load_crosswalk(path: Path) -> pd.DataFrame:
    """Load and normalise the IBS KTS-6 crosswalk.

    The IBS file contains one row per (historical_code, year) pair, or one
    row per gmina with yearly validity columns — exact format varies by
    version.  This function tries to produce a tidy DataFrame with columns:

        teryt_historical (str), teryt_canonical (str), year_from (int),
        year_to (int), change_type (str), gmina_name (str)

    Parameters
    ----------
    path:
        Path to the IBS XLSX crosswalk file.

    Returns
    -------
    pd.DataFrame
        Normalised crosswalk.
    """
    log.info("Loading crosswalk: %s", path)
    # Try all sheets; pick the one with the most rows
    xl = pd.ExcelFile(path)
    best_df: Optional[pd.DataFrame] = None
    for sheet in xl.sheet_names:
        try:
            df = xl.parse(sheet, dtype=str)
            if best_df is None or len(df) > len(best_df):
                best_df = df
        except Exception as exc:
            log.warning("Could not parse sheet %s: %s", sheet, exc)

    if best_df is None:
        raise ValueError(f"No parseable sheets found in {path}")

    df = best_df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    log.debug("Crosswalk columns: %s", list(df.columns))

    # Identify the canonical ID column (the "stable" code used throughout)
    canonical_candidates = [
        c for c in df.columns
        if any(kw in c for kw in ("canonical", "kts6", "id_stable", "teryt_stable", "kod_stable"))
    ]
    hist_candidates = [
        c for c in df.columns
        if any(kw in c for kw in ("historical", "hist", "teryt", "kod_gminy", "gmkod"))
    ]
    name_candidates = [c for c in df.columns if "nazwa" in c or "name" in c]

    canonical_col = canonical_candidates[0] if canonical_candidates else df.columns[0]
    hist_col = hist_candidates[0] if hist_candidates else df.columns[0]
    name_col = name_candidates[0] if name_candidates else None

    # Year columns: detect columns like "1999", "2000", …, "2023"
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]

    if year_cols:
        # Wide format: each year column contains the valid TERYT code for that year
        records = []
        for _, row in df.iterrows():
            canonical = _normalize_teryt(row[canonical_col]) if pd.notna(row[canonical_col]) else None
            if not canonical:
                continue
            name = str(row[name_col]).strip() if name_col else ""
            prev_hist = None
            yr_start = None
            for yc in sorted(year_cols):
                yr = int(yc)
                hist_code = _normalize_teryt(row[yc]) if pd.notna(row[yc]) else None
                if hist_code != prev_hist:
                    if prev_hist is not None and yr_start is not None:
                        records.append({
                            "teryt_historical": prev_hist,
                            "teryt_canonical": canonical,
                            "year_from": yr_start,
                            "year_to": yr - 1,
                            "gmina_name": name,
                        })
                    prev_hist = hist_code
                    yr_start = yr
            # Close last interval
            if prev_hist is not None and yr_start is not None:
                records.append({
                    "teryt_historical": prev_hist,
                    "teryt_canonical": canonical,
                    "year_from": yr_start,
                    "year_to": int(year_cols[-1]),
                    "gmina_name": name,
                })
        crosswalk = pd.DataFrame(records)
    else:
        # Long format: each row is a (historical, canonical, year_from, year_to) tuple
        year_from_col = next((c for c in df.columns if "from" in c or "od" in c or "start" in c), None)
        year_to_col = next((c for c in df.columns if "to" in c or "do" in c or "end" in c), None)

        crosswalk = pd.DataFrame({
            "teryt_historical": df[hist_col].apply(_normalize_teryt),
            "teryt_canonical": df[canonical_col].apply(_normalize_teryt),
            "year_from": pd.to_numeric(df[year_from_col], errors="coerce").fillna(PANEL_YEAR_FROM).astype(int)
            if year_from_col else PANEL_YEAR_FROM,
            "year_to": pd.to_numeric(df[year_to_col], errors="coerce").fillna(PANEL_YEAR_TO).astype(int)
            if year_to_col else PANEL_YEAR_TO,
            "gmina_name": df[name_col].fillna("").astype(str) if name_col else "",
        })

    crosswalk = crosswalk.dropna(subset=["teryt_historical", "teryt_canonical"])
    log.info("Crosswalk loaded: %d records", len(crosswalk))
    return crosswalk


# ---------------------------------------------------------------------------
# Stable panel units
# ---------------------------------------------------------------------------

def build_stable_panel_units(crosswalk_df: pd.DataFrame) -> pd.DataFrame:
    """Identify stable gminas and build a canonical unit list.

    A gmina is "stable" if its TERYT code never changed between
    PANEL_YEAR_FROM and PANEL_YEAR_TO.

    Parameters
    ----------
    crosswalk_df:
        Output of :func:`load_crosswalk`.

    Returns
    -------
    pd.DataFrame
        Columns: teryt_canonical, gmina_name, is_stable (bool),
        n_historical_codes (int).
    """
    # For each canonical code, how many distinct historical codes map to it?
    agg = (
        crosswalk_df.groupby("teryt_canonical")
        .agg(
            gmina_name=("gmina_name", "first"),
            n_historical_codes=("teryt_historical", "nunique"),
        )
        .reset_index()
    )
    agg["is_stable"] = (
        (agg["n_historical_codes"] == 1)
        & (agg["teryt_canonical"] == agg["teryt_canonical"])
    )
    # Stable also requires the historical code to equal the canonical code
    self_map = crosswalk_df[
        crosswalk_df["teryt_historical"] == crosswalk_df["teryt_canonical"]
    ]["teryt_canonical"].unique()
    agg.loc[~agg["teryt_canonical"].isin(self_map), "is_stable"] = False

    log.info(
        "Stable gminas: %d / %d total canonical units",
        agg["is_stable"].sum(),
        len(agg),
    )
    return agg


# ---------------------------------------------------------------------------
# Lookup function
# ---------------------------------------------------------------------------

def _build_lookup(crosswalk_df: pd.DataFrame) -> dict[tuple[str, int], str]:
    """Build (teryt_historical, year) → teryt_canonical lookup dict.

    Parameters
    ----------
    crosswalk_df:
        Normalised crosswalk DataFrame.

    Returns
    -------
    dict
        Keys are ``(historical_code, year)`` tuples; values are canonical codes.
    """
    lookup: dict[tuple[str, int], str] = {}
    for _, row in crosswalk_df.iterrows():
        hist = row["teryt_historical"]
        canon = row["teryt_canonical"]
        for yr in range(int(row["year_from"]), int(row["year_to"]) + 1):
            lookup[(hist, yr)] = canon
    return lookup


_LOOKUP_CACHE: Optional[dict[tuple[str, int], str]] = None
_CROSSWALK_CACHE: Optional[pd.DataFrame] = None


def _ensure_lookup() -> dict[tuple[str, int], str]:
    """Lazily load the crosswalk lookup into module-level cache."""
    global _LOOKUP_CACHE, _CROSSWALK_CACHE
    if _LOOKUP_CACHE is None:
        if not CROSSWALK_LOCAL.exists():
            raise FileNotFoundError(
                f"Crosswalk file not found: {CROSSWALK_LOCAL}\n"
                "Run: python -m src.data.harmonize_teryt  to download it."
            )
        _CROSSWALK_CACHE = load_crosswalk(CROSSWALK_LOCAL)
        _LOOKUP_CACHE = _build_lookup(_CROSSWALK_CACHE)
    return _LOOKUP_CACHE


def harmonize_teryt_code(code: str, year: int) -> str:
    """Convert a historical TERYT gmina code to the canonical panel code.

    Parameters
    ----------
    code:
        TERYT code as it appears in the data for *year*.
    year:
        Calendar year of the observation.

    Returns
    -------
    str
        Canonical (stable) TERYT code used throughout the panel.  Falls back
        to the normalised input code if no mapping is found.
    """
    norm = _normalize_teryt(code)
    lookup = _ensure_lookup()
    return lookup.get((norm, year), norm)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def generate_report(crosswalk_df: pd.DataFrame, stable_df: pd.DataFrame) -> str:
    """Generate a plain-text harmonisation summary report.

    Parameters
    ----------
    crosswalk_df:
        Normalised crosswalk.
    stable_df:
        Output of :func:`build_stable_panel_units`.

    Returns
    -------
    str
        Multi-line report string.
    """
    n_total = len(stable_df)
    n_stable = stable_df["is_stable"].sum()
    n_harmonised = n_total - n_stable
    n_hist = len(crosswalk_df)

    report_lines = [
        "=" * 60,
        "TERYT Harmonisation Report",
        "=" * 60,
        f"Panel period           : {PANEL_YEAR_FROM}–{PANEL_YEAR_TO}",
        f"Total canonical units  : {n_total:>6}",
        f"Stable (unchanged TERYT): {n_stable:>6}  ({100*n_stable/n_total:.1f}%)",
        f"Required harmonisation : {n_harmonised:>6}  ({100*n_harmonised/n_total:.1f}%)",
        f"Historical code records: {n_hist:>6}",
        "=" * 60,
    ]
    # Gminas with most historical codes (mergers/splits)
    top = stable_df.nlargest(10, "n_historical_codes")[["teryt_canonical", "gmina_name", "n_historical_codes"]]
    report_lines.append("Top 10 most-changed gminas:")
    report_lines.append(top.to_string(index=False))
    report_lines.append("=" * 60)

    return "\n".join(report_lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(force: bool = False) -> None:
    """Download crosswalk, build stable units, save outputs.

    Parameters
    ----------
    force:
        Re-download the crosswalk XLSX even if cached.
    """
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    download_crosswalk(force=force)
    crosswalk_df = load_crosswalk(CROSSWALK_LOCAL)

    stable_df = build_stable_panel_units(crosswalk_df)

    # Save outputs
    cw_out = INTERIM_DIR / "teryt_crosswalk.parquet"
    stable_out = INTERIM_DIR / "teryt_stable_units.parquet"
    crosswalk_df.to_parquet(cw_out, index=False)
    stable_df.to_parquet(stable_out, index=False)
    log.info("Saved crosswalk: %s", cw_out)
    log.info("Saved stable units: %s", stable_out)

    report = generate_report(crosswalk_df, stable_df)
    print(report)

    report_path = PROJECT_ROOT / "reports" / "teryt_harmonisation_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    log.info("Report saved: %s", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harmonise TERYT codes using IBS crosswalk.")
    parser.add_argument("--force", action="store_true", help="Re-download the crosswalk file.")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    run(force=args.force)
