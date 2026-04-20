"""
Download gmina-level economic indicators from BDL GUS REST API (v1).

API docs: https://bdl.stat.gov.pl/api/v1/swagger

Key variable IDs (numeric BDL IDs, verified 2026-04-16):
    76973  – dochody budżetów gmin na 1 mieszkańca (PLN), subject P2627, lata 2002-2024
    76976  – dochody własne budżetów gmin na 1 mieszkańca (PLN), subject P2627, lata 2002-2024
    72305  – ludność ogółem (osoba), subject P2137, lata 1995-2025
    76049  – udziały gmin w podatkach PIT+CIT łącznie (PLN), subject P2622, lata 1995-2024

Usage:
    python -m src.data.download_economic
    python -m src.data.download_economic --force
    python -m src.data.download_economic --list-variables   # explore BDL subjects
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ECON_DIR = PROJECT_ROOT / "data" / "raw" / "economic"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

load_dotenv(PROJECT_ROOT / ".env")

BDL_BASE_URL = "https://bdl.stat.gov.pl/api/v1"
BDL_PAGE_SIZE = 100  # max allowed by API
UNIT_LEVEL_GMINA = 6  # BDL unit level for gminas

# Variable IDs to download (numeric BDL IDs, verified 2026-04-16)
VARIABLES: dict[str, str] = {
    "76973": "dochody_per_capita",        # dochody budżetów gmin na 1 mieszkańca (PLN), 2002-2024
    "76976": "dochody_wlasne_per_capita", # dochody własne gmin na 1 mieszkańca (PLN), 2002-2024
    "72305": "ludnosc",                   # ludność ogółem (osoba), 1995-2025
}

# Optional PIT/CIT udziały variable — available at gmina level from 1995
# Set to None to skip; use "76049" to include
VARIABLE_PIT_ID: Optional[str] = "76049"  # udziały gmin w podatkach PIT+CIT łącznie (PLN)

YEAR_FROM = 2000
YEAR_TO = 2023

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
# BDL API client
# ---------------------------------------------------------------------------

def _build_headers() -> dict[str, str]:
    """Return request headers, injecting BDL_API_KEY if available."""
    headers = {"Accept": "application/json"}
    api_key = os.getenv("BDL_API_KEY")
    if api_key:
        headers["X-ClientId"] = api_key
    return headers


def _get(endpoint: str, params: Optional[dict] = None, retries: int = 3) -> dict:
    """GET *endpoint* from BDL API with retry/backoff.

    Parameters
    ----------
    endpoint:
        Relative path, e.g. ``/data/by-variable/P2137``.
    params:
        Query parameters.
    retries:
        Number of retry attempts on transient errors.

    Returns
    -------
    dict
        Parsed JSON response.

    Raises
    ------
    requests.HTTPError
        On non-2xx responses after all retries.
    """
    url = f"{BDL_BASE_URL}{endpoint}"
    headers = _build_headers()
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            log.warning("Request failed (%s), retry %d/%d in %ds…", exc, attempt, retries, wait)
            time.sleep(wait)
    return {}  # unreachable


def get_bdl_variable(
    variable_id: str,
    year_from: int,
    year_to: int,
    unit_level: int = UNIT_LEVEL_GMINA,
) -> pd.DataFrame:
    """Fetch all gmina rows for *variable_id* across *year_from*–*year_to*.

    Handles BDL API pagination transparently.

    Parameters
    ----------
    variable_id:
        BDL variable identifier, e.g. ``"P2137"``.
    year_from:
        First year of the range (inclusive).
    year_to:
        Last year of the range (inclusive).
    unit_level:
        BDL territorial unit level: 6 = gmina, 5 = powiat, 4 = województwo.

    Returns
    -------
    pd.DataFrame
        Columns: unit_id (str), unit_name (str), year (int), value (float),
        variable_id (str).
    """
    log.info("Fetching BDL variable %s  years %d–%d  level %d", variable_id, year_from, year_to, unit_level)
    records: list[dict] = []
    page = 0

    with tqdm(desc=f"BDL {variable_id}", unit="pages", leave=False) as bar:
        while True:
            params = {
                "unit-level": unit_level,
                "year": list(range(year_from, year_to + 1)),
                "page": page,
                "page-size": BDL_PAGE_SIZE,
                "format": "json",
                "lang": "pl",
            }
            try:
                data = _get(f"/data/by-variable/{variable_id}", params=params)
            except requests.HTTPError as exc:
                log.error("HTTP error fetching %s page %d: %s", variable_id, page, exc)
                break

            results = data.get("results", [])
            if not results:
                break

            for unit in results:
                unit_id = str(unit.get("id", "")).strip()
                unit_name = unit.get("name", "")
                for val_entry in unit.get("values", []):
                    yr = int(val_entry.get("year", 0))
                    val = val_entry.get("val")
                    records.append(
                        {
                            "unit_id": unit_id,
                            "unit_name": unit_name,
                            "year": yr,
                            "value": float(val) if val is not None else None,
                            "variable_id": variable_id,
                        }
                    )

            bar.update(1)
            total_records = data.get("totalRecords", 0)
            fetched_so_far = (page + 1) * BDL_PAGE_SIZE
            if fetched_so_far >= total_records or len(results) < BDL_PAGE_SIZE:
                break
            page += 1

    df = pd.DataFrame(records)
    log.info("  → %d rows for variable %s", len(df), variable_id)
    return df


# ---------------------------------------------------------------------------
# TERYT normalisation for BDL unit IDs
# ---------------------------------------------------------------------------

def _bdl_unit_to_teryt(unit_id: str) -> str:
    """Convert a BDL gmina unit_id to a 7-digit TERYT code.

    BDL gmina IDs are typically 7-digit strings matching TERYT gmina codes
    (powiat_id = 4 digits, gmina appended).  This function zero-pads to 7.

    Parameters
    ----------
    unit_id:
        BDL unit identifier string.

    Returns
    -------
    str
        7-character TERYT gmina code.
    """
    s = str(unit_id).strip().lstrip("0")
    # BDL often returns codes without leading zeros; TERYT gmina = 7 digits
    return s.zfill(7)


# ---------------------------------------------------------------------------
# Relative PIT / income index
# ---------------------------------------------------------------------------

def compute_relative_income(
    df: pd.DataFrame,
    value_col: str = "dochody_per_capita",
    pop_col: str = "ludnosc",
) -> pd.DataFrame:
    """Add a population-weighted relative income column.

    Computes ``pit_relative_i_t = value_i_t / mean_weighted_t`` where
    ``mean_weighted_t`` is the population-weighted mean across all gminas in
    year *t*.

    Parameters
    ----------
    df:
        Wide panel DataFrame with columns including *value_col*, *pop_col*,
        *year*, and *teryt*.
    value_col:
        Name of the per-capita income column to normalise.
    pop_col:
        Name of the population column used as weights.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an additional column
        ``{value_col}_relative``.
    """
    def _wm_scalar(group: pd.DataFrame) -> float:
        mask = group[value_col].notna() & group[pop_col].notna() & (group[pop_col] > 0)
        g = group[mask]
        if g.empty:
            return float("nan")
        return (g[value_col] * g[pop_col]).sum() / g[pop_col].sum()

    nat_mean_col = f"{value_col}_nat_mean"
    means = df.groupby("year").apply(_wm_scalar, include_groups=False).rename(nat_mean_col).reset_index()
    df = df.merge(means, on="year", how="left")
    df[f"{value_col}_relative"] = df[value_col] / df[f"{value_col}_nat_mean"]
    return df


# ---------------------------------------------------------------------------
# List BDL subjects (exploratory helper)
# ---------------------------------------------------------------------------

def list_bdl_subjects(parent_id: Optional[str] = None) -> pd.DataFrame:
    """Return BDL subject tree (top-level or children of *parent_id*).

    Parameters
    ----------
    parent_id:
        If given, fetch children of this subject node; else fetch roots.

    Returns
    -------
    pd.DataFrame
        Columns: id, name, hasVariables.
    """
    params: dict = {"page": 0, "page-size": 100, "lang": "pl"}
    if parent_id:
        params["parent-id"] = parent_id
    data = _get("/subjects", params=params)
    rows = []
    for item in data.get("results", []):
        rows.append({"id": item["id"], "name": item["name"], "hasVariables": item.get("hasVariables", False)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(force: bool = False) -> None:
    """Download all configured BDL variables and build the economic panel.

    Parameters
    ----------
    force:
        Re-download raw files even if they exist locally.
    """
    RAW_ECON_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    long_frames: list[pd.DataFrame] = []

    effective_vars = dict(VARIABLES)
    if VARIABLE_PIT_ID:
        effective_vars[VARIABLE_PIT_ID] = "pit_per_capita"

    for var_id, label in effective_vars.items():
        # Check cached parquet (one per variable)
        cache_path = RAW_ECON_DIR / f"bdl_{var_id}_{YEAR_FROM}_{YEAR_TO}.parquet"
        if cache_path.exists() and not force:
            log.info("Loading cached %s from %s", var_id, cache_path.name)
            df_var = pd.read_parquet(cache_path)
        else:
            df_var = get_bdl_variable(var_id, YEAR_FROM, YEAR_TO)
            df_var.to_parquet(cache_path, index=False)
            log.info("Saved raw cache: %s", cache_path.name)

        df_var = df_var.rename(columns={"value": label})
        df_var["teryt"] = df_var["unit_id"].apply(_bdl_unit_to_teryt)
        long_frames.append(df_var[["teryt", "unit_name", "year", label]])

    if not long_frames:
        log.error("No economic data downloaded.")
        return

    # Merge all variables on (teryt, year)
    panel = long_frames[0]
    for df_extra in long_frames[1:]:
        merge_cols = ["teryt", "year"]
        # Include unit_name only from first frame
        extra_cols = [c for c in df_extra.columns if c not in ["unit_name", *merge_cols]]
        panel = panel.merge(df_extra[merge_cols + extra_cols], on=merge_cols, how="outer")

    # Compute relative income if both income and population columns are present
    if "dochody_per_capita" in panel.columns and "ludnosc" in panel.columns:
        panel = compute_relative_income(panel, "dochody_per_capita", "ludnosc")
    if "dochody_wlasne_per_capita" in panel.columns and "ludnosc" in panel.columns:
        panel = compute_relative_income(panel, "dochody_wlasne_per_capita", "ludnosc")

    panel = panel.sort_values(["teryt", "year"]).reset_index(drop=True)
    out_path = INTERIM_DIR / "economic_panel.parquet"
    panel.to_parquet(out_path, index=False)
    log.info(
        "Economic panel saved: %s  (%d rows, %d gminas, years %d–%d)",
        out_path,
        len(panel),
        panel["teryt"].nunique(),
        panel["year"].min() if len(panel) else 0,
        panel["year"].max() if len(panel) else 0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download BDL GUS economic indicators.")
    parser.add_argument("--force", action="store_true", help="Re-download all data.")
    parser.add_argument(
        "--list-variables",
        action="store_true",
        help="Print top-level BDL subject tree and exit.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.list_variables:
        subjects = list_bdl_subjects()
        print(subjects.to_string(index=False))
        raise SystemExit(0)

    run(force=args.force)
