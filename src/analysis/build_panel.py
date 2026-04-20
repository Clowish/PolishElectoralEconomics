"""
Build the final analytical panel for the Polish electoral economics project.

One observation = one gmina × one election year.
Key: (teryt6, year) where year ∈ {2001, 2005, 2007, 2011, 2015, 2019, 2023}

Usage:
    uv run python3 -m src.analysis.build_panel
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR  = PROJECT_ROOT / "data" / "interim"
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
OUT_DIR      = PROJECT_ROOT / "data" / "processed"

ELECTIONS_PATH  = INTERIM_DIR / "elections_panel_raw.parquet"
ECONOMIC_PATH   = INTERIM_DIR / "economic_panel.parquet"
CROSSWALK_PATH  = INTERIM_DIR / "crosswalk_normalized.parquet"
GEOJSON_PATH    = RAW_DIR / "spatial" / "gminy.geojson"
OUT_PATH        = OUT_DIR / "panel_final.parquet"

ELECTION_YEARS = [2001, 2005, 2007, 2011, 2015, 2019, 2023]

# Mapping from category value to column stem
CATEGORY_COLS = {
    "populist_right":      "populist_right",
    "mainstream_right":    "mainstream_right",
    "post_communist_left": "post_communist_left",
    "liberal_center":      "liberal_center",
    "other":               "other",
}

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
# Step 1: Electoral variables
# ---------------------------------------------------------------------------

def build_electoral_block(elections_raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate party-level vote columns into ideological category totals.

    For each row (gmina × year) sum all `votes_N - Party` columns whose
    corresponding `category_N - Party` value matches the target category.
    """
    log.info("Building electoral block …")

    df = elections_raw.copy()

    # Identify (number, party_name) → category mapping from category_* columns
    # Each column name: "category_N - Party Name"
    cat_cols   = [c for c in df.columns if c.startswith("category_")]
    votes_cols = [c for c in df.columns if c.startswith("votes_")]

    # Build mapping: votes_column_name → category string
    #
    # Two legitimate formats exist:
    #   2001–2015: "votes_N - Party Name"   (N = digit)
    #   2019–2023: "votes_KOMITET WYBORCZY …" (ALL-CAPS committee names)
    #
    # Auxiliary metadata columns (e.g. "votes_Karty ważne", "votes_Nr okręgu")
    # use Title Case and must be excluded because they contain ballot-count
    # metadata that would inflate category totals.
    import re as _re

    def _is_party_col(colname: str) -> bool:
        suffix = colname[len("votes_"):]
        if not suffix:
            return False
        # Numbered format: "5 - Prawo i Sprawiedliwość"
        if _re.match(r"^\d+\s*-\s*", suffix):
            return True
        # Full-name format: first word is ALL-CAPS (Polish alphabet included)
        # e.g. "KOMITET WYBORCZY …", "KOALICYJNY KOMITET …"
        # Auxiliary columns use Title Case: "Karty", "Głosy", "Nr", "Odd."
        first = suffix.split()[0]
        return len(first) >= 2 and first == first.upper() and first.replace("-", "").isalpha()

    votes_to_cat: dict[str, str] = {}
    for cc in cat_cols:
        suffix = cc[len("category_"):]           # "N - Party Name"
        vc = "votes_" + suffix
        if vc in df.columns and _is_party_col(vc):
            # category value is constant per (vc, election year) pair;
            # take the mode to handle any stray NaN
            mode = df[cc].mode(dropna=True)
            if not mode.empty:
                votes_to_cat[vc] = str(mode.iloc[0])

    log.info("  Mapped %d vote columns to categories", len(votes_to_cat))

    # Group vote columns by category
    cat_vote_cols: dict[str, list[str]] = {cat: [] for cat in CATEGORY_COLS}
    unmapped = []
    for vc, cat in votes_to_cat.items():
        if cat in cat_vote_cols:
            cat_vote_cols[cat].append(vc)
        else:
            unmapped.append((vc, cat))
    if unmapped:
        log.warning("  %d vote columns with unrecognised category: %s",
                    len(unmapped), unmapped[:5])

    # Sum votes per category, treating NaN as 0
    result = df[["teryt", "gmina_nazwa", "year", "waznych", "frekwencja", "uprawnionych"]].copy()
    result = result.rename(columns={"teryt": "teryt6", "waznych": "votes_total"})

    for cat, col_list in cat_vote_cols.items():
        if col_list:
            result[f"votes_{cat}"] = df[col_list].fillna(0).sum(axis=1)
        else:
            log.warning("  No vote columns found for category '%s'", cat)
            result[f"votes_{cat}"] = np.nan

    # Percentage shares
    vt = result["votes_total"].replace(0, np.nan)
    for cat in CATEGORY_COLS:
        result[f"pct_{cat}"] = result[f"votes_{cat}"] / vt * 100

    log.info("  Electoral block: %d rows × %d cols", *result.shape)
    return result


# ---------------------------------------------------------------------------
# Step 2: Economic variables  (pre-election averages)
# ---------------------------------------------------------------------------

def build_economic_block(economic: pd.DataFrame) -> pd.DataFrame:
    """For each election year T compute the T-2/T-1 average of economic vars.

    Returns a DataFrame indexed by (teryt6, year_election).
    """
    log.info("Building economic block (T-2 / T-1 averages) …")

    eco = economic.rename(columns={"teryt": "teryt6"}).copy()

    records = []
    for t in ELECTION_YEARS:
        # Use years T-2 and T-1 for flow variables; T-1 for stock (population)
        eco_window = eco[eco["year"].isin([t - 2, t - 1])].copy()
        eco_t1     = eco[eco["year"] == t - 1].copy()

        # Average of T-2, T-1 per gmina
        avg = (
            eco_window
            .groupby("teryt6")
            .agg(
                dochody_pc=("dochody_per_capita", "mean"),
                dochody_wlasne_pc=("dochody_wlasne_per_capita", "mean"),
                pit_pc=("pit_per_capita", "mean"),
            )
            .reset_index()
        )
        # Population from T-1
        pop = eco_t1[["teryt6", "ludnosc", "unit_name"]].drop_duplicates("teryt6")

        block = avg.merge(pop, on="teryt6", how="outer")
        block["year"] = t
        records.append(block)

    econ_panel = pd.concat(records, ignore_index=True)
    log.info("  Economic block: %d rows × %d cols", *econ_panel.shape)
    return econ_panel


# ---------------------------------------------------------------------------
# Step 3: Relative income indices (population-weighted national mean)
# ---------------------------------------------------------------------------

def add_relative_indices(panel: pd.DataFrame) -> pd.DataFrame:
    """Add population-weighted relative income / PIT indices per election year."""
    log.info("Computing relative indices …")

    def _wm(g: pd.DataFrame, val_col: str) -> float:
        mask = g[val_col].notna() & g["ludnosc"].notna() & (g["ludnosc"] > 0)
        sub = g[mask]
        if sub.empty:
            return np.nan
        return (sub[val_col] * sub["ludnosc"]).sum() / sub["ludnosc"].sum()

    for val_col, out_col in [
        ("dochody_pc",  "dochody_pc_relative"),
        ("pit_pc",      "pit_pc_relative"),
    ]:
        nat_means = (
            panel.groupby("year")
            .apply(lambda g: _wm(g, val_col), include_groups=False)
            .rename("nat_mean")
            .reset_index()
        )
        panel = panel.merge(nat_means, on="year", how="left")
        panel[out_col] = panel[val_col] / panel["nat_mean"]
        panel = panel.drop(columns=["nat_mean"])

    return panel


# ---------------------------------------------------------------------------
# Step 4: gmina type from crosswalk / GeoJSON
# ---------------------------------------------------------------------------

def add_gmina_type(panel: pd.DataFrame,
                   crosswalk: pd.DataFrame,
                   geojson_path: Path) -> pd.DataFrame:
    """Attach typ_gminy (1/2/3) from crosswalk; fill gaps from GeoJSON."""
    log.info("Adding gmina type …")

    # Crosswalk: teryt6_2023 → typ_gminy
    cw_type = (
        crosswalk[["teryt6_2023", "typ_gminy"]]
        .dropna(subset=["teryt6_2023"])
        .drop_duplicates("teryt6_2023")
        .rename(columns={"teryt6_2023": "teryt6"})
    )
    cw_type["typ_gminy"] = pd.to_numeric(cw_type["typ_gminy"], errors="coerce").astype("Int64")

    panel = panel.merge(cw_type, on="teryt6", how="left")

    # Fill gaps from GeoJSON
    missing_mask = panel["typ_gminy"].isna()
    if missing_mask.any():
        gdf = gpd.read_file(geojson_path)[["teryt6", "gmina_type"]]
        type_map = {
            "Gmina(urban)":                1,
            "Miastonaprawachpowiatu":       1,
            "Gmina(rural)":                2,
            "Gmina(urban-rural)":          3,
        }
        gdf["typ_gminy_geo"] = gdf["gmina_type"].map(type_map)
        gdf = gdf[["teryt6", "typ_gminy_geo"]].drop_duplicates("teryt6")

        panel = panel.merge(gdf, on="teryt6", how="left")
        panel.loc[missing_mask, "typ_gminy"] = panel.loc[missing_mask, "typ_gminy_geo"]
        panel = panel.drop(columns=["typ_gminy_geo"])

    n_filled = missing_mask.sum() - panel["typ_gminy"].isna().sum()
    log.info("  typ_gminy: %d NaN remaining (filled %d from GeoJSON)",
             panel["typ_gminy"].isna().sum(), n_filled)

    return panel


# ---------------------------------------------------------------------------
# Step 5: First-difference variables
# ---------------------------------------------------------------------------

def add_first_differences(panel: pd.DataFrame) -> pd.DataFrame:
    """Add delta columns: change since previous election year."""
    log.info("Computing first differences …")

    diff_pairs = [
        ("pct_populist_right",    "delta_pct_populist_right"),
        ("dochody_pc_relative",   "delta_dochody_pc_relative"),
        ("pit_pc_relative",       "delta_pit_pc_relative"),
        ("frekwencja",            "delta_frekwencja"),
    ]

    panel = panel.sort_values(["teryt6", "year"]).copy()

    for src, dst in diff_pairs:
        panel[dst] = panel.groupby("teryt6")[src].diff()

    return panel


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(panel: pd.DataFrame) -> None:
    log.info("Validating panel …")

    # Shape
    n_gminas = panel["teryt6"].nunique()
    n_years  = panel["year"].nunique()
    log.info("  Shape: %s  |  gminas: %d  |  years: %d", panel.shape, n_gminas, n_years)

    expected_max = n_gminas * n_years
    log.info("  Expected max rows (balanced): %d  |  actual: %d  (%.1f%%)",
             expected_max, len(panel), 100 * len(panel) / expected_max)

    # NaN counts
    nan_counts = panel.isna().sum()
    problematic = nan_counts[nan_counts > 0]
    if not problematic.empty:
        log.info("  NaN counts:\n%s", problematic.to_string())

    # Sanity: pct columns should sum to ~100 (within tolerance)
    pct_sum = panel[[f"pct_{c}" for c in CATEGORY_COLS]].sum(axis=1)
    off = (pct_sum - 100).abs()
    # Ignore rows where any pct is NaN
    valid_rows = panel[[f"pct_{c}" for c in CATEGORY_COLS]].notna().all(axis=1)
    if valid_rows.any():
        max_off = off[valid_rows].max()
        mean_off = off[valid_rows].mean()
        log.info("  pct sum deviation: max=%.4f  mean=%.4f", max_off, mean_off)

    # Delta: first year should be all NaN
    first_year = ELECTION_YEARS[0]
    n_nan_delta_first = panel.loc[panel["year"] == first_year, "delta_pct_populist_right"].isna().sum()
    n_first = (panel["year"] == first_year).sum()
    log.info("  delta NaN in year %d: %d/%d (expected all NaN)", first_year, n_nan_delta_first, n_first)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=== Loading source data ===")
    elections_raw = pd.read_parquet(ELECTIONS_PATH)
    economic      = pd.read_parquet(ECONOMIC_PATH)
    crosswalk     = pd.read_parquet(CROSSWALK_PATH)
    log.info("  elections_raw : %s", elections_raw.shape)
    log.info("  economic      : %s", economic.shape)
    log.info("  crosswalk     : %s", crosswalk.shape)

    # ── Step 1: Electoral block ──────────────────────────────────────────────
    electoral = build_electoral_block(elections_raw)

    # ── Step 2: Economic block ───────────────────────────────────────────────
    econ = build_economic_block(economic)

    # ── Step 3: Merge electoral + economic ──────────────────────────────────
    log.info("Merging electoral + economic …")
    panel = electoral.merge(econ, on=["teryt6", "year"], how="left")
    log.info("  After merge: %s", panel.shape)

    # ── Step 4: Relative indices ─────────────────────────────────────────────
    panel = add_relative_indices(panel)

    # ── Step 5: Gmina type ───────────────────────────────────────────────────
    panel = add_gmina_type(panel, crosswalk, GEOJSON_PATH)

    # ── Step 6: First differences ────────────────────────────────────────────
    panel = add_first_differences(panel)

    # ── Reorder columns ──────────────────────────────────────────────────────
    id_cols   = ["teryt6", "gmina_nazwa", "year", "typ_gminy"]
    vote_cols = (
        [f"votes_{c}" for c in [*CATEGORY_COLS, "total"]] +
        [f"pct_{c}"   for c in CATEGORY_COLS] +
        ["frekwencja", "uprawnionych"]
    )
    econ_cols  = ["dochody_pc", "dochody_wlasne_pc", "pit_pc", "ludnosc",
                  "dochody_pc_relative", "pit_pc_relative"]
    delta_cols = ["delta_pct_populist_right", "delta_dochody_pc_relative",
                  "delta_pit_pc_relative", "delta_frekwencja"]
    extra_cols = [c for c in panel.columns
                  if c not in id_cols + vote_cols + econ_cols + delta_cols]

    ordered = id_cols + vote_cols + econ_cols + delta_cols + extra_cols
    ordered = [c for c in ordered if c in panel.columns]
    panel   = panel[ordered].sort_values(["teryt6", "year"]).reset_index(drop=True)

    # ── Validation ───────────────────────────────────────────────────────────
    validate(panel)

    # ── Save ─────────────────────────────────────────────────────────────────
    panel.to_parquet(OUT_PATH, index=False)
    log.info("Saved: %s  (%d rows, %d cols)", OUT_PATH, *panel.shape)

    return panel


if __name__ == "__main__":
    panel = main()

    print("\n" + "=" * 65)
    print("PANEL SUMMARY")
    print("=" * 65)
    print(f"\nShape: {panel.shape}")
    print(f"\nDtypes:\n{panel.dtypes.to_string()}")

    key_cols = [
        "teryt6", "year",
        "pct_populist_right", "dochody_pc_relative",
        "delta_pct_populist_right", "delta_dochody_pc_relative",
    ]
    print(f"\nDescribe (key columns):\n{panel[key_cols].describe().to_string()}")

    print(f"\nNaN counts (non-zero only):")
    nan_counts = panel.isna().sum()
    nan_nonzero = nan_counts[nan_counts > 0]
    print(nan_nonzero.to_string() if not nan_nonzero.empty else "  none")

    print(f"\nBochnia (teryt6=120101) przez lata:")
    bochnia = panel[panel["teryt6"] == "120101"].sort_values("year")
    show = ["year", "pct_populist_right", "dochody_pc_relative",
            "delta_pct_populist_right", "delta_dochody_pc_relative"]
    print(bochnia[show].to_string(index=False))
