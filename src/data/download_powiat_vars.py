"""
Download powiat-level BDL variables and extend the analytical panel.

The script downloads:
    - 60270: stopa bezrobocia rejestrowanego
    - 64428: przeciętne miesięczne wynagrodzenie brutto

Both series are only available at powiat level in BDL. They are transformed
into pre-election T-2/T-1 averages and merged onto the gmina-level election
panel using the first four digits of teryt6 (powiat code).

Usage:
    uv run python3 -m src.data.download_powiat_vars
"""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.download_economic import RAW_ECON_DIR, get_bdl_variable, _get

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CROSSWALK_PATH = PROJECT_ROOT / "data" / "raw" / "crosswalks" / "ibs_teryt_crosswalk_1999_2025.xlsx"

PANEL_PATH = PROCESSED_DIR / "panel_final.parquet"
OUT_PANEL_PATH = PROCESSED_DIR / "panel_final_v2.parquet"

EXISTING_MAPPING_PATH = RAW_ECON_DIR / "bdl_units_mapping.parquet"
POWIAT_MAPPING_PATH = RAW_ECON_DIR / "bdl_powiat_mapping.parquet"

RAW_UNEMPLOYMENT_PATH = RAW_ECON_DIR / "bdl_bezrobocie_powiat.parquet"
RAW_WAGE_PATH = RAW_ECON_DIR / "bdl_wynagrodzenia_powiat.parquet"

YEAR_FROM = 2000
YEAR_TO = 2023
UNIT_LEVEL_POWIAT = 5
BDL_UNIT_ID_WIDTH = 12

VARIABLE_SPECS = {
    "60270": {
        "label": "bezrobocie_powiat",
        "raw_path": RAW_UNEMPLOYMENT_PATH,
        "available_election_years": [2007, 2011, 2015, 2019, 2023],
    },
    "64428": {
        "label": "wynagrodzenie_powiat",
        "raw_path": RAW_WAGE_PATH,
        "available_election_years": [2005, 2007, 2011, 2015, 2019, 2023],
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def normalize_powiat_name(text: str) -> str:
    """Normalize powiat names for robust name-based matching."""

    s = str(text).strip().lower()
    s = s.replace("powiat m. ", "")
    s = s.replace("powiat ", "")
    s = s.replace("miasto na prawach powiatu ", "")
    s = s.replace("miasto ", "")
    s = re.sub(r"\s+(do|od)\s+\d{4}$", "", s)
    s = re.sub(r"\s*\([^)]*\)", "", s)
    s = s.replace("-", " ")
    s = " ".join(s.split())
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return s


def save_raw_variable(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Save raw powiat data in the requested compact schema."""

    raw = (
        df.rename(columns={"unit_id": "bdl_unit_id"})[["bdl_unit_id", "year", "value"]]
        .copy()
        .sort_values(["bdl_unit_id", "year"])
        .reset_index(drop=True)
    )
    raw["bdl_unit_id"] = raw["bdl_unit_id"].astype(str).str.zfill(BDL_UNIT_ID_WIDTH)
    raw["year"] = raw["year"].astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(output_path, index=False)

    log.info(
        "Saved raw powiat series: %s | records=%d | years=%d-%d | units=%d",
        output_path.name,
        len(raw),
        raw["year"].min(),
        raw["year"].max(),
        raw["bdl_unit_id"].nunique(),
    )
    return raw


def download_powiat_variable(variable_id: str, output_path: Path, force: bool = False) -> pd.DataFrame:
    """Download or load a cached powiat-level BDL variable."""

    if output_path.exists() and not force:
        raw = pd.read_parquet(output_path)
        raw["bdl_unit_id"] = raw["bdl_unit_id"].astype(str).str.zfill(BDL_UNIT_ID_WIDTH)
        raw["year"] = raw["year"].astype(int)
        log.info(
            "Loaded cached raw powiat series: %s | records=%d | years=%d-%d | units=%d",
            output_path.name,
            len(raw),
            raw["year"].min(),
            raw["year"].max(),
            raw["bdl_unit_id"].nunique(),
        )
        return raw

    df = get_bdl_variable(
        variable_id=variable_id,
        year_from=YEAR_FROM,
        year_to=YEAR_TO,
        unit_level=UNIT_LEVEL_POWIAT,
    )
    return save_raw_variable(df, output_path)


def fetch_all_powiat_units() -> pd.DataFrame:
    """Fetch the full BDL powiat unit list using the same API client."""

    records: list[dict] = []
    page = 0

    while True:
        data = _get(
            "/units",
            params={
                "level": UNIT_LEVEL_POWIAT,
                "page": page,
                "page-size": 100,
                "format": "json",
                "lang": "pl",
            },
        )
        results = data.get("results", [])
        if not results:
            break

        for row in results:
            records.append(
                {
                    "bdl_unit_id": str(row["id"]).strip().zfill(BDL_UNIT_ID_WIDTH),
                    "unit_name": row["name"],
                    # BDL powiat IDs embed the 2-digit voivodeship code in positions 3-4.
                    "voivodeship_code": str(row["id"]).strip().zfill(BDL_UNIT_ID_WIDTH)[2:4],
                }
            )

        total_records = data.get("totalRecords", 0)
        if (page + 1) * 100 >= total_records:
            break
        page += 1

    units = pd.DataFrame(records).drop_duplicates("bdl_unit_id").sort_values("bdl_unit_id").reset_index(drop=True)
    log.info("Fetched %d powiat units from /units?level=5", len(units))
    return units


def try_existing_mapping(mapping_path: Path) -> pd.DataFrame | None:
    """Use an existing BDL mapping file if it contains powiat-level records."""

    if not mapping_path.exists():
        log.info("Mapping file not found: %s", mapping_path)
        return None

    mapping = pd.read_parquet(mapping_path)
    col_lookup = {c.lower(): c for c in mapping.columns}
    level_col = col_lookup.get("level")
    if level_col is None:
        log.info("Existing mapping file has no 'level' column; building dedicated powiat mapping.")
        return None

    powiat_rows = mapping.loc[pd.to_numeric(mapping[level_col], errors="coerce") == UNIT_LEVEL_POWIAT].copy()
    if powiat_rows.empty:
        log.info("Existing mapping file contains no level=5 records; building dedicated powiat mapping.")
        return None

    bdl_col = next((col_lookup.get(name) for name in ["bdl_unit_id", "unit_id", "id"] if col_lookup.get(name)), None)
    teryt_col = next((col_lookup.get(name) for name in ["teryt4", "teryt"] if col_lookup.get(name)), None)
    if bdl_col is None or teryt_col is None:
        log.info("Existing mapping file lacks powiat ID columns; building dedicated powiat mapping.")
        return None

    result = powiat_rows[[bdl_col, teryt_col]].copy()
    result.columns = ["bdl_unit_id", "teryt4"]
    result["bdl_unit_id"] = result["bdl_unit_id"].astype(str).str.zfill(BDL_UNIT_ID_WIDTH)
    result["teryt4"] = result["teryt4"].astype(str).str.extract(r"(\d+)")[0].str.zfill(4)
    result = result.dropna().drop_duplicates("bdl_unit_id").sort_values("bdl_unit_id").reset_index(drop=True)

    log.info("Using %d powiat mappings from %s", len(result), mapping_path.name)
    return result


def build_powiat_mapping_from_units() -> pd.DataFrame:
    """Build bdl_unit_id -> teryt4 using BDL units and the IBS powiat crosswalk.

    The /units endpoint does not expose TERYT directly. We therefore match BDL
    powiat names to the IBS powiat crosswalk within voivodeship, which is
    sufficient to resolve repeated names such as 'powiat brzeski'.
    """

    units = fetch_all_powiat_units()
    units["name_norm"] = units["unit_name"].map(normalize_powiat_name)

    ibs = pd.read_excel(CROSSWALK_PATH, sheet_name="powiaty")
    ibs["teryt4"] = ibs["teryt_2023"].astype("Int64").astype(str).str.zfill(4)
    ibs["voivodeship_code"] = ibs["region"].astype("Int64").astype(str).str.zfill(2)

    name_frames = []
    for col in ["nazwa_powiatu", "nazwa_powiatu_gus", "nazwa_powiatu_unikalna"]:
        frame = ibs[["teryt4", "voivodeship_code", col]].copy()
        frame["name_norm"] = frame[col].map(normalize_powiat_name)
        name_frames.append(frame[["teryt4", "voivodeship_code", "name_norm"]])

    ibs_names = pd.concat(name_frames, ignore_index=True).dropna().drop_duplicates()

    mapping = units.merge(ibs_names, on=["voivodeship_code", "name_norm"], how="left", validate="m:1")
    if mapping["teryt4"].isna().any():
        sample = mapping.loc[mapping["teryt4"].isna(), ["bdl_unit_id", "unit_name"]].head(10).to_dict(orient="records")
        raise ValueError(f"Could not map some powiat BDL IDs to TERYT4: {sample}")

    result = mapping[["bdl_unit_id", "teryt4"]].drop_duplicates("bdl_unit_id").sort_values("bdl_unit_id").reset_index(drop=True)
    POWIAT_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(POWIAT_MAPPING_PATH, index=False)

    log.info("Built powiat mapping via /units: %d BDL units mapped", len(result))
    return result


def load_powiat_mapping() -> pd.DataFrame:
    """Load powiat mapping from an existing file or build it from scratch."""

    if POWIAT_MAPPING_PATH.exists():
        mapping = pd.read_parquet(POWIAT_MAPPING_PATH)
        mapping["bdl_unit_id"] = mapping["bdl_unit_id"].astype(str).str.zfill(BDL_UNIT_ID_WIDTH)
        mapping["teryt4"] = mapping["teryt4"].astype(str).str.zfill(4)
        log.info("Loaded cached powiat mapping: %d BDL units", len(mapping))
        return mapping

    existing = try_existing_mapping(EXISTING_MAPPING_PATH)
    if existing is not None:
        POWIAT_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing.to_parquet(POWIAT_MAPPING_PATH, index=False)
        return existing

    return build_powiat_mapping_from_units()


def map_raw_to_teryt4(raw: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Attach teryt4 to raw powiat data and verify merge coverage."""

    merged = raw.merge(mapping, on="bdl_unit_id", how="left", validate="m:1")
    missing = merged["teryt4"].isna()
    if missing.any():
        sample = merged.loc[missing, "bdl_unit_id"].drop_duplicates().head(10).tolist()
        raise ValueError(f"Missing powiat mapping for BDL IDs: {sample}")

    merged = merged[["teryt4", "year", "value"]].copy()
    merged["teryt4"] = merged["teryt4"].astype(str).str.zfill(4)

    duplicates = merged.groupby(["teryt4", "year"]).size()
    overlapping = duplicates[duplicates > 1]
    if not overlapping.empty:
        sample = overlapping.head(10).to_dict()
        raise ValueError(f"Multiple powiat values for the same (teryt4, year): {sample}")

    return merged


def build_election_average(series_df: pd.DataFrame, label: str, election_years: Iterable[int]) -> pd.DataFrame:
    """Average powiat-level annual data over T-2 and T-1 for each election year.

    The average is only defined when both years are present. This mirrors the
    logic used in the main economic panel and preserves structural NaN for
    unavailable early years (e.g. unemployment in 2005).
    """

    records = []
    for election_year in election_years:
        subset = series_df.loc[series_df["year"].isin([election_year - 2, election_year - 1])].copy()
        block = (
            subset.groupby("teryt4")
            .agg(
                mean_value=("value", "mean"),
                n_non_missing=("value", lambda s: s.notna().sum()),
            )
            .reset_index()
        )
        block[label] = block["mean_value"].where(block["n_non_missing"] == 2)
        block["year"] = int(election_year)
        records.append(block[["teryt4", "year", label]])

    return pd.concat(records, ignore_index=True).sort_values(["teryt4", "year"]).reset_index(drop=True)


def weighted_mean(group: pd.DataFrame, value_col: str, weight_col: str = "ludnosc") -> float:
    """Population-weighted mean that ignores missing values."""

    mask = group[value_col].notna() & group[weight_col].notna() & (group[weight_col] > 0)
    sub = group.loc[mask, [value_col, weight_col]]
    if sub.empty:
        return float("nan")
    return float((sub[value_col] * sub[weight_col]).sum() / sub[weight_col].sum())


def add_relative_columns(powiat_panel: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Add population-weighted relative powiat measures by election year."""

    result = powiat_panel.copy()
    for value_col in value_cols:
        nat_means = (
            result.groupby("year")
            .apply(lambda g: weighted_mean(g, value_col), include_groups=False)
            .rename(f"{value_col}_nat_mean")
            .reset_index()
        )
        result = result.merge(nat_means, on="year", how="left")
        result[f"{value_col}_relative"] = result[value_col] / result[f"{value_col}_nat_mean"]
        result = result.drop(columns=[f"{value_col}_nat_mean"])

    return result


def build_powiat_election_panel(mapping: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Download, map and transform both powiat variables into election-year blocks."""

    raw_frames: dict[str, pd.DataFrame] = {}
    for variable_id, spec in VARIABLE_SPECS.items():
        raw_frames[spec["label"]] = download_powiat_variable(variable_id, spec["raw_path"], force=force)

    election_years = sorted(pd.read_parquet(PANEL_PATH, columns=["year"])["year"].unique().tolist())
    blocks = []
    for label, raw in raw_frames.items():
        mapped = map_raw_to_teryt4(raw, mapping)
        blocks.append(build_election_average(mapped, label, election_years))

    powiat_panel = blocks[0]
    for block in blocks[1:]:
        powiat_panel = powiat_panel.merge(block, on=["teryt4", "year"], how="outer", validate="1:1")

    return powiat_panel.sort_values(["teryt4", "year"]).reset_index(drop=True)


def attach_powiat_variables_to_panel(powiat_election_panel: pd.DataFrame) -> pd.DataFrame:
    """Merge powiat variables onto the gmina-level analytical panel."""

    panel = pd.read_parquet(PANEL_PATH)
    panel["powiat4"] = panel["teryt6"].astype(str).str[:4]

    powiat_population = (
        panel.groupby(["powiat4", "year"], as_index=False)["ludnosc"]
        .sum(min_count=1)
        .rename(columns={"powiat4": "teryt4"})
    )

    powiat_panel = powiat_population.merge(powiat_election_panel, on=["teryt4", "year"], how="left", validate="1:1")
    powiat_panel = add_relative_columns(powiat_panel, ["bezrobocie_powiat", "wynagrodzenie_powiat"])

    merged = panel.merge(
        powiat_panel[
            [
                "teryt4",
                "year",
                "bezrobocie_powiat",
                "bezrobocie_powiat_relative",
                "wynagrodzenie_powiat",
                "wynagrodzenie_powiat_relative",
            ]
        ].rename(columns={"teryt4": "powiat4"}),
        on=["powiat4", "year"],
        how="left",
        validate="m:1",
    )

    for value_col, available_years in {
        "bezrobocie_powiat": VARIABLE_SPECS["60270"]["available_election_years"],
        "wynagrodzenie_powiat": VARIABLE_SPECS["64428"]["available_election_years"],
    }.items():
        filled = int(merged[value_col].notna().sum())
        missing = int(merged[value_col].isna().sum())
        structural_missing = int((merged["year"].isin(sorted(set(merged["year"]) - set(available_years))) & merged[value_col].isna()).sum())
        non_structural_missing = int((~merged["year"].isin(sorted(set(merged["year"]) - set(available_years))) & merged[value_col].isna()).sum())

        log.info(
            "Coverage for %s: filled=%d | missing=%d | structural_missing=%d | missing_outside_structural_years=%d",
            value_col,
            filled,
            missing,
            structural_missing,
            non_structural_missing,
        )
        log.info(
            "NaN breakdown by year for %s: %s",
            value_col,
            merged.loc[merged[value_col].isna(), "year"].value_counts().sort_index().to_dict(),
        )

    merged = merged.drop(columns=["powiat4"])
    return merged


def run(force: bool = False) -> None:
    """Execute the full powiat-variable extension pipeline."""

    mapping = load_powiat_mapping()
    log.info("Powiat mapping available for %d BDL units", len(mapping))

    powiat_election_panel = build_powiat_election_panel(mapping=mapping, force=force)
    panel_v2 = attach_powiat_variables_to_panel(powiat_election_panel)

    OUT_PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel_v2.to_parquet(OUT_PANEL_PATH, index=False)
    log.info("Saved extended panel: %s | shape=%s", OUT_PANEL_PATH, panel_v2.shape)

    desc_cols = [
        "bezrobocie_powiat",
        "bezrobocie_powiat_relative",
        "wynagrodzenie_powiat",
        "wynagrodzenie_powiat_relative",
    ]
    print(f"\nExtended panel shape: {panel_v2.shape}")
    print("\nNew variable summary:")
    print(panel_v2[desc_cols].describe().round(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download powiat-level BDL variables and extend the panel.")
    parser.add_argument("--force", action="store_true", help="Re-download raw BDL files even if cached.")
    args = parser.parse_args()

    run(force=args.force)
