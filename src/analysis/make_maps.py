"""
Create static choropleth small-multiple maps for key project variables.

Usage:
    uv run python3 -m src.analysis.make_maps
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "panel_final.parquet"
GEOJSON_PATH = PROJECT_ROOT / "data" / "raw" / "spatial" / "gminy.geojson"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_geodata(geojson_path: str) -> gpd.GeoDataFrame:
    """Load gmina geometries keyed by 6-digit TERYT."""

    gdf = gpd.read_file(geojson_path)
    gdf["teryt6"] = gdf["teryt6"].astype(str).str.zfill(6)

    duplicated = gdf["teryt6"].duplicated()
    if duplicated.any():
        sample = gdf.loc[duplicated, "teryt6"].head().tolist()
        raise ValueError(f"GeoJSON contains duplicate teryt6 values: {sample}")

    if gdf.crs is None:
        raise ValueError("GeoJSON has no CRS; expected EPSG:4326.")

    if gdf.crs.to_string() != "EPSG:4326":
        log.info("Reprojecting geodata from %s to EPSG:4326", gdf.crs)
        gdf = gdf.to_crs(epsg=4326)

    log.info("Loaded geodata: %d geometries, CRS=%s", len(gdf), gdf.crs)
    return gdf


def load_panel(parquet_path: str) -> pd.DataFrame:
    """Load the minimal set of panel columns needed for the maps."""

    cols = [
        "teryt6",
        "year",
        "pct_populist_right",
        "dochody_pc_relative",
        "delta_pct_populist_right",
        "delta_dochody_pc_relative",
    ]
    df = pd.read_parquet(parquet_path, columns=cols)
    df["teryt6"] = df["teryt6"].astype(str).str.zfill(6)

    duplicated = df.duplicated(subset=["teryt6", "year"])
    if duplicated.any():
        sample = (
            df.loc[duplicated, ["teryt6", "year"]]
            .head()
            .to_dict(orient="records")
        )
        raise ValueError(f"Panel contains duplicate (teryt6, year) keys: {sample}")

    log.info("Loaded panel: %d rows, %d unique gminas", len(df), df["teryt6"].nunique())
    return df


def check_coverage(gdf: gpd.GeoDataFrame, df: pd.DataFrame) -> None:
    """Log TERYT coverage diagnostics for panel and geodata."""

    geo_teryt = set(gdf["teryt6"])
    panel_teryt = set(df["teryt6"])

    panel_missing_in_geo = sorted(panel_teryt - geo_teryt)
    geo_missing_in_panel = sorted(geo_teryt - panel_teryt)

    log.info(
        "Coverage check: %d panel TERYT6, %d GeoJSON TERYT6",
        len(panel_teryt),
        len(geo_teryt),
    )
    log.info(
        "Coverage check: panel without geometry=%d, geometry without panel=%d",
        len(panel_missing_in_geo),
        len(geo_missing_in_panel),
    )

    if panel_missing_in_geo:
        log.info(
            "Example panel TERYT6 without geometry: %s",
            ", ".join(panel_missing_in_geo[:10]),
        )
    if geo_missing_in_panel:
        log.info(
            "Example GeoJSON TERYT6 without panel row: %s",
            ", ".join(geo_missing_in_panel[:10]),
        )


def prepare_map_geodata(
    gdf: gpd.GeoDataFrame,
    df: pd.DataFrame,
    variable: str,
    years: list[int],
) -> gpd.GeoDataFrame:
    """Expand geodata to all requested years and merge the target variable."""

    panel_subset = df.loc[df["year"].isin(years), ["teryt6", "year", variable]].copy()
    year_frame = pd.DataFrame({"year": years})

    expanded = (
        gdf.assign(_merge_key=1)
        .merge(year_frame.assign(_merge_key=1), on="_merge_key", how="inner")
        .drop(columns="_merge_key")
    )

    merged = expanded.merge(
        panel_subset,
        on=["teryt6", "year"],
        how="left",
        validate="1:1",
    )
    merged_gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=gdf.crs)

    missing_rows = int(merged_gdf[variable].isna().sum())
    log.info(
        "Prepared map frame for %s: %d rows (%d missing values across selected years)",
        variable,
        len(merged_gdf),
        missing_rows,
    )
    return merged_gdf


def make_small_multiples(
    gdf_merged: gpd.GeoDataFrame,
    variable: str,
    years: list[int],
    cmap: str,
    norm,
    suptitle: str,
    output_path: str,
    dpi: int = 150,
) -> None:
    """Render one-row small multiples with a shared horizontal colorbar."""

    fig, axes = plt.subplots(1, len(years), figsize=(3 * len(years), 4.2), dpi=dpi)
    axes = np.atleast_1d(axes)

    cmap_obj = plt.get_cmap(cmap).copy()

    for ax, year in zip(axes, years):
        year_gdf = gdf_merged.loc[gdf_merged["year"] == year].copy()

        year_gdf.plot(ax=ax, color="lightgrey", linewidth=0, edgecolor="none")

        has_data = year_gdf[variable].notna()
        if has_data.any():
            year_gdf.loc[has_data].plot(
                column=variable,
                cmap=cmap_obj,
                norm=norm,
                ax=ax,
                linewidth=0,
                edgecolor="none",
            )
        else:
            log.info("Year %s for %s contains only missing values", year, variable)

        ax.set_title(str(year), fontsize=11)
        ax.set_axis_off()

    fig.suptitle(suptitle, fontsize=14)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.16, wspace=0.01)

    scalar_mappable = ScalarMappable(norm=norm, cmap=cmap_obj)
    scalar_mappable.set_array([])
    fig.colorbar(
        scalar_mappable,
        ax=axes.tolist(),
        orientation="horizontal",
        fraction=0.05,
        pad=0.04,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved map: %s", output)


def make_symmetric_norm(df: pd.DataFrame, variable: str, years: list[int]) -> mcolors.TwoSlopeNorm:
    """Build a symmetric p1-p99 TwoSlopeNorm centered at zero."""

    values = df.loc[df["year"].isin(years), variable].dropna().to_numpy()
    if values.size == 0:
        raise ValueError(f"No non-missing values available for {variable} in selected years.")

    p1, p99 = np.nanpercentile(values, [1, 99])
    bound = max(abs(float(p1)), abs(float(p99)))

    if not np.isfinite(bound) or bound == 0:
        bound = float(np.nanmax(np.abs(values)))
    if not np.isfinite(bound) or bound == 0:
        bound = 1.0

    log.info(
        "Color scale for %s uses symmetric bound +/- %.4f based on p1=%.4f, p99=%.4f",
        variable,
        bound,
        float(p1),
        float(p99),
    )
    return mcolors.TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    gdf = load_geodata(str(GEOJSON_PATH))
    df = load_panel(str(PANEL_PATH))
    check_coverage(gdf, df)

    configs = [
        {
            "variable": "pct_populist_right",
            "years": [2001, 2005, 2007, 2011, 2015, 2019, 2023],
            "cmap": "YlOrRd",
            "norm": mcolors.Normalize(vmin=0, vmax=60),
            "suptitle": "Udział głosów na populistyczną prawicę (%)",
            "output_path": FIGURES_DIR / "map_pct_populist_right.png",
        },
        {
            "variable": "dochody_pc_relative",
            "years": [2007, 2011, 2015, 2019, 2023],
            "cmap": "RdBu_r",
            "norm": mcolors.TwoSlopeNorm(vmin=0.3, vcenter=1.0, vmax=2.0),
            "suptitle": "Dochody gminy per capita względem średniej krajowej",
            "output_path": FIGURES_DIR / "map_dochody_pc_relative.png",
        },
        {
            "variable": "delta_pct_populist_right",
            "years": [2005, 2007, 2011, 2015, 2019, 2023],
            "cmap": "RdBu_r",
            "norm": make_symmetric_norm(
                df,
                variable="delta_pct_populist_right",
                years=[2005, 2007, 2011, 2015, 2019, 2023],
            ),
            "suptitle": "Zmiana udziału głosów na populistyczną prawicę (p.p.)",
            "output_path": FIGURES_DIR / "map_delta_pct_populist_right.png",
        },
        {
            "variable": "delta_dochody_pc_relative",
            "years": [2007, 2011, 2015, 2019, 2023],
            "cmap": "RdBu_r",
            "norm": make_symmetric_norm(
                df,
                variable="delta_dochody_pc_relative",
                years=[2007, 2011, 2015, 2019, 2023],
            ),
            "suptitle": "Zmiana relatywnych dochodów gminy per capita",
            "output_path": FIGURES_DIR / "map_delta_dochody_pc_relative.png",
        },
    ]

    for config in configs:
        merged = prepare_map_geodata(
            gdf=gdf,
            df=df,
            variable=config["variable"],
            years=config["years"],
        )
        make_small_multiples(
            gdf_merged=merged,
            variable=config["variable"],
            years=config["years"],
            cmap=config["cmap"],
            norm=config["norm"],
            suptitle=config["suptitle"],
            output_path=str(config["output_path"]),
            dpi=150,
        )


if __name__ == "__main__":
    main()
