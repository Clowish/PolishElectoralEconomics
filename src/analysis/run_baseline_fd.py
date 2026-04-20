"""
Estimate the baseline first-difference model for populist-right vote share.

Specification:
    delta_pct_populist_right_it = beta1 * delta_dochody_pc_relative_it
                                + beta2 * delta_frekwencja_it
                                + gamma_t + epsilon_it

The dataset already contains manually constructed first differences, so this
module runs a pooled regression on differenced outcomes with election-year
fixed effects. Gmina fixed effects are removed by differencing and are not
added back to avoid redundant absorption.

Standard errors are clustered at the powiat level, proxied by the first four
digits of the 6-digit TERYT gmina code.

Usage:
    uv run python3 -m src.analysis.run_baseline_fd
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from linearmodels.panel import PanelOLS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "panel_final.parquet"
REPORTS_DIR = PROJECT_ROOT / "reports" / "tables"
SUMMARY_PATH = REPORTS_DIR / "baseline_fd_summary.txt"
COEFFICIENTS_PATH = REPORTS_DIR / "baseline_fd_coefficients.csv"
METADATA_PATH = REPORTS_DIR / "baseline_fd_metadata.json"

MODEL_YEARS = [2007, 2011, 2015, 2019, 2023]
DEPENDENT_VAR = "delta_pct_populist_right"
REGRESSORS = ["delta_dochody_pc_relative", "delta_frekwencja"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def prepare_baseline_fd_sample(panel: pd.DataFrame) -> pd.DataFrame:
    """Build the estimation sample for the baseline FD specification.

    The first non-missing economic differences appear in 2007 because the
    income variables are built from T-2/T-1 averages. We therefore keep only
    election years 2007-2023 and require complete data for all model columns.
    """

    required_cols = ["teryt6", "year", DEPENDENT_VAR, *REGRESSORS]
    missing_cols = [col for col in required_cols if col not in panel.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    sample = panel.loc[panel["year"].isin(MODEL_YEARS), required_cols].copy()
    sample["teryt6"] = sample["teryt6"].astype(str).str.zfill(6)

    invalid_teryt = ~sample["teryt6"].str.fullmatch(r"\d{6}")
    if invalid_teryt.any():
        bad_values = sample.loc[invalid_teryt, "teryt6"].head().tolist()
        raise ValueError(f"Found invalid TERYT values in estimation sample: {bad_values}")

    sample = sample.dropna(subset=[DEPENDENT_VAR, *REGRESSORS]).copy()
    sample["powiat_cluster"] = sample["teryt6"].str[:4]
    sample = sample.set_index(["teryt6", "year"]).sort_index()

    if sample.empty:
        raise ValueError("Estimation sample is empty after filtering years and missing values.")

    return sample


def fit_baseline_fd(sample: pd.DataFrame):
    """Estimate the first-difference model with election-year fixed effects."""

    outcome = sample[DEPENDENT_VAR]
    exog = sample[REGRESSORS]
    clusters = sample[["powiat_cluster"]]

    model = PanelOLS(outcome, exog, time_effects=True)
    return model.fit(cov_type="clustered", clusters=clusters)


def tidy_coefficients(result) -> pd.DataFrame:
    """Convert the linearmodels result object into a compact coefficient table."""

    conf_int = result.conf_int()
    return (
        pd.DataFrame(
            {
                "term": result.params.index,
                "coefficient": result.params.values,
                "std_error": result.std_errors.values,
                "t_stat": result.tstats.values,
                "p_value": result.pvalues.values,
                "ci_lower": conf_int.iloc[:, 0].values,
                "ci_upper": conf_int.iloc[:, 1].values,
            }
        )
        .sort_values("term")
        .reset_index(drop=True)
    )


def build_metadata(sample: pd.DataFrame, result) -> dict[str, object]:
    """Collect model metadata that is useful for downstream reporting."""

    years = sorted(sample.index.get_level_values("year").unique().tolist())
    return {
        "input_panel": str(PANEL_PATH.relative_to(PROJECT_ROOT)),
        "specification": (
            "delta_pct_populist_right ~ delta_dochody_pc_relative + "
            "delta_frekwencja + election_year_FE"
        ),
        "dependent_variable": DEPENDENT_VAR,
        "regressors": REGRESSORS,
        "model_years": years,
        "n_observations": int(result.nobs),
        "n_gminas": int(sample.index.get_level_values("teryt6").nunique()),
        "n_powiat_clusters": int(sample["powiat_cluster"].nunique()),
        "time_effects": True,
        "entity_effects": False,
        "cluster_definition": "First 4 digits of teryt6 (powiat proxy).",
        "r_squared": float(result.rsquared),
        "r_squared_within": float(result.rsquared_within),
        "covariance_estimator": "clustered",
    }


def write_outputs(result, sample: pd.DataFrame) -> None:
    """Persist the regression summary and tidy outputs under reports/tables."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    SUMMARY_PATH.write_text(result.summary.as_text() + "\n", encoding="utf-8")
    tidy_coefficients(result).to_csv(COEFFICIENTS_PATH, index=False)
    METADATA_PATH.write_text(
        json.dumps(build_metadata(sample, result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    log.info("Loading panel: %s", PANEL_PATH)
    panel = pd.read_parquet(PANEL_PATH)

    sample = prepare_baseline_fd_sample(panel)
    log.info(
        "Prepared FD sample: %d observations, %d gminas, %d powiat clusters",
        len(sample),
        sample.index.get_level_values("teryt6").nunique(),
        sample["powiat_cluster"].nunique(),
    )

    result = fit_baseline_fd(sample)
    write_outputs(result, sample)

    log.info("Saved summary: %s", SUMMARY_PATH)
    log.info("Saved coefficients: %s", COEFFICIENTS_PATH)
    log.info("Saved metadata: %s", METADATA_PATH)

    print(result.summary)


if __name__ == "__main__":
    main()
