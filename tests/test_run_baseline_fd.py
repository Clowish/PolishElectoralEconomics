from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.run_baseline_fd import prepare_baseline_fd_sample


def test_prepare_baseline_fd_sample_filters_years_and_adds_powiat_clusters() -> None:
    panel = pd.DataFrame(
        {
            "teryt6": ["20101", "020101", "020101", "020102", "020102"],
            "year": [2005, 2007, 2011, 2007, 2023],
            "delta_pct_populist_right": [0.1, 1.1, 1.4, 0.5, -0.2],
            "delta_dochody_pc_relative": [None, 0.02, 0.05, 0.01, -0.03],
            "delta_frekwencja": [0.4, -0.1, 0.2, 0.3, 0.1],
        }
    )

    sample = prepare_baseline_fd_sample(panel)

    assert sample.index.names == ["teryt6", "year"]
    assert sample.index.get_level_values("year").tolist() == [2007, 2011, 2007, 2023]
    assert sample.index.get_level_values("teryt6").tolist() == ["020101", "020101", "020102", "020102"]
    assert sample["powiat_cluster"].tolist() == ["0201", "0201", "0201", "0201"]


def test_prepare_baseline_fd_sample_rejects_invalid_teryt_codes() -> None:
    panel = pd.DataFrame(
        {
            "teryt6": ["02A101"],
            "year": [2007],
            "delta_pct_populist_right": [1.0],
            "delta_dochody_pc_relative": [0.2],
            "delta_frekwencja": [0.1],
        }
    )

    with pytest.raises(ValueError, match="invalid TERYT"):
        prepare_baseline_fd_sample(panel)
