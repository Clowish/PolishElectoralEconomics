# Polish Electoral Economics

A panel-data research project studying the relationship between local economic conditions and electoral support for populist / mainstream parties across Polish gminas (municipalities) during the 2001–2023 period.

## Research Hypothesis

> Gminas experiencing relative economic decline — measured by below-average own-revenue growth and lagging PIT receipts — exhibit systematically higher support for populist-right parties, even after controlling for gmina fixed effects and national electoral trends.

The project operationalises this as a two-way fixed-effects regression:

```
vote_share_populist_it = β · income_relative_it + α_i + γ_t + ε_it
```

where `α_i` are gmina fixed effects, `γ_t` are election-year fixed effects, and `income_relative_it` is a population-weighted relative income measure.

## Data Sources

| Dataset | Source | URL | Coverage |
|---|---|---|---|
| Sejm election results (gmina level) | Krajowe Biuro Wyborcze / PKW | [wyniki.pkw.gov.pl](https://wyniki.pkw.gov.pl) · [danewyborcze.kbw.gov.pl](https://danewyborcze.kbw.gov.pl) | 2001, 2005, 2007, 2011, 2015, 2019, 2023 |
| Municipal budget revenues per capita | BDL GUS (API v1) | [bdl.stat.gov.pl/api/v1](https://bdl.stat.gov.pl/api/v1/swagger) | 2000–2023 |
| Population by gmina | BDL GUS (API v1) | [bdl.stat.gov.pl/api/v1](https://bdl.stat.gov.pl/api/v1/swagger) | 2000–2023 |
| TERYT crosswalk (gmina boundaries) | IBS KTS-6 | [ibs.org.pl/en/resources/crosswalks-for-polish-counties-and-municipalities-kts-5-kts-6-1999-2024/](https://ibs.org.pl/en/resources/crosswalks-for-polish-counties-and-municipalities-kts-5-kts-6-1999-2024/) | 1999–2024 |

## Installation

### Requirements

- Python ≥ 3.11
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# Clone repository
git clone <repo-url>
cd polish-electoral-economics

# Create virtual environment and install all dependencies
uv venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

uv pip install -e .
```

### Optional: BDL API key

A BDL API key increases the rate limit from ~60 to ~500 requests/minute.
Register at [api.stat.gov.pl](https://api.stat.gov.pl/Home/BdlApi).

```bash
cp .env.example .env
# Edit .env and set BDL_API_KEY=<your_key>
```

## Usage

Run all data-collection steps in order:

```bash
# Step 1 – Download election results (2001–2023)
python -m src.data.download_elections

# Step 2 – Download BDL economic indicators
python -m src.data.download_economic

# Step 3 – Harmonise TERYT codes
python -m src.data.harmonize_teryt
```

Add `--force` to any script to re-download existing files.

Explore BDL subject tree:

```bash
python -m src.data.download_economic --list-variables
```

## Directory Structure

```
polish-electoral-economics/
├── src/
│   ├── data/
│   │   ├── download_elections.py   # PKW/KBW election data pipeline
│   │   ├── download_economic.py    # BDL GUS API client
│   │   └── harmonize_teryt.py      # TERYT crosswalk & harmonisation
│   └── analysis/                   # Panel regression models (TODO)
├── data/
│   ├── raw/
│   │   ├── elections/{year}/       # Downloaded ZIP/CSV files
│   │   └── economic/               # Cached BDL parquet files
│   ├── interim/                    # Cleaned, harmonised parquets
│   └── processed/                  # Analysis-ready panel datasets
├── notebooks/                      # Exploratory Jupyter notebooks
├── reports/
│   └── figures/                    # Publication-quality charts
├── tests/                          # pytest test suite
├── pyproject.toml
└── .env.example
```

## Intermediate outputs

| File | Description |
|---|---|
| `data/interim/elections_panel_raw.parquet` | Raw vote counts per gmina × year |
| `data/interim/economic_panel.parquet` | BDL indicators per gmina × year |
| `data/interim/teryt_crosswalk.parquet` | Full historical TERYT mapping |
| `data/interim/teryt_stable_units.parquet` | Canonical gmina list with stability flag |

## Tests

```bash
pytest
```

## License

MIT License © 2024. See [LICENSE](LICENSE) for details.
