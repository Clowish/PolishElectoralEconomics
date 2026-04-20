# Polish Electoral Economics

Projekt badawczy o relacji między lokalną sytuacją ekonomiczną a zmianami poparcia dla partii populistycznych w wyborach do Sejmu w Polsce. Jednostką obserwacji jest gmina w roku wyborczym, a panel obejmuje wybory z lat 2001, 2005, 2007, 2011, 2015, 2019 i 2023.

## Pytanie badawcze

Czy gminy, których relatywna pozycja ekonomiczna pogarsza się względem średniej krajowej, przesuwają się silniej w stronę populistycznej prawicy?

Projekt traktuje to jako analizę panelową, a nie eksperymentalną identyfikację przyczynową. Bazowa specyfikacja używa first differences:

```text
Δpct_populist_right_it =
    β1 · Δdochody_pc_relative_it
  + β2 · Δfrekwencja_it
  + γ_t
  + ε_it
```

Błędy standardowe są klastrowane na poziomie powiatu, zdefiniowanego jako pierwsze cztery cyfry kodu TERYT gminy.

## Najważniejszy wynik

Model bazowy FD nie potwierdza hipotezy relative deprivation:

- `β1 = -0.276`
- `p = 0.508`

Model TWFE daje większy, granicznie istotny efekt (`β1 = -0.967`, `p = 0.075`), a robustness check z PIT daje wynik dodatni i istotny (`β1 = 2.175`, `p = 0.0003`). To sugeruje, że dochody budżetowe i PIT nie są zamiennymi miarami lokalnej sytuacji ekonomicznej.

## Dane

Główny panel analityczny:

- `data/processed/panel_final.parquet` - bazowy panel gmina × rok wyborczy.
- `data/processed/panel_final_v2.parquet` - panel rozszerzony o zmienne powiatowe: bezrobocie rejestrowane i przeciętne wynagrodzenie.

Dane w katalogu `data/` nie są wersjonowane w Git, ponieważ zawierają surowe i przetworzone pliki wejściowe. Pipeline w `src/data/` służy do ich odtworzenia.

Klucz łączenia danych to zawsze 6-cyfrowy string `teryt6`, nigdy liczba całkowita.

## Źródła danych

| Dane | Źródło | Zakres |
|---|---|---|
| Wyniki wyborów do Sejmu na poziomie gmin | PKW / KBW, `danewyborcze.kbw.gov.pl` | 2001-2023 |
| Dochody budżetów gmin per capita | BDL GUS API | 2002-2023 |
| Dochody własne gmin per capita | BDL GUS API | 2002-2023 |
| PIT/CIT per capita | BDL GUS API | 2002-2023 w panelu wyborczym |
| Ludność | BDL GUS API | używana jako waga krajowa |
| Bezrobocie rejestrowane | BDL GUS API, poziom powiatu | 2004-2023 |
| Przeciętne wynagrodzenie brutto | BDL GUS API, poziom powiatu | 2002-2023 |
| Granice gmin | GeoJSON lokalny w `data/raw/spatial/` | aktualny przekrój przestrzenny |
| Crosswalk TERYT | IBS | 1999-2025 |

## Instalacja

Projekt używa `uv`.

```bash
uv sync
```

Opcjonalnie można ustawić klucz BDL API:

```bash
cp .env.example .env
# BDL_API_KEY=<twój_klucz>
```

## Odtworzenie pipeline

Wszystkie komendy uruchamiaj z katalogu głównego repozytorium.

```bash
uv run python3 -m src.data.download_elections
uv run python3 -m src.data.download_economic
uv run python3 -m src.data.download_powiat_vars
uv run python3 -m src.data.harmonize_teryt
uv run python3 -m src.analysis.build_panel
```

## Analiza

Model bazowy:

```bash
uv run python3 -m src.analysis.run_baseline_fd
```

Mapy statyczne:

```bash
uv run python3 -m src.analysis.make_maps
```

Notebook podsumowujący:

```bash
uv run jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=120 \
  notebooks/01-research-summary.ipynb \
  --inplace
```

## Aplikacja interaktywna

Aplikacja Dash pozwala eksplorować mapę, ranking gmin i trajektorie wyborczo-ekonomiczne.

```bash
uv run python3 -m src.app.app
```

Domyślny adres:

```text
http://127.0.0.1:8050/
```

W aplikacji wykres wyborczy w panelu gminy pokazuje pełną kompozycję do 100%: populistyczną prawicę, liberalne centrum, lewicę postkomunistyczną, główny nurt prawicy i inne komitety.

## Struktura

```text
polish-electoral-economics/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   └── 01-research-summary.ipynb
├── reports/
│   ├── figures/
│   └── tables/
├── src/
│   ├── analysis/
│   ├── app/
│   └── data/
├── tests/
├── pyproject.toml
└── README.md
```

## Testy

```bash
uv run pytest
```

Aktualnie testy obejmują przygotowanie próby dla modelu FD i walidację kodów TERYT.
