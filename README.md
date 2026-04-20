# Polish Electoral Economics

Projekt badawczy o relacji miedzy lokalna sytuacja ekonomiczna a zmianami poparcia dla partii populistycznych w wyborach do Sejmu w Polsce. Jednostka obserwacji jest gmina w roku wyborczym, a panel obejmuje wybory z lat 2001, 2005, 2007, 2011, 2015, 2019 i 2023.

## Pytanie badawcze

Czy gminy, ktorych relatywna pozycja ekonomiczna pogarsza sie wzgledem sredniej krajowej, przesuwaja sie silniej w strone populistycznej prawicy?

Projekt traktuje to jako analize panelowa, a nie eksperymentalna identyfikacje przyczynowa. Bazowa specyfikacja uzywa first differences:

```text
delta_pct_populist_right_it =
    beta1 * delta_dochody_pc_relative_it
  + beta2 * delta_frekwencja_it
  + gamma_t
  + epsilon_it
```

Bledy standardowe sa klastrowane na poziomie powiatu, zdefiniowanego jako pierwsze cztery cyfry kodu TERYT gminy.

## Najwazniejszy wynik

Model bazowy FD nie potwierdza hipotezy relative deprivation:

- `beta1 = -0.276`
- `p = 0.508`

Model TWFE daje wiekszy, granicznie istotny efekt (`beta1 = -0.967`, `p = 0.075`), a robustness check z PIT daje wynik dodatni i istotny (`beta1 = 2.175`, `p = 0.0003`). To sugeruje, ze dochody budzetowe i PIT nie sa zamiennymi miarami lokalnej sytuacji ekonomicznej.

## Dane

Glowny panel analityczny:

- `data/processed/panel_final.parquet` - bazowy panel gmina x rok wyborczy.
- `data/processed/panel_final_v2.parquet` - panel rozszerzony o zmienne powiatowe: bezrobocie rejestrowane i przecietne wynagrodzenie.

Dane w katalogu `data/` nie sa wersjonowane w Git, poniewaz zawieraja surowe i przetworzone pliki wejsciowe. Pipeline w `src/data/` sluzy do ich odtworzenia.

Klucz laczenia danych to zawsze 6-cyfrowy string `teryt6`, nigdy liczba calkowita.

## Zrodla danych

| Dane | Zrodlo | Zakres |
|---|---|---|
| Wyniki wyborow do Sejmu na poziomie gmin | PKW / KBW, `danewyborcze.kbw.gov.pl` | 2001-2023 |
| Dochody budzetow gmin per capita | BDL GUS API | 2002-2023 |
| Dochody wlasne gmin per capita | BDL GUS API | 2002-2023 |
| PIT/CIT per capita | BDL GUS API | 2002-2023 w panelu wyborczym |
| Ludnosc | BDL GUS API | uzywana jako waga krajowa |
| Bezrobocie rejestrowane | BDL GUS API, poziom powiatu | 2004-2023 |
| Przecietne wynagrodzenie brutto | BDL GUS API, poziom powiatu | 2002-2023 |
| Granice gmin | GeoJSON lokalny w `data/raw/spatial/` | aktualny przekroj przestrzenny |
| Crosswalk TERYT | IBS | 1999-2025 |

## Instalacja

Projekt uzywa `uv`.

```bash
uv sync
```

Opcjonalnie mozna ustawic klucz BDL API:

```bash
cp .env.example .env
# BDL_API_KEY=<twoj_klucz>
```

## Odtworzenie pipeline

Wszystkie komendy uruchamiaj z katalogu glownego repozytorium.

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

Notebook podsumowujacy:

```bash
uv run jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=120 \
  notebooks/01-research-summary.ipynb \
  --inplace
```

## Aplikacja interaktywna

Aplikacja Dash pozwala eksplorowac mape, ranking gmin i trajektorie wyborczo-ekonomiczne.

```bash
uv run python3 -m src.app.app
```

Domyslny adres:

```text
http://127.0.0.1:8050/
```

W aplikacji wykres wyborczy w panelu gminy pokazuje pelna kompozycje do 100%: populistyczna prawice, liberalne centrum, lewice postkomunistyczna, glowny nurt prawicy i inne komitety.

## Struktura

```text
polish-electoral-economics/
|-- data/
|   |-- raw/
|   |-- interim/
|   `-- processed/
|-- notebooks/
|   `-- 01-research-summary.ipynb
|-- reports/
|   |-- figures/
|   `-- tables/
|-- src/
|   |-- analysis/
|   |-- app/
|   `-- data/
|-- tests/
|-- pyproject.toml
`-- README.md
```

## Testy

```bash
uv run pytest
```

Aktualnie testy obejmuja przygotowanie proby dla modelu FD i walidacje kodow TERYT.
