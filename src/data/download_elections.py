"""
Download and parse Sejm election results (2001–2023) from PKW/KBW sources.

Data sources (wszystkie na danewyborcze.kbw.gov.pl, zweryfikowane 2026-04-16):
- 2001: dane/2001/sejm/sejm2001-lis-gm.xls (gminy, ~541KB)
- 2005: dane/2005/sejm/1456225675_36795.xls (gminy, ~739KB)
- 2007: dane/2007/sejm/sejm2007-gm-listy.xls (gminy, ~600KB)
- 2011: dane/2011/sejmsenat/2011-gl-lis-gm.xls (gminy, ~729KB)
- 2015: dane/2015/sejm/2015-gl-lis-gm.zip (gminy, ~254KB)
- 2019: dane/2019/sejmsenat/wyniki_gl_na_listy_po_gminach_sejm_csv.zip (gminy CSV, ~139KB)
- 2023: dane/2023/sejmsenat/wyniki_gl_na_listy_po_gminach_sejm_csv.zip (gminy CSV UTF-8, ~152KB)

Usage:
    python -m src.data.download_elections            # skip if files exist
    python -m src.data.download_elections --force    # re-download all
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "elections"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

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
# Party classification
# ---------------------------------------------------------------------------
PARTY_CLASSIFICATION: dict[str, str] = {
    # populist_right
    "PRAWO I SPRAWIEDLIWOŚĆ": "populist_right",
    "PIS": "populist_right",
    "PRAWO I SPRAWIEDLIWOSC": "populist_right",
    "SAMOOBRONA RZECZPOSPOLITEJ POLSKIEJ": "populist_right",
    "SAMOOBRONA RP": "populist_right",
    "LIGA POLSKICH RODZIN": "populist_right",
    "LPR": "populist_right",
    "KUKIZ'15": "populist_right",
    "KUKIZ15": "populist_right",
    "KUKIZ 15": "populist_right",
    "KONFEDERACJA WOLNOŚĆ I NIEPODLEGŁOŚĆ": "populist_right",
    "KONFEDERACJA": "populist_right",
    "KONFEDERACJA LIBERTY INDEPENDENCE": "populist_right",
    # mainstream_right
    "AKCJA WYBORCZA SOLIDARNOŚĆ": "mainstream_right",
    "AWS": "mainstream_right",
    "PLATFORMA OBYWATELSKA": "mainstream_right",
    "PO": "mainstream_right",
    "POLSKIE STRONNICTWO LUDOWE": "mainstream_right",
    "PSL": "mainstream_right",
    "ZJEDNOCZONA PRAWICA": "mainstream_right",
    "KOALICJA POLSKA": "mainstream_right",
    "TRZECIA DROGA": "mainstream_right",
    # post_communist_left
    "SOJUSZ LEWICY DEMOKRATYCZNEJ": "post_communist_left",
    "SLD": "post_communist_left",
    "SLD-UP": "post_communist_left",
    "LEWICA": "post_communist_left",
    "KOALICJA LEWICA": "post_communist_left",
    "NOWA LEWICA": "post_communist_left",
    # liberal_center
    "UNIA WOLNOŚCI": "liberal_center",
    "UW": "liberal_center",
    "PARTIA DEMOKRATYCZNA": "liberal_center",
    "NOWOCZESNA": "liberal_center",
    ".NOWOCZESNA": "liberal_center",
    "KOALICJA OBYWATELSKA": "liberal_center",
    "KO": "liberal_center",
    "POLSKA 2050": "liberal_center",
    "POLSKA 2050 SZYMONA HOŁOWNI": "liberal_center",
}


def classify_committee(name: str) -> str:
    """Map raw committee name to one of five political categories.

    Parameters
    ----------
    name:
        Raw committee name as it appears in PKW data.

    Returns
    -------
    str
        One of: populist_right, mainstream_right, post_communist_left,
        liberal_center, other.
    """
    upper = name.upper().strip()
    # exact match
    if upper in PARTY_CLASSIFICATION:
        return PARTY_CLASSIFICATION[upper]
    # partial match – iterate by length descending to prefer longer keys
    for key, cat in sorted(PARTY_CLASSIFICATION.items(), key=lambda x: -len(x[0])):
        if key in upper:
            return cat
    return "other"


# ---------------------------------------------------------------------------
# Download URLs
# ---------------------------------------------------------------------------
# Each entry: year → list of (url, local_filename)
# Wszystkie pliki z danewyborcze.kbw.gov.pl — zweryfikowane 2026-04-16.
# Pliki zawierają wyniki głosowania na listy Sejmu na poziomie gmin z kodami TERYT.
_KBW = "https://danewyborcze.kbw.gov.pl"
ELECTION_SOURCES: dict[int, list[tuple[str, str]]] = {
    2001: [
        (
            f"{_KBW}/dane/2001/sejm/sejm2001-lis-gm.xls",
            "wyniki_2001_gm.xls",
        )
    ],
    2005: [
        (
            f"{_KBW}/dane/2005/sejm/1456225675_36795.xls",
            "wyniki_2005_gm.xls",
        )
    ],
    2007: [
        (
            f"{_KBW}/dane/2007/sejm/sejm2007-gm-listy.xls",
            "wyniki_2007_gm.xls",
        )
    ],
    2011: [
        (
            f"{_KBW}/dane/2011/sejmsenat/2011-gl-lis-gm.xls",
            "wyniki_2011_gm.xls",
        )
    ],
    2015: [
        (
            f"{_KBW}/dane/2015/sejm/2015-gl-lis-gm.zip",
            "wyniki_2015_gm.zip",
        )
    ],
    2019: [
        (
            f"{_KBW}/dane/2019/sejmsenat/wyniki_gl_na_listy_po_gminach_sejm_csv.zip",
            "wyniki_2019_gm_csv.zip",
        )
    ],
    2023: [
        (
            f"{_KBW}/dane/2023/sejmsenat/wyniki_gl_na_listy_po_gminach_sejm_csv.zip",
            "wyniki_2023_gm_csv.zip",
        )
    ],
}


def download_file(url: str, dest: Path, force: bool = False) -> Path:
    """Download *url* to *dest*; skip if exists unless *force* is True.

    Parameters
    ----------
    url:
        Remote URL.
    dest:
        Local destination path.
    force:
        Re-download even if *dest* already exists.

    Returns
    -------
    Path
        Path to the downloaded (or pre-existing) file.
    """
    if dest.exists() and not force:
        log.info("Skip download (exists): %s", dest.name)
        return dest
    log.info("Downloading %s → %s", url, dest.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
            ) as bar:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
                    bar.update(len(chunk))
    except requests.RequestException as exc:
        log.warning("Download failed for %s: %s", url, exc)
        # Remove partial file
        if dest.exists():
            dest.unlink()
        raise
    return dest


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _normalize_teryt(code: str) -> str:
    """Ensure TERYT code is a zero-padded 6- or 7-character string."""
    s = str(code).strip().split(".")[0]  # remove any decimal
    s = re.sub(r"\D", "", s)
    if len(s) in (6, 7):
        return s
    if len(s) < 6:
        return s.zfill(6)
    return s[:7]


def _find_csv_in_zip(zf: zipfile.ZipFile, pattern: str) -> Optional[str]:
    """Return the first ZipFile member matching *pattern* (case-insensitive)."""
    pat = pattern.lower()
    for name in zf.namelist():
        if pat in name.lower() and name.lower().endswith((".csv", ".xlsx", ".xls")):
            return name
    return None


def _read_tabular(path_or_bytes: Path | bytes, **kwargs) -> pd.DataFrame:
    """Read CSV or XLSX into a DataFrame, handling encoding quirks."""
    if isinstance(path_or_bytes, bytes):
        buf = BytesIO(path_or_bytes)
        try:
            return pd.read_csv(buf, sep=";", encoding="utf-8", **kwargs)
        except Exception:
            buf.seek(0)
            try:
                return pd.read_csv(buf, sep=";", encoding="cp1250", **kwargs)
            except Exception:
                buf.seek(0)
                return pd.read_excel(buf, **kwargs)
    else:
        suffix = path_or_bytes.suffix.lower()
        if suffix in (".xlsx", ".xls"):
            return pd.read_excel(path_or_bytes, **kwargs)
        for enc in ("utf-8", "cp1250"):
            try:
                return pd.read_csv(path_or_bytes, sep=";", encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path_or_bytes, sep=";", encoding="utf-8", errors="replace", **kwargs)


# ---------------------------------------------------------------------------
# Year-specific parsers
# ---------------------------------------------------------------------------

def _parse_generic(df_raw: pd.DataFrame, year: int) -> pd.DataFrame:
    """Generic column-mapping parser for KBW CSV format.

    The KBW archives use slightly different column names each year.  This
    function tries common patterns and falls back gracefully.

    Parameters
    ----------
    df_raw:
        Raw DataFrame read from the CSV/XLSX file.
    year:
        Election year (used to set the *year* column).

    Returns
    -------
    pd.DataFrame
        Standardised election DataFrame.
    """
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # --- Identify TERYT column ---
    teryt_candidates = [c for c in df.columns if "teryt" in c.lower() or "kod" in c.lower()]
    if not teryt_candidates:
        # fallback: first column that looks like numeric codes
        teryt_candidates = [df.columns[0]]
    teryt_col = teryt_candidates[0]

    # --- Identify gmina name ---
    name_candidates = [c for c in df.columns if "gmina" in c.lower() or "nazwa" in c.lower()]
    name_col = name_candidates[0] if name_candidates else None

    # --- Uprawnionych / głosujących / ważnych ---
    def find_col(keywords: list[str], exclude: list[str] | None = None) -> Optional[str]:
        for kw in keywords:
            matches = [c for c in df.columns if kw in c.lower()]
            if exclude:
                matches = [c for c in matches if not any(ex in c.lower() for ex in exclude)]
            if matches:
                return matches[0]
        return None

    uprawn_col = find_col(["uprawnionych do głosowania", "liczba wyborców", "uprawni", "l. upr", "upr."])
    # Prefer aggregate "total issued" cards over pełnomocnik/zaświadczenie sub-counts
    glosuj_col = find_col(
        [
            "wydano karty do głosowania",   # 2019, 2023
            "kart wydanych",                # 2011: "Liczba kart wydanych"
            "wydane karty",                 # 2015: "Wydane karty"
            "oddanych kart",                # 2011: "Liczba oddanych kart"
            "gł. odd",                      # 2001-2007 XLS abbreviated headers
        ],
        exclude=["pełnomocnika", "zaświadczenia", "wysłano pakiety", "nie wykorzys"],
    )
    # Prefer explicit "głosy ważne" / "głosów ważnych oddanych łącznie" over generic "ważnych"
    waznych_col = find_col(
        [
            "głosów ważnych oddanych łącznie",  # 2019, 2023 CSV
            "głosy ważne",                      # 2011, 2015 XLS
            "głosów ważnych",                   # fallback
            "ważne",                            # 2001-2007 XLS: "Ważne" column
        ],
        exclude=["nieważnych", "nieważne", "nieważny"],
    )
    frekw_col = find_col(["frekwencja", "frekw"])

    out: dict[str, list] = {
        "teryt": [],
        "gmina_nazwa": [],
        "year": [],
        "uprawnionych": [],
        "glosujacych": [],
        "waznych": [],
        "frekwencja": [],
    }

    # Collect committee vote columns (int-ish columns after the main columns)
    fixed_cols = {teryt_col, name_col, uprawn_col, glosuj_col, waznych_col, frekw_col} - {None}
    # Infrastructure/metadata column keywords to exclude from committee votes
    _INFRA_KEYWORDS = [
        "nr okr", "nr okręgu", "komisja otrzymała", "nie wykorzystano",
        "pakiety wyborcze", "kopert", "urny", "kart wyjętych", "kart nieważnych",
        "kart ważnych", "głosów nieważnych", "głosów ważnych", "z powodu",
        "łącznie na wszystkie", "liczba komisji", "pełnomocnika", "zaświadczenia",
        "korespondenc", "wysłano pakiety", "powiat", "województwo",
    ]
    committee_cols = [
        c for c in df.columns
        if c not in fixed_cols and not any(kw in c.lower() for kw in _INFRA_KEYWORDS)
    ]
    # Filter to columns that are likely vote counts (numeric-ish)
    vote_cols: list[str] = []
    for c in committee_cols:
        try:
            pd.to_numeric(df[c].dropna().iloc[:5], errors="raise")
            vote_cols.append(c)
        except Exception:
            pass

    # Build committee vote dict placeholders
    committee_data: dict[str, list] = {}
    for vc in vote_cols:
        clean_name = re.sub(r"\s+", " ", vc).strip()
        committee_data[f"votes_{clean_name}"] = []
        committee_data[f"pct_{clean_name}"] = []

    def safe_int(val) -> int:
        try:
            # Use float() first to handle "28421.0" strings (pandas reads NaN-containing columns as float64)
            s = str(val).strip().replace(" ", "").replace("\xa0", "").replace(",", "")
            return int(float(s))
        except (ValueError, TypeError):
            return 0

    def safe_float(val) -> float:
        try:
            return float(str(val).replace(",", ".").replace(" ", ""))
        except (ValueError, TypeError):
            return 0.0

    for _, row in df.iterrows():
        teryt_raw = str(row[teryt_col]) if teryt_col else ""
        if not re.search(r"\d{4,}", teryt_raw):
            continue  # skip header-like rows

        out["teryt"].append(_normalize_teryt(teryt_raw))
        out["gmina_nazwa"].append(str(row[name_col]).strip() if name_col else "")
        out["year"].append(year)
        out["uprawnionych"].append(safe_int(row[uprawn_col]) if uprawn_col else 0)
        out["glosujacych"].append(safe_int(row[glosuj_col]) if glosuj_col else 0)
        waznych = safe_int(row[waznych_col]) if waznych_col else 0
        out["waznych"].append(waznych)
        if frekw_col:
            fval = safe_float(row[frekw_col])
            # Normalise: if value > 1 it's an absolute count (XLS files) → divide by uprawnionych
            upraw_val = out["uprawnionych"][-1]
            if fval > 1 and upraw_val > 0:
                fval = round(fval / upraw_val * 100, 2)
            out["frekwencja"].append(fval)
        else:
            upraw = out["uprawnionych"][-1]
            glos = out["glosujacych"][-1]
            out["frekwencja"].append(round(glos / upraw * 100, 2) if upraw else 0.0)

        total_valid = waznych if waznych > 0 else 1
        for vc in vote_cols:
            votes = safe_int(row[vc])
            clean_name = re.sub(r"\s+", " ", vc).strip()
            committee_data[f"votes_{clean_name}"].append(votes)
            committee_data[f"pct_{clean_name}"].append(round(votes / total_valid * 100, 4))

    result = pd.DataFrame({**out, **committee_data})

    # Add classification columns for each vote column
    for vc in vote_cols:
        clean_name = re.sub(r"\s+", " ", vc).strip()
        result[f"category_{clean_name}"] = classify_committee(clean_name)

    return result


def parse_election_results(year: int, filepath: Path) -> pd.DataFrame:
    """Parse raw election file(s) for *year* into a standardised DataFrame.

    Parameters
    ----------
    year:
        Election year.  Supported: 2001, 2005, 2007, 2011, 2015, 2019, 2023.
    filepath:
        Path to a downloaded raw file (ZIP, CSV, or XLSX).

    Returns
    -------
    pd.DataFrame
        Columns: teryt, gmina_nazwa, year, uprawnionych, glosujacych,
        waznych, frekwencja, votes_<committee>, pct_<committee>,
        category_<committee>.

    Raises
    ------
    ValueError
        If the file cannot be parsed or year is unsupported.
    """
    log.info("Parsing %d from %s", year, filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Raw file not found: {filepath}")

    frames: list[pd.DataFrame] = []

    if filepath.suffix.lower() == ".zip":
        with zipfile.ZipFile(filepath) as zf:
            members = zf.namelist()
            log.debug("ZIP members: %s", members[:20])
            csv_members = [
                m for m in members
                if m.lower().endswith((".csv", ".xlsx", ".xls"))
                and "gmina" in m.lower()
            ]
            if not csv_members:
                # Fallback: any CSV/XLSX
                csv_members = [m for m in members if m.lower().endswith((".csv", ".xlsx", ".xls"))]
            for member in csv_members:
                log.info("  Reading member: %s", member)
                raw_bytes = zf.read(member)
                try:
                    df_raw = _read_tabular(raw_bytes)
                    parsed = _parse_generic(df_raw, year)
                    if len(parsed) > 10:
                        frames.append(parsed)
                except Exception as exc:
                    log.warning("  Skipping %s: %s", member, exc)
    else:
        df_raw = _read_tabular(filepath)
        frames.append(_parse_generic(df_raw, year))

    if not frames:
        raise ValueError(f"No parseable data found in {filepath}")

    result = pd.concat(frames, ignore_index=True)
    # Deduplicate by teryt (keep first)
    result = result.drop_duplicates(subset=["teryt"]).reset_index(drop=True)
    log.info("  → %d gminas for year %d", len(result), year)
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(force: bool = False) -> None:
    """Download all election data and build the interim panel.

    Parameters
    ----------
    force:
        If True, re-download files even if they already exist.
    """
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    all_frames: list[pd.DataFrame] = []

    for year, sources in ELECTION_SOURCES.items():
        year_dir = RAW_DIR / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        parsed_path = year_dir / f"elections_{year}_parsed.parquet"

        if parsed_path.exists() and not force:
            log.info("Loading cached parse for %d", year)
            all_frames.append(pd.read_parquet(parsed_path))
            continue

        year_frames: list[pd.DataFrame] = []
        for url, fname in sources:
            dest = year_dir / fname
            try:
                download_file(url, dest, force=force)
            except requests.RequestException:
                log.error("Could not download %s — skipping year %d", url, year)
                continue

            try:
                df = parse_election_results(year, dest)
                year_frames.append(df)
            except Exception as exc:
                log.error("Parse error for year %d: %s", year, exc)

        if year_frames:
            df_year = pd.concat(year_frames, ignore_index=True).drop_duplicates("teryt")
            df_year.to_parquet(parsed_path, index=False)
            all_frames.append(df_year)
        else:
            log.warning("No data parsed for year %d", year)

    if not all_frames:
        log.error("No election data collected — panel not written.")
        return

    panel = pd.concat(all_frames, ignore_index=True)
    out_path = INTERIM_DIR / "elections_panel_raw.parquet"
    panel.to_parquet(out_path, index=False)
    log.info("Panel saved: %s  (%d rows, %d years)", out_path, len(panel), panel["year"].nunique())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PKW/KBW Sejm election data.")
    parser.add_argument("--force", action="store_true", help="Re-download and re-parse all files.")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    run(force=args.force)
