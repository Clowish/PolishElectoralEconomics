"""
Download and parse Sejm election results (2001–2023) from PKW/KBW sources.

Data sources:
- 2019, 2023: wyniki.pkw.gov.pl (CSV/XLSX per gmina)
- 2001–2015: danewyborcze.kbw.gov.pl (CSV archives)

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
# PKW / KBW public archives (verified as of 2025).
ELECTION_SOURCES: dict[int, list[tuple[str, str]]] = {
    2001: [
        (
            "https://danewyborcze.kbw.gov.pl/resources/pliki/Wyniki_wyborow_sejmowych_2001.zip",
            "wyniki_2001.zip",
        )
    ],
    2005: [
        (
            "https://danewyborcze.kbw.gov.pl/resources/pliki/Wyniki_wyborow_sejmowych_2005.zip",
            "wyniki_2005.zip",
        )
    ],
    2007: [
        (
            "https://danewyborcze.kbw.gov.pl/resources/pliki/Wyniki_wyborow_sejmowych_2007.zip",
            "wyniki_2007.zip",
        )
    ],
    2011: [
        (
            "https://danewyborcze.kbw.gov.pl/resources/pliki/Wyniki_wyborow_sejmowych_2011.zip",
            "wyniki_2011.zip",
        )
    ],
    2015: [
        (
            "https://danewyborcze.kbw.gov.pl/resources/pliki/Wyniki_wyborow_sejmowych_2015.zip",
            "wyniki_2015.zip",
        )
    ],
    2019: [
        (
            "https://wyniki.pkw.gov.pl/wyniki/pub/510/Wyniki_GL_wr_okr_gm_do_sejmu_w_2019.zip",
            "wyniki_2019.zip",
        )
    ],
    2023: [
        (
            "https://wyniki.pkw.gov.pl/wyniki-sejm/pub/530/wyniki_gl_na_listy_po_gminach_sejm_2023.zip",
            "wyniki_2023.zip",
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
    def find_col(keywords: list[str]) -> Optional[str]:
        for kw in keywords:
            matches = [c for c in df.columns if kw in c.lower()]
            if matches:
                return matches[0]
        return None

    uprawn_col = find_col(["uprawni", "uprawnionych"])
    glosuj_col = find_col(["głosuj", "glosuj", "wydano"])
    waznych_col = find_col(["ważnych", "waznych", "głosów ważnych"])
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
    committee_cols = [c for c in df.columns if c not in fixed_cols]
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
            return int(str(val).replace(" ", "").replace(",", "").replace("\xa0", ""))
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
            out["frekwencja"].append(safe_float(row[frekw_col]))
        else:
            upraw = out["uprawnionych"][-1]
            out["frekwencja"].append(round(out["glosujacych"][-1] / upraw * 100, 2) if upraw else 0.0)

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
