from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
import requests

# ChEMBL's own release resource is the temporal authority. ChEMBL 33 introduced
# CHEMBL_RELEASE(CHEMBL_RELEASE, CREATION_DATE); the public API exposes it.
CHEMBL_RELEASE_API_URL = 'https://www.ebi.ac.uk/chembl/api/data/chembl_release.json?limit=1000'
FTP_RELEASE_DIR = 'https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_{token}/'


def _normalise_release_label(value) -> str:
    """Normalise CHEMBL_2 / CHEMBL02 / 2 / 22.1 -> CHEMBL02 / CHEMBL22.1."""
    raw = str(value or '').strip().upper().replace('CHEMBL', '').replace('_', '')
    if not raw:
        return ''
    try:
        if '.' in raw:
            a, b = raw.split('.', 1)
            return f'CHEMBL{int(a):02d}.{b}'
        return f'CHEMBL{int(float(raw)):02d}'
    except Exception:
        return ''


def _coerce_iso_date(value) -> str | None:
    """Return YYYY-MM-DD from a date/datetime-like value, else None."""
    if value is None:
        return None
    s = str(value).strip()
    m = re.match(r'^(\d{4}-\d{2}-\d{2})', s)
    return m.group(1) if m else None


def _iter_dicts(obj):
    """Yield nested dictionaries so API envelope naming can change safely."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_dicts(v)


def _extract_chembl_release_dates(payload: dict) -> dict[str, str]:
    """Extract {CHEMBLxx: YYYY-MM-DD} from ChEMBL chembl_release JSON.

    Parsing is deliberately envelope-agnostic. A record is accepted only when
    one dictionary contains both a release-like field and a creation-date-like
    field. Conflicting exact dates for the same release fail closed.
    """
    out: dict[str, str] = {}
    release_keys = {'chembl_release', 'release', 'release_name'}
    date_keys = {'creation_date', 'release_creation_date'}
    for d in _iter_dicts(payload):
        lower = {str(k).lower(): v for k, v in d.items()}
        rvalue = next((lower[k] for k in release_keys if k in lower), None)
        dvalue = next((lower[k] for k in date_keys if k in lower), None)
        label = _normalise_release_label(rvalue)
        dt = _coerce_iso_date(dvalue)
        if not label or not dt:
            continue
        if label in out and out[label] != dt:
            raise RuntimeError(f'Conflicting ChEMBL creation dates for {label}: {out[label]} vs {dt}')
        out[label] = dt
    return out


def fetch_chembl_release_dates(timeout: int = 30, session=None) -> tuple[dict[str, str], dict]:
    """Fetch exact release creation dates from the official ChEMBL API."""
    getter = session.get if session is not None else requests.get
    resp = getter(CHEMBL_RELEASE_API_URL, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    dates = _extract_chembl_release_dates(payload)
    if not dates:
        raise RuntimeError('Official ChEMBL chembl_release resource returned no parseable creation dates.')
    audit = {
        'source': 'ChEMBL chembl_release API',
        'url': CHEMBL_RELEASE_API_URL,
        'n_release_dates': len(dates),
        'http_status': getattr(resp, 'status_code', None),
    }
    return dates, audit


def resolve_exact_release_dates(manifest: pd.DataFrame, overrides: pd.DataFrame | None = None,
                                timeout: int = 30, session=None) -> pd.DataFrame:
    """Resolve release dates using manual overrides, then ChEMBL CREATION_DATE.

    DOI registration metadata is intentionally *not* used for temporal layer
    assignment: the citable DOI may have been minted after the historical
    database release. The ChEMBL release creation date is the authority.
    """
    out = manifest.copy()
    if 'exact_release_date' not in out.columns:
        out['exact_release_date'] = pd.Series(pd.NA, index=out.index, dtype='string')
    else:
        out['exact_release_date'] = out['exact_release_date'].astype('string')
    for col in ['exact_date_source', 'exact_date_type', 'exact_date_field', 'exact_date_error']:
        if col not in out.columns:
            out[col] = pd.Series(pd.NA, index=out.index, dtype='string')
        else:
            out[col] = out[col].astype('string')

    # Explicit, versioned manual overrides remain highest priority.
    if overrides is not None and not overrides.empty:
        ov = overrides.copy()
        if 'exact_release_date' in ov.columns:
            ov = ov.dropna(subset=['exact_release_date'])
            for _, r in ov.iterrows():
                mask = out['release'].astype(str) == str(r['release'])
                out.loc[mask, 'exact_release_date'] = str(r['exact_release_date'])
                out.loc[mask, 'exact_date_source'] = str(r.get('source', 'manual_override'))
                out.loc[mask, 'exact_date_type'] = str(r.get('date_type', 'manual_override'))
                out.loc[mask, 'exact_date_field'] = str(r.get('date_field', 'manual_override'))

    unresolved = out['exact_release_date'].isna() | out['exact_release_date'].str.strip().eq('')
    if unresolved.any():
        try:
            official, _audit = fetch_chembl_release_dates(timeout=timeout, session=session)
            for idx in out.index[unresolved]:
                label = _normalise_release_label(out.at[idx, 'release_label'])
                dt = official.get(label)
                if dt:
                    out.at[idx, 'exact_release_date'] = dt
                    out.at[idx, 'exact_date_source'] = 'ChEMBL chembl_release API'
                    out.at[idx, 'exact_date_type'] = 'creation_date'
                    out.at[idx, 'exact_date_field'] = 'CHEMBL_RELEASE.CREATION_DATE'
                else:
                    out.at[idx, 'exact_date_error'] = 'release_missing_from_official_resource'
        except Exception as e:
            msg = f'{type(e).__name__}: {e}'
            out.loc[unresolved, 'exact_date_error'] = msg
            out.loc[unresolved & out['exact_date_source'].isna(), 'exact_date_source'] = 'unresolved:ChEMBL_release_API'
    return out


def validate_exact_dates(manifest: pd.DataFrame, require_exact: bool = True) -> list[str]:
    issues = []
    for _, r in manifest.iterrows():
        raw = r.get('exact_release_date')
        dt = '' if pd.isna(raw) else str(raw).strip()
        if require_exact and not re.fullmatch(r'\d{4}-\d{2}-\d{2}', dt):
            err = '' if pd.isna(r.get('exact_date_error')) else str(r.get('exact_date_error')).strip()
            suffix = f' [{err}]' if err else ''
            issues.append(f"{r['release_label']}: exact release date unresolved{suffix}")
    return issues


def discover_archive_candidates(release) -> list[str]:
    raw=str(release).upper().replace('CHEMBL','')
    if '.' in raw:
        a,b=raw.split('.',1); token=f'{int(a):02d}_{b}'
    else:
        token=f'{int(float(raw)):02d}'
    url = FTP_RELEASE_DIR.format(token=token)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    hrefs = re.findall(r'href=[\"\']([^\"\']+)[\"\']', resp.text, flags=re.I)
    candidates = []
    for h in hrefs:
        low = h.lower()
        if any(x in low for x in ['sqlite','mysql','postgres']) and not low.endswith('/'):
            candidates.append(url + h)
    return sorted(set(candidates))


def validate_local_archives(manifest: pd.DataFrame, root: Path) -> pd.DataFrame:
    out = manifest.copy()
    for idx, r in out.iterrows():
        rel = '' if pd.isna(r.get('local_db_path')) else str(r.get('local_db_path')).strip()
        if rel:
            p = (root / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
            out.at[idx, 'status'] = 'ready' if p.exists() else 'missing_local_file'
        else:
            out.at[idx, 'status'] = 'missing'
    return out
