from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import shutil
import tarfile
import tempfile
import requests

CHEMBL_RELEASE_ROOT = 'https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases'


def format_release_token(version: str | int | float) -> str:
    raw = str(version).upper().replace('CHEMBL', '').strip().replace('.', '_')
    if '_' in raw:
        major, minor = raw.split('_', 1)
        return f'{int(major):02d}_{minor}'
    return f'{int(float(raw)):02d}'


def sqlite_archive_urls(version: str | int | float) -> list[str]:
    token = format_release_token(version)
    filename = f'chembl_{token}_sqlite.tar.gz'
    base = f'{CHEMBL_RELEASE_ROOT}/chembl_{token}'
    return [f'{base}/{filename}', f'{base}/archived/{filename}']


def _sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _safe_extract_single_db(tar_path: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, 'r:*') as tf:
        members = [m for m in tf.getmembers() if m.isfile() and Path(m.name).suffix.lower() in {'.db','.sqlite','.sqlite3'}]
        if len(members) != 1:
            raise RuntimeError(f'Expected exactly one SQLite DB in {tar_path.name}; found {[m.name for m in members]}')
        m = members[0]
        out = dest_dir / Path(m.name).name
        src = tf.extractfile(m)
        if src is None:
            raise RuntimeError(f'Cannot extract {m.name}')
        with out.open('wb') as dst:
            shutil.copyfileobj(src, dst)
    return out


@dataclass(frozen=True)
class DownloadedSQLite:
    version: str
    url: str
    archive_path: Path
    db_path: Path
    archive_sha256: str
    db_sha256: str


def download_extract_sqlite(
    version: str | int | float,
    archive_dir: Path,
    db_dir: Path | None = None,
    *,
    timeout: int = 90,
    chunk_size: int = 1024 * 1024,
    retain_archive: bool = True,
) -> DownloadedSQLite:
    """Download one official ChEMBL backfilled SQLite archive and extract its DB.

    ChEMBL announced official SQLite backfills for all releases (v1 onward) in September 2025.
    The two official release-directory URL forms used by ``chembl-downloader`` are tried.
    Existing DB files are reused; no bulk loop is hidden inside this helper.
    """
    token = format_release_token(version)
    archive_dir = Path(archive_dir)
    db_dir = Path(db_dir) if db_dir is not None else archive_dir
    archive_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f'chembl_{token}_sqlite.tar.gz'
    expected_db = db_dir / f'chembl_{token}.db'

    chosen_url = ''
    if not expected_db.exists():
        if not archive_path.exists():
            last_error = None
            for url in sqlite_archive_urls(version):
                try:
                    with requests.get(url, stream=True, timeout=timeout) as r:
                        r.raise_for_status()
                        with tempfile.NamedTemporaryFile('wb', delete=False, dir=archive_dir, prefix=f'.chembl_{token}_', suffix='.part') as tmp:
                            tmp_path = Path(tmp.name)
                            for block in r.iter_content(chunk_size=chunk_size):
                                if block:
                                    tmp.write(block)
                    tmp_path.replace(archive_path)
                    chosen_url = url
                    break
                except Exception as e:
                    last_error = e
                    try:
                        if 'tmp_path' in locals() and tmp_path.exists():
                            tmp_path.unlink()
                    except Exception:
                        pass
            else:
                raise RuntimeError(f'Unable to download ChEMBL {version} SQLite from official URLs: {last_error}')
        db = _safe_extract_single_db(archive_path, db_dir)
        if db != expected_db:
            if expected_db.exists():
                expected_db.unlink()
            db.replace(expected_db)
    else:
        db = expected_db

    archive_hash = _sha256(archive_path) if archive_path.exists() else ''
    db_hash = _sha256(expected_db)
    if not retain_archive and archive_path.exists():
        archive_path.unlink()
    return DownloadedSQLite(
        version=str(version),
        url=chosen_url or 'local-cache',
        archive_path=archive_path,
        db_path=expected_db,
        archive_sha256=archive_hash,
        db_sha256=db_hash,
    )
