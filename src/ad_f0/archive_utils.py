from __future__ import annotations
from pathlib import Path
import gzip, shutil, tarfile, zipfile


def unpack_archive(path: Path, dest: Path) -> list[Path]:
    path=Path(path); dest=Path(dest); dest.mkdir(parents=True,exist_ok=True)
    low=path.name.lower()
    if tarfile.is_tarfile(path):
        with tarfile.open(path,'r:*') as tf: tf.extractall(dest)
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf: zf.extractall(dest)
    elif low.endswith('.gz'):
        out=dest/path.name[:-3]
        with gzip.open(path,'rb') as src, open(out,'wb') as dst: shutil.copyfileobj(src,dst)
    else:
        raise ValueError(f'Unsupported archive: {path}')
    return [p for p in dest.rglob('*') if p.is_file()]


def find_sqlite_candidates(root: Path) -> list[Path]:
    result=[]
    for p in Path(root).rglob('*'):
        if not p.is_file(): continue
        if p.suffix.lower() in {'.db','.sqlite','.sqlite3'} or 'sqlite' in p.name.lower():
            result.append(p)
    return sorted(result)
