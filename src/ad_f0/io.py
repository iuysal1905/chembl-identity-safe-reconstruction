from __future__ import annotations
from pathlib import Path
import json, yaml, hashlib
import pandas as pd


def project_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / 'config' / 'f0_config.yaml').exists():
            return candidate
    raise FileNotFoundError('Could not locate project root containing config/f0_config.yaml')


def load_config(root: Path | None = None) -> dict:
    root = root or project_root()
    with open(root / 'config' / 'f0_config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_dirs(root: Path, cfg: dict) -> None:
    for key, rel in cfg['paths'].items():
        if key.endswith('_dir'):
            (root / rel).mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path)
