from __future__ import annotations
from functools import lru_cache
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

_lfc = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
_norm = rdMolStandardize.Normalizer()
_reion = rdMolStandardize.Reionizer()
_uncharger = rdMolStandardize.Uncharger()
_taut = rdMolStandardize.TautomerEnumerator()
# Binding rule: stereochemistry must survive tautomer canonicalisation.
if hasattr(_taut, 'SetRemoveSp3Stereo'):
    _taut.SetRemoveSp3Stereo(False)
if hasattr(_taut, 'SetRemoveBondStereo'):
    _taut.SetRemoveBondStereo(False)


def _empty_result(input_repr: str | None, input_format: str) -> dict:
    return {
        'input_structure': input_repr,
        'input_format': input_format,
        'canonical_isomeric_smiles': None,
        'full_inchikey': None,
        'murcko_scaffold_smiles': None,
        'status': 'failed',
        'error': None,
    }


def _standardize_mol(mol: Chem.Mol, input_repr: str | None, input_format: str) -> dict:
    result = _empty_result(input_repr, input_format)
    try:
        if mol is None:
            raise ValueError('RDKit parse failure')
        mol = Chem.Mol(mol)
        mol = _lfc.choose(mol)
        mol = _norm.normalize(mol)
        mol = _reion.reionize(mol)
        mol = _uncharger.uncharge(mol)
        mol = _taut.Canonicalize(mol)
        Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        ik = Chem.MolToInchiKey(mol)
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        scaf_smi = Chem.MolToSmiles(scaf, canonical=True, isomericSmiles=True) if scaf.GetNumAtoms() else ''
        if not ik:
            raise ValueError('InChIKey generation failure')
        result.update(
            canonical_isomeric_smiles=smi,
            full_inchikey=ik,
            murcko_scaffold_smiles=scaf_smi,
            status='ok',
        )
    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'
    return result


SMILES_CACHE_SIZE = 50_000
MOLBLOCK_CACHE_SIZE = 10_000


def configure_rdkit_logging(show_info: bool = False) -> None:
    """Control RDKit informational chatter without hiding warnings/errors."""
    if show_info:
        RDLogger.EnableLog('rdApp.info')
    else:
        RDLogger.DisableLog('rdApp.info')


def clear_standardization_caches() -> None:
    """Release memoized structures between ChEMBL snapshots; results are unchanged."""
    standardize_smiles.cache_clear()
    standardize_molblock.cache_clear()


@lru_cache(maxsize=SMILES_CACHE_SIZE)
def standardize_smiles(smiles: str) -> dict:
    text = '' if smiles is None else str(smiles)
    try:
        mol = Chem.MolFromSmiles(text)
    except Exception:
        mol = None
    out = _standardize_mol(mol, text, 'smiles')
    # Backward compatibility for previous audit consumers.
    out['input_smiles'] = text
    return out


@lru_cache(maxsize=MOLBLOCK_CACHE_SIZE)
def standardize_molblock(molblock: str) -> dict:
    text = '' if molblock is None else str(molblock)
    try:
        # sanitize=False lets the common standardisation path perform sanitisation after cleanup.
        mol = Chem.MolFromMolBlock(text, sanitize=False, removeHs=False, strictParsing=False)
        if mol is not None:
            Chem.SanitizeMol(mol)
    except Exception:
        mol = None
    return _standardize_mol(mol, text, 'molblock')


def standardize_structure(smiles: str | None = None, molblock: str | None = None) -> dict:
    """Standardize historical ChEMBL structures with a deterministic source preference.

    Canonical SMILES is preferred when available. Pre-ChEMBL-09 releases may only expose the
    original molfile in ``compounds``; in that case the mol block is parsed directly. The same
    downstream parent/charge/tautomer/stereo policy is applied to both inputs.
    """
    if smiles is not None and str(smiles).strip() and str(smiles).strip().lower() not in {'nan', 'none'}:
        r = standardize_smiles(str(smiles))
        if r['status'] == 'ok':
            return r
    if molblock is not None and str(molblock).strip() and str(molblock).strip().lower() not in {'nan', 'none'}:
        return standardize_molblock(str(molblock))
    return _empty_result(None, 'missing')

# Chemical algorithm version, separate from notebook/extraction policy. Increment only if
# parent/charge/tautomer/stereo semantics change. Runtime/cache changes do not change this id.
STRUCTURE_STANDARDIZATION_POLICY_VERSION = 'F0-structure-v1.5i-r1'


def _present_structure(v) -> str:
    if v is None:
        return ''
    try:
        if v != v:  # NaN
            return ''
    except Exception:
        pass
    text = str(v)
    return '' if text.strip().lower() in {'', 'nan', 'none'} else text


def _structure_sha256(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _cache_connect(path):
    import sqlite3
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=60)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS structure_standardization_cache (
            policy_version TEXT NOT NULL,
            input_format TEXT NOT NULL,
            input_sha256 TEXT NOT NULL,
            input_length INTEGER NOT NULL,
            canonical_isomeric_smiles TEXT,
            full_inchikey TEXT,
            murcko_scaffold_smiles TEXT,
            status TEXT NOT NULL,
            error TEXT,
            PRIMARY KEY (policy_version, input_format, input_sha256)
        )
    ''')
    return conn


def _resolve_unique_inputs(texts, input_format: str, cache_db_path=None, progress_label: str | None = None):
    """Resolve each unique exact input once, optionally through a persistent content-addressed cache."""
    import time
    unique = list(dict.fromkeys(t for t in texts if t))
    resolved = {}
    hits = 0
    new = 0
    conn = _cache_connect(cache_db_path) if cache_db_path else None
    hashes = {t: _structure_sha256(t) for t in unique}

    if conn is not None and unique:
        # Fetch existing cache rows in conservative chunks below SQLite's parameter ceiling.
        inv = {}
        for t, h in hashes.items():
            inv.setdefault(h, []).append(t)
        hs = list(inv)
        for start in range(0, len(hs), 400):
            chunk = hs[start:start+400]
            marks = ','.join('?' for _ in chunk)
            q = f'''SELECT input_sha256,input_length,canonical_isomeric_smiles,full_inchikey,
                           murcko_scaffold_smiles,status,error
                    FROM structure_standardization_cache
                    WHERE policy_version=? AND input_format=? AND input_sha256 IN ({marks})'''
            params = [STRUCTURE_STANDARDIZATION_POLICY_VERSION, input_format, *chunk]
            for row in conn.execute(q, params):
                h, input_length, smi, ik, scaf, status, error = row
                for text in inv.get(h, []):
                    if len(text) != int(input_length):
                        raise RuntimeError('Structure-cache SHA-256 collision/length mismatch; fail closed.')
                    resolved[text] = {
                        'input_structure': text,
                        'input_format': input_format,
                        'canonical_isomeric_smiles': smi,
                        'full_inchikey': ik,
                        'murcko_scaffold_smiles': scaf,
                        'status': status,
                        'error': error,
                    }
                    hits += 1

    missing = [t for t in unique if t not in resolved]
    started = time.time()
    pending = []
    for i, text in enumerate(missing, 1):
        out = standardize_smiles(text) if input_format == 'smiles' else standardize_molblock(text)
        # Store only the stable result fields used downstream.
        rec = {
            'input_structure': text,
            'input_format': input_format,
            'canonical_isomeric_smiles': out.get('canonical_isomeric_smiles'),
            'full_inchikey': out.get('full_inchikey'),
            'murcko_scaffold_smiles': out.get('murcko_scaffold_smiles'),
            'status': out.get('status', 'failed'),
            'error': out.get('error'),
        }
        resolved[text] = rec
        new += 1
        if conn is not None:
            pending.append((
                STRUCTURE_STANDARDIZATION_POLICY_VERSION, input_format, hashes[text], len(text),
                rec['canonical_isomeric_smiles'], rec['full_inchikey'], rec['murcko_scaffold_smiles'],
                rec['status'], rec['error'],
            ))
            if len(pending) >= 100:
                conn.executemany('''INSERT OR REPLACE INTO structure_standardization_cache
                    (policy_version,input_format,input_sha256,input_length,canonical_isomeric_smiles,
                     full_inchikey,murcko_scaffold_smiles,status,error)
                    VALUES (?,?,?,?,?,?,?,?,?)''', pending)
                conn.commit(); pending.clear()
        if progress_label and (i == 1 or i % 250 == 0 or i == len(missing)):
            elapsed = time.time() - started
            print(f'{progress_label}: standardized {i}/{len(missing)} new unique {input_format} inputs '
                  f'({hits} persistent-cache hits; {elapsed:.1f}s)')
    if conn is not None:
        if pending:
            conn.executemany('''INSERT OR REPLACE INTO structure_standardization_cache
                (policy_version,input_format,input_sha256,input_length,canonical_isomeric_smiles,
                 full_inchikey,murcko_scaffold_smiles,status,error)
                VALUES (?,?,?,?,?,?,?,?,?)''', pending)
            conn.commit()
        conn.close()
    return resolved, {'unique_inputs': len(unique), 'persistent_cache_hits': hits, 'standardized_new': new}


def standardize_structure_batch(smiles_values, molblock_values, *, cache_db_path=None, progress_label: str | None = None):
    """Vector-equivalent standardisation with exact source preference and unique-input reuse.

    Semantics are identical to ``standardize_structure``: try canonical SMILES first; if it is
    missing or fails, fall back to molblock. The optimization only changes how often an identical
    raw structure is computed, not the chemistry algorithm or returned identity.
    """
    import time
    smiles = [_present_structure(v) for v in smiles_values]
    molblocks = [_present_structure(v) for v in molblock_values]
    if len(smiles) != len(molblocks):
        raise ValueError('smiles_values and molblock_values must have identical length')
    started = time.time()
    smi_map, smi_stats = _resolve_unique_inputs(smiles, 'smiles', cache_db_path, progress_label)
    results = [None] * len(smiles)
    fallback_molblocks = []
    for i, smi in enumerate(smiles):
        if smi:
            r = smi_map[smi]
            if r.get('status') == 'ok':
                results[i] = r
                continue
        if molblocks[i]:
            fallback_molblocks.append(molblocks[i])
    mb_map, mb_stats = _resolve_unique_inputs(fallback_molblocks, 'molblock', cache_db_path, progress_label)
    for i, r in enumerate(results):
        if r is not None:
            continue
        mb = molblocks[i]
        if mb:
            results[i] = mb_map[mb]
        else:
            results[i] = _empty_result(None, 'missing')
    stats = {
        'input_rows': len(results),
        'unique_smiles_inputs': smi_stats['unique_inputs'],
        'unique_molblock_fallback_inputs': mb_stats['unique_inputs'],
        'persistent_cache_hits': smi_stats['persistent_cache_hits'] + mb_stats['persistent_cache_hits'],
        'standardized_new': smi_stats['standardized_new'] + mb_stats['standardized_new'],
        'standardization_seconds': time.time() - started,
        'policy_version': STRUCTURE_STANDARDIZATION_POLICY_VERSION,
    }
    return results, stats
