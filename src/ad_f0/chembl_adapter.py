from __future__ import annotations
from pathlib import Path
import hashlib
import re
import sqlite3
import pandas as pd

from .schema import require_tables, pick_column, columns, table_names
from .standardize import standardize_structure, standardize_structure_batch
from .units import to_molar, pactivity_from_molar

# ChEMBL has three materially different schema families relevant to the locked temporal design:
#   v1-v8: compounds / early assay2target schema (no molecule_dictionary)
#   v9-v14: molecule_dictionary + compound_structures, assay2target target mapping
#   v15+: assays.tid + target_components/component_sequences
POST9_BASE_REQUIRED = ['activities', 'assays', 'target_dictionary', 'compound_structures', 'molecule_dictionary', 'docs']
POST15_REQUIRED = POST9_BASE_REQUIRED + ['target_components', 'component_sequences']
PRE15_REQUIRED = POST9_BASE_REQUIRED + ['assay2target']
PRE9_REQUIRED = ['activities', 'assays', 'target_dictionary', 'assay2target', 'compounds', 'docs']

HISTORICAL_SINGLE_PROTEIN_NATIVE_LABEL = {
    'pre9_early': 'PROTEIN',
    'pre15_legacy': 'PROTEIN',
    'post15': 'SINGLE PROTEIN',
}


class MultiComponentThresholdError(RuntimeError):
    """Raised when multi-accession assay exclusions exceed the preregistered release threshold."""
    def __init__(self, message: str, audit: pd.DataFrame):
        super().__init__(message)
        self.audit = audit


class HistoricalSchemaSemanticsError(RuntimeError):
    """Raised when a historical schema can be read but its locked target semantics cannot be certified."""


def _col_expr(alias: str, col: str | None, out: str) -> str:
    return f'{alias}."{col}" AS "{out}"' if col else f'NULL AS "{out}"'


def _norm_text(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ''
    return re.sub(r'\s+', ' ', str(v).strip()).upper()


def _sha(parts: list[str]) -> str:
    return hashlib.sha256('\x1f'.join(parts).encode('utf-8')).hexdigest()


def detect_schema_generation(conn: sqlite3.Connection) -> str:
    """Detect the historical ChEMBL schema by introspection rather than release number alone."""
    have = table_names(conn)
    if not {'activities', 'assays', 'target_dictionary', 'docs'}.issubset(have):
        missing = sorted({'activities', 'assays', 'target_dictionary', 'docs'} - have)
        raise RuntimeError(f'Missing core ChEMBL tables: {missing}')

    assay_cols = columns(conn, 'assays')
    td_cols = columns(conn, 'target_dictionary')

    if {'target_components', 'component_sequences'}.issubset(have) and {'tid', 'confidence_score'}.issubset(assay_cols):
        require_tables(conn, POST15_REQUIRED)
        return 'post15'

    if 'assay2target' in have and 'molecule_dictionary' in have and 'compound_structures' in have:
        require_tables(conn, PRE15_REQUIRED)
        return 'pre15_legacy'

    if 'assay2target' in have and 'compounds' in have and 'molecule_dictionary' not in have:
        require_tables(conn, PRE9_REQUIRED)
        return 'pre9_early'

    raise RuntimeError(
        'Unsupported/ambiguous historical ChEMBL schema. '
        f'tables={sorted(have)}; assays_cols={sorted(assay_cols)}; target_dictionary_cols={sorted(td_cols)}'
    )


def _native_target_type_for_generation(generation: str, requested_target_type: str) -> str:
    """Map the frozen scientific single-protein concept onto historical ChEMBL labels.

    ChEMBL 1-14 used the legacy target_dictionary label ``PROTEIN`` for protein
    targets. ChEMBL 15 introduced the modern target data model with explicit
    ``SINGLE PROTEIN`` / family / complex target types. This function is a
    schema-semantic adapter, not a biological widening rule.
    """
    requested = _norm_text(requested_target_type)
    if requested != 'SINGLE PROTEIN':
        raise HistoricalSchemaSemanticsError(
            'F0 historical target adapter is frozen for requested target_type=SINGLE PROTEIN; '
            f'got {requested_target_type!r}'
        )
    try:
        return HISTORICAL_SINGLE_PROTEIN_NATIVE_LABEL[generation]
    except KeyError:
        raise HistoricalSchemaSemanticsError(f'Unsupported schema generation for target-type semantics: {generation}')


def _legacy_confidence_descriptor(conn: sqlite3.Connection, generation: str) -> dict:
    """Describe, but do not guess, historical confidence semantics."""
    if generation == 'post15':
        return {
            'confidence_column': 'confidence_score',
            'confidence_regime': 'modern_0_9',
            'direct_primary_value': 9,
            'direct_semantics_source': 'assays.confidence_score',
        }
    a2t_cols = columns(conn, 'assay2target')
    if 'confidence_score' in a2t_cols:
        return {
            'confidence_column': 'confidence_score',
            'confidence_regime': 'modern_0_9_legacy_location',
            'direct_primary_value': 9,
            'direct_semantics_source': 'assay2target.confidence_score',
        }
    if 'confidence' in a2t_cols:
        # The 2009 ChEMBL/StARlite SQL example explicitly uses confidence=7 as "high confidence".
        # It is not silently relabelled as modern score 9; primary equivalence additionally requires
        # a direct relationship flag when the historical schema supplies one.
        return {
            'confidence_column': 'confidence',
            'confidence_regime': 'early_0_7_high_confidence',
            'direct_primary_value': 7,
            'direct_semantics_source': 'assay2target.confidence + relationship_type_if_available',
        }
    raise HistoricalSchemaSemanticsError(
        f'{generation}: assay2target has neither confidence_score nor confidence; have={sorted(a2t_cols)}'
    )


def _target_identity_capability(conn: sqlite3.Connection, generation: str) -> dict:
    td = columns(conn, 'target_dictionary')
    if generation == 'post15':
        return {
            'target_accession_native': True,
            'target_sequence_native': 'sequence' in columns(conn, 'component_sequences'),
            'target_identity_mode': 'component_sequences.accession',
            'target_type_native': 'target_type' in td,
        }
    if 'protein_accession' in td:
        return {
            'target_accession_native': True,
            'target_sequence_native': 'protein_sequence' in td,
            'target_identity_mode': 'target_dictionary.protein_accession',
            'target_type_native': 'target_type' in td,
        }
    return {
        'target_accession_native': False,
        'target_sequence_native': 'protein_sequence' in td,
        'target_identity_mode': 'exact_sequence_bridge_required' if 'protein_sequence' in td else 'unsupported_no_accession_or_sequence',
        'target_type_native': 'target_type' in td,
    }


def inspect_schema(db_path: Path) -> dict:
    """Return an auditable schema snapshot for early, pre-15, or post-15 SQLite releases."""
    with sqlite3.connect(str(db_path)) as conn:
        generation = detect_schema_generation(conn)
        required = POST15_REQUIRED if generation == 'post15' else PRE15_REQUIRED if generation == 'pre15_legacy' else PRE9_REQUIRED
        require_tables(conn, required)
        a2t_cols = columns(conn, 'assay2target') if 'assay2target' in table_names(conn) else set()
        act_cols = columns(conn, 'activities')
        assay_cols = columns(conn, 'assays')
        compound_cols = columns(conn, 'compounds') if generation == 'pre9_early' else set()
        conf = _legacy_confidence_descriptor(conn, generation)
        ident = _target_identity_capability(conn, generation)
        return {
            'schema_generation': generation,
            'tables': {t: sorted(columns(conn, t)) for t in required},
            'assay_type_available': 'assay_type' in assay_cols,
            'relationship_type_available': generation == 'post15' or 'relationship_type' in a2t_cols,
            'confidence_column': conf['confidence_column'],
            'confidence_regime': conf['confidence_regime'],
            'direct_primary_value': conf['direct_primary_value'],
            'target_accession_native': ident['target_accession_native'],
            'target_sequence_native': ident['target_sequence_native'],
            'target_identity_mode': ident['target_identity_mode'],
            'target_type_native': bool(ident['target_type_native']),
            'activity_endpoint_column': 'standard_type' if 'standard_type' in act_cols else 'activity_type' if 'activity_type' in act_cols else None,
            'activity_relation_column': 'standard_relation' if 'standard_relation' in act_cols else 'relation' if 'relation' in act_cols else None,
            'compound_smiles_available': generation != 'pre9_early' or 'canonical_smiles' in compound_cols,
            'compound_molfile_available': generation == 'pre9_early' and 'molfile' in compound_cols,
            'molecule_dictionary_available': 'molecule_dictionary' in table_names(conn),
            'requested_scientific_target_type': 'SINGLE PROTEIN',
            'historical_native_target_type_equivalent': _native_target_type_for_generation(generation, 'SINGLE PROTEIN'),
            'potential_duplicate_available': 'potential_duplicate' in act_cols,
            'data_validity_comment_available': 'data_validity_comment' in act_cols,
        }


def _linked_document_metadata_profile(db_path: Path) -> dict:
    """Metadata-only completeness profile for documents referenced by assays in a probe release."""
    with sqlite3.connect(str(db_path)) as conn:
        assay_doc=pick_column(conn,'assays',['doc_id'],required=False)
        d_doc=pick_column(conn,'docs',['doc_id'],required=False)
        if not assay_doc or not d_doc:
            return {
                'linked_document_count':0,
                'document_year_null_rate':float('nan'),
                'document_title_null_rate':float('nan'),
                'document_first_page_null_rate':float('nan'),
                'document_weak_alias_bridgeable_rate':float('nan'),
            }
        dcols=columns(conn,'docs')
        wanted={k:(k if k in dcols else None) for k in ['year','title','first_page','journal','volume']}
        sel=[f'doc."{d_doc}" AS doc_id']
        for k,c in wanted.items():
            sel.append(f'doc."{c}" AS "{k}"' if c else f'NULL AS "{k}"')
        select_sql=', '.join(sel)
        q=(f'SELECT DISTINCT {select_sql} '
           f'FROM assays aa JOIN docs doc ON aa."{assay_doc}"=doc."{d_doc}" '
           f'WHERE aa."{assay_doc}" IS NOT NULL')
        x=pd.read_sql_query(q,conn)
    n=len(x)
    if n==0:
        return {
            'linked_document_count':0,
            'document_year_null_rate':float('nan'),
            'document_title_null_rate':float('nan'),
            'document_first_page_null_rate':float('nan'),
            'document_weak_alias_bridgeable_rate':float('nan'),
        }
    def present(c):
        return x[c].map(lambda v: bool(_norm_text(v)))
    year=present('year'); title=present('title'); first=present('first_page')
    journal=present('journal'); volume=present('volume')
    bridgeable=(year & title) | (year & journal & first) | (journal & volume & first)
    return {
        'linked_document_count':int(n),
        'document_year_null_rate':float((~year).mean()),
        'document_title_null_rate':float((~title).mean()),
        'document_first_page_null_rate':float((~first).mean()),
        'document_weak_alias_bridgeable_rate':float(bridgeable.mean()),
    }


def schema_probe_row(db_path: Path, release_label: str | None = None) -> dict:
    """Compact row used by the mandatory early/pre15/post15 preflight schema probe."""
    info = inspect_schema(db_path)
    doc_profile=_linked_document_metadata_profile(db_path)
    early = info['schema_generation'] == 'pre9_early'
    relationship_requirement = (not early) or bool(info['relationship_type_available'])
    structure_ok = info['molecule_dictionary_available'] or info['compound_smiles_available'] or info['compound_molfile_available']
    target_identity_structurally_possible = (info['target_accession_native'] and info['target_type_native']) or info['target_sequence_native']
    native_target_type = info['historical_native_target_type_equivalent']
    with sqlite3.connect(str(db_path)) as conn:
        td_type = pick_column(conn, 'target_dictionary', ['target_type'], required=False)
        td_org = pick_column(conn, 'target_dictionary', ['organism'], required=False)
        if td_type and td_org:
            q = f'''SELECT COUNT(*) FROM target_dictionary
                    WHERE UPPER(TRIM("{td_type}"))=? AND "{td_org}"=?'''
            target_type_equivalent_target_count = int(conn.execute(q, (native_target_type, 'Homo sapiens')).fetchone()[0])
        else:
            target_type_equivalent_target_count = None
    target_type_semantics_probe_pass = (
        (target_type_equivalent_target_count is not None and target_type_equivalent_target_count > 0)
        or (early and td_type is None and bool(info['target_sequence_native']))
    )
    return {
        'release_label': release_label,
        'db_path': str(db_path),
        **doc_profile,
        'schema_generation': info['schema_generation'],
        'assay_type_available': bool(info['assay_type_available']),
        'relationship_type_available': bool(info['relationship_type_available']),
        'confidence_column': info['confidence_column'],
        'confidence_regime': info['confidence_regime'],
        'direct_primary_value': info['direct_primary_value'],
        'target_identity_mode': info['target_identity_mode'],
        'target_accession_native': bool(info['target_accession_native']),
        'target_sequence_native': bool(info['target_sequence_native']),
        'target_type_native': bool(info['target_type_native']),
        'requested_scientific_target_type': info['requested_scientific_target_type'],
        'historical_native_target_type_equivalent': info['historical_native_target_type_equivalent'],
        'target_type_equivalent_human_target_count': target_type_equivalent_target_count,
        'target_type_semantics_probe_pass': target_type_semantics_probe_pass,
        'activity_endpoint_column': info['activity_endpoint_column'],
        'activity_relation_column': info['activity_relation_column'],
        'compound_smiles_available': bool(info['compound_smiles_available']),
        'compound_molfile_available': bool(info['compound_molfile_available']),
        'potential_duplicate_available': bool(info['potential_duplicate_available']),
        'data_validity_comment_available': bool(info['data_validity_comment_available']),
        # For v1-v8, a structurally readable DB can still be scientifically uncertified for the primary
        # direct-target track if relationship_type is absent. Probe reports this separately.
        'schema_structural_probe_pass': bool(
            info['assay_type_available'] and info['activity_endpoint_column'] and info['activity_relation_column']
            and structure_ok and target_identity_structurally_possible and target_type_semantics_probe_pass
        ),
        'primary_semantics_probe_pass': bool(
            info['assay_type_available'] and info['activity_endpoint_column'] and info['activity_relation_column']
            and structure_ok and target_identity_structurally_possible and relationship_requirement and target_type_semantics_probe_pass
        ),
        # Compatibility aggregate: both structural readability and the locked primary target semantics.
        'schema_probe_pass': bool(
            info['assay_type_available'] and info['activity_endpoint_column'] and info['activity_relation_column']
            and structure_ok and target_identity_structurally_possible and relationship_requirement and target_type_semantics_probe_pass
        ),
    }


def _normalize_sequence(s) -> str:
    if s is None or pd.isna(s):
        return ''
    return re.sub(r'[^A-Za-z]', '', str(s)).upper()


def build_target_sequence_bridge(reference_db_path: Path) -> pd.DataFrame:
    """Build a one-to-one exact-sequence → UniProt identity bridge from a fixed historical reference DB.

    The bridge uses target identity metadata only; no activity row is queried. It is intended only for
    early releases whose target_dictionary contains a protein sequence but no UniProt accession.
    Ambiguous sequence/organism combinations are retained with ``bridge_status='ambiguous'`` and are not
    eligible for automatic mapping.
    """
    with sqlite3.connect(str(reference_db_path)) as conn:
        gen = detect_schema_generation(conn)
        if gen == 'post15':
            td_tid = pick_column(conn, 'target_dictionary', ['tid'])
            td_type = pick_column(conn, 'target_dictionary', ['target_type'])
            td_org = pick_column(conn, 'target_dictionary', ['organism'])
            tc_tid = pick_column(conn, 'target_components', ['tid'])
            tc_comp = pick_column(conn, 'target_components', ['component_id'])
            cs_comp = pick_column(conn, 'component_sequences', ['component_id'])
            cs_acc = pick_column(conn, 'component_sequences', ['accession'])
            cs_seq = pick_column(conn, 'component_sequences', ['sequence'])
            q = f'''SELECT td."{td_org}" AS organism, td."{td_type}" AS target_type,
                           seq."{cs_acc}" AS target_accession, seq."{cs_seq}" AS target_sequence
                    FROM target_dictionary td
                    JOIN target_components tc ON td."{td_tid}"=tc."{tc_tid}"
                    JOIN component_sequences seq ON tc."{tc_comp}"=seq."{cs_comp}"
                    WHERE seq."{cs_acc}" IS NOT NULL AND seq."{cs_seq}" IS NOT NULL'''
        else:
            td = columns(conn, 'target_dictionary')
            if 'protein_accession' not in td or 'protein_sequence' not in td:
                raise HistoricalSchemaSemanticsError(
                    'Reference target bridge DB must contain target_dictionary.protein_accession and protein_sequence'
                )
            td_org = pick_column(conn, 'target_dictionary', ['organism'])
            td_type = pick_column(conn, 'target_dictionary', ['target_type'], required=False)
            q = f'''SELECT td."{td_org}" AS organism,
                           {_col_expr('td', td_type, 'target_type')},
                           td."protein_accession" AS target_accession,
                           td."protein_sequence" AS target_sequence
                    FROM target_dictionary td
                    WHERE td."protein_accession" IS NOT NULL AND td."protein_sequence" IS NOT NULL'''
        raw = pd.read_sql_query(q, conn)
    raw['sequence_normalized'] = raw['target_sequence'].map(_normalize_sequence)
    raw['organism_normalized'] = raw['organism'].map(_norm_text)
    raw = raw[raw.sequence_normalized.ne('') & raw.target_accession.notna()].copy()
    g = (raw.groupby(['sequence_normalized','organism_normalized'], dropna=False)
         .agg(target_accessions=('target_accession', lambda x: '|'.join(sorted(set(map(str,x))))),
              n_accessions=('target_accession', lambda x: len(set(map(str,x)))),
              target_types=('target_type', lambda x: '|'.join(sorted(set(str(v) for v in x if pd.notna(v))))))
         .reset_index())
    g['bridge_status'] = g['n_accessions'].eq(1).map({True:'unique', False:'ambiguous'})
    g['target_accession'] = g['target_accessions'].where(g['n_accessions'].eq(1))
    g['target_type_bridge'] = g['target_types'].where(g['n_accessions'].eq(1))
    return g


def _potential_duplicate_flag(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors='coerce')
    text = series.fillna('').astype(str).str.strip().str.upper()
    return numeric.eq(1) | text.isin({'1', '1.0', 'Y', 'YES', 'TRUE', 'T'})


def _validity_eligible(series: pd.Series) -> pd.Series:
    s = series.fillna('').astype(str).str.strip().str.upper()
    return s.isin({'', 'MANUALLY VALIDATED'})


def _assay_target_mapping_audit_rows(valid: pd.DataFrame, id_col: str, release_label: str | None,
                                     candidate_assays: int, excluded_assays: int, fraction: float,
                                     *, exclusion_reason: str, threshold_applicable: bool) -> pd.DataFrame:
    rows=[]
    n=valid.groupby(id_col)['target_accession'].nunique()
    bad=n[n > 1]
    for assay_key,n_acc in bad.items():
        z=valid.loc[valid[id_col].eq(assay_key)].copy()
        rel=z.get('target_relationship_type',pd.Series('',index=z.index)).fillna('').astype(str).str.upper()
        d_targets=sorted(z.loc[rel.eq('D'),'target_accession'].dropna().astype(str).unique())
        h_targets=sorted(z.loc[rel.eq('H'),'target_accession'].dropna().astype(str).unique())
        acc=sorted(z.target_accession.dropna().astype(str).unique())
        rows.append({
            'release_label':release_label,
            'native_assay_id':assay_key if id_col=='native_assay_id' else None,
            'assay_chembl_id':'|'.join(sorted(z.get('assay_chembl_id',pd.Series(dtype=object)).dropna().astype(str).unique())),
            'assay_type':'|'.join(sorted(z.get('assay_type',pd.Series(dtype=object)).dropna().astype(str).unique())),
            'n_target_accessions':int(n_acc),
            'n_D_targets':len(d_targets),
            'n_H_targets':len(h_targets),
            'relationship_types':'|'.join(sorted(set(rel[rel.ne('')]))),
            'confidence_values':'|'.join(sorted(set(z.get('confidence_score',pd.Series(dtype=object)).dropna().astype(str)))),
            'target_accessions':'|'.join(acc),
            'schema_generation':'|'.join(sorted(z.get('schema_generation',pd.Series(dtype=object)).dropna().astype(str).unique())),
            'candidate_assays':candidate_assays,
            'excluded_assays':excluded_assays,
            'exclusion_fraction':fraction,
            'exclusion_reason':exclusion_reason,
            'threshold_applicable':bool(threshold_applicable),
        })
    return pd.DataFrame(rows)


def _pre15_multitarget_assay_audit_and_filter(
    df: pd.DataFrame,
    release_label: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude historical assay2target one-to-many mappings from single-protein evidence.

    Before ChEMBL 15, assay2target legitimately allowed one assay to map to multiple targets.
    Such assays are not a curation anomaly and therefore are NOT governed by the post-15 0.1%
    multicomponent threshold. They are deterministically excluded from all frozen single-protein
    evidence tracks and reported separately. Within-release identity is the native assay_id; a
    cross-release assay fingerprint must never be used to detect this ambiguity.
    """
    cols=['release_label','native_assay_id','assay_chembl_id','assay_type','n_target_accessions',
          'n_D_targets','n_H_targets','relationship_types','confidence_values','target_accessions',
          'schema_generation','candidate_assays','excluded_assays','exclusion_fraction',
          'exclusion_reason','threshold_applicable']
    if df.empty:
        return df,pd.DataFrame(columns=cols)
    if 'native_assay_id' not in df.columns:
        raise HistoricalSchemaSemanticsError('pre-15 assay2target ambiguity audit requires native_assay_id')
    valid=df.dropna(subset=['native_assay_id','target_accession']).copy()
    candidate_assays=int(valid['native_assay_id'].nunique())
    n=valid.groupby('native_assay_id')['target_accession'].nunique()
    bad=n[n > 1]
    excluded_assays=int(len(bad))
    fraction=excluded_assays/candidate_assays if candidate_assays else 0.0
    audit=_assay_target_mapping_audit_rows(
        valid,'native_assay_id',release_label,candidate_assays,excluded_assays,fraction,
        exclusion_reason='historical_assay2target_multitarget',threshold_applicable=False)
    if excluded_assays:
        df=df.loc[~df['native_assay_id'].isin(set(bad.index))].copy()
    return df,audit


def _multicomponent_audit_and_filter(
    df: pd.DataFrame,
    release_label: str | None,
    max_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Post-ChEMBL15 SINGLE PROTEIN multi-component anomaly gate.

    ChEMBL15+ links each assay to one target. If a target labelled SINGLE PROTEIN resolves to
    multiple UniProt accessions, treat that as a curation/representation anomaly: exclude it,
    audit it, and fail closed above the preregistered 0.1% release threshold. Detection uses the
    within-release native assay_id when available, never the longitudinal assay identity key.
    """
    cols=['release_label','native_assay_id','assay_chembl_id','assay_type','n_target_accessions',
          'n_D_targets','n_H_targets','relationship_types','confidence_values','target_accessions',
          'schema_generation','candidate_assays','excluded_assays','exclusion_fraction',
          'exclusion_reason','threshold_applicable']
    if df.empty:
        return df,pd.DataFrame(columns=cols)
    if 'native_assay_id' in df.columns:
        id_col='native_assay_id'
    elif 'assay_chembl_id' in df.columns:
        id_col='assay_chembl_id'
    else:
        raise HistoricalSchemaSemanticsError('post-15 multicomponent audit requires native_assay_id or assay_chembl_id')
    valid=df.dropna(subset=[id_col,'target_accession']).copy()
    candidate_assays=int(valid[id_col].nunique())
    n=valid.groupby(id_col)['target_accession'].nunique()
    bad=n[n > 1]
    excluded_assays=int(len(bad))
    fraction=excluded_assays/candidate_assays if candidate_assays else 0.0
    audit=_assay_target_mapping_audit_rows(
        valid,id_col,release_label,candidate_assays,excluded_assays,fraction,
        exclusion_reason='post15_single_protein_multicomponent',threshold_applicable=True)
    if fraction > max_fraction:
        raise MultiComponentThresholdError(
            'Post-15 SINGLE PROTEIN target produced assays with multiple target accessions; '
            'exclusion fraction exceeds preregistered threshold: '
            f'{excluded_assays}/{candidate_assays}={fraction:.6f} > {max_fraction:.6f}',audit)
    if excluded_assays:
        df=df.loc[~df[id_col].isin(set(bad.index))].copy()
    return df,audit


def _document_identity(df: pd.DataFrame) -> pd.Series:
    def one(r):
        chem=_norm_text(r.get('document_chembl_id'))
        if chem:
            return 'CHEMBL_DOC:'+chem
        doi=_norm_text(r.get('document_doi'))
        if doi:
            return 'DOI:'+doi
        pubmed=_norm_text(r.get('document_pubmed_id'))
        if pubmed:
            return 'PMID:'+pubmed
        parts=[_norm_text(r.get(c)) for c in ['document_journal','document_year','document_volume','document_issue','document_first_page','document_title']]
        return 'BIB:'+_sha(parts)
    return df.apply(one,axis=1)


def _assay_identity(df: pd.DataFrame) -> pd.Series:
    """Create a stable assay identity when early releases lack assay ChEMBL IDs.

    ChEMBL assay IDs were introduced later than the first database releases. When absent, the
    fallback is a deterministic bibliographic/content fingerprint. The native internal assay_id is
    retained for audit but is not trusted as a cross-release identity by itself.
    """
    out=[]
    for _,r in df.iterrows():
        chem=_norm_text(r.get('assay_chembl_id'))
        if chem:
            out.append('CHEMBL_ASSAY:'+chem); continue
        parts=[
            _norm_text(r.get('document_identity_key')),
            _norm_text(r.get('assay_description')),
            _norm_text(r.get('assay_type')),
            _norm_text(r.get('target_accession')),
        ]
        out.append('LEGACY_ASSAY_FP:'+_sha(parts))
    return pd.Series(out,index=df.index)


def _extract_post15(conn, endpoints, extraction_min_confidence, organism, target_type) -> pd.DataFrame:
    require_tables(conn, POST15_REQUIRED)
    native_target_type = _native_target_type_for_generation('post15', target_type)
    a_assay=pick_column(conn,'activities',['assay_id']); a_mol=pick_column(conn,'activities',['molregno'])
    a_type=pick_column(conn,'activities',['standard_type']); a_rel=pick_column(conn,'activities',['standard_relation','relation'])
    a_val=pick_column(conn,'activities',['standard_value']); a_unit=pick_column(conn,'activities',['standard_units'])
    a_valid=pick_column(conn,'activities',['data_validity_comment'],required=False); a_dup=pick_column(conn,'activities',['potential_duplicate'],required=False)
    assay_id=pick_column(conn,'assays',['assay_id']); assay_chembl=pick_column(conn,'assays',['chembl_id','assay_chembl_id'])
    assay_tid=pick_column(conn,'assays',['tid']); assay_doc=pick_column(conn,'assays',['doc_id']); assay_conf=pick_column(conn,'assays',['confidence_score'])
    assay_type_col=pick_column(conn,'assays',['assay_type']); assay_desc=pick_column(conn,'assays',['description'],required=False)
    td_tid=pick_column(conn,'target_dictionary',['tid']); td_type=pick_column(conn,'target_dictionary',['target_type']); td_org=pick_column(conn,'target_dictionary',['organism'])
    td_chembl=pick_column(conn,'target_dictionary',['chembl_id'],required=False)
    tc_tid=pick_column(conn,'target_components',['tid']); tc_comp=pick_column(conn,'target_components',['component_id'])
    cs_comp=pick_column(conn,'component_sequences',['component_id']); cs_acc=pick_column(conn,'component_sequences',['accession']); cs_seq=pick_column(conn,'component_sequences',['sequence'],required=False)
    return _run_post9_query(conn=conn,endpoints=endpoints,extraction_min_confidence=extraction_min_confidence,organism=organism,target_type=native_target_type,
        a_assay=a_assay,a_mol=a_mol,a_type=a_type,a_rel=a_rel,a_val=a_val,a_unit=a_unit,a_valid=a_valid,a_dup=a_dup,
        assay_id=assay_id,assay_chembl=assay_chembl,assay_doc=assay_doc,assay_type_col=assay_type_col,assay_desc=assay_desc,
        target_join=f'''JOIN target_dictionary td ON aa."{assay_tid}"=td."{td_tid}"
                        JOIN target_components tc ON td."{td_tid}"=tc."{tc_tid}"
                        JOIN component_sequences seq ON tc."{tc_comp}"=seq."{cs_comp}"''',
        confidence_expr=f'aa."{assay_conf}"',relationship_expr='NULL',target_accession_expr=f'seq."{cs_acc}"',
        target_sequence_expr=_col_expr('seq',cs_seq,'target_sequence'),td_type=td_type,td_org=td_org,td_chembl=td_chembl,schema_generation='post15')


def _extract_legacy(conn, endpoints, extraction_min_confidence, organism, target_type) -> pd.DataFrame:
    require_tables(conn, PRE15_REQUIRED)
    native_target_type = _native_target_type_for_generation('pre15_legacy', target_type)
    a_assay=pick_column(conn,'activities',['assay_id']); a_mol=pick_column(conn,'activities',['molregno'])
    a_type=pick_column(conn,'activities',['standard_type','activity_type']); a_rel=pick_column(conn,'activities',['standard_relation','relation'])
    a_val=pick_column(conn,'activities',['standard_value']); a_unit=pick_column(conn,'activities',['standard_units'])
    a_valid=pick_column(conn,'activities',['data_validity_comment'],required=False); a_dup=pick_column(conn,'activities',['potential_duplicate'],required=False)
    assay_id=pick_column(conn,'assays',['assay_id']); assay_chembl=pick_column(conn,'assays',['chembl_id','assay_chembl_id'],required=False)
    assay_doc=pick_column(conn,'assays',['doc_id']); assay_type_col=pick_column(conn,'assays',['assay_type']); assay_desc=pick_column(conn,'assays',['description'],required=False)
    a2t_assay=pick_column(conn,'assay2target',['assay_id']); a2t_tid=pick_column(conn,'assay2target',['tid'])
    confmeta=_legacy_confidence_descriptor(conn,'pre15_legacy'); a2t_conf=confmeta['confidence_column']
    a2t_rel=pick_column(conn,'assay2target',['relationship_type'],required=False)
    td_tid=pick_column(conn,'target_dictionary',['tid']); td_type=pick_column(conn,'target_dictionary',['target_type']); td_org=pick_column(conn,'target_dictionary',['organism'])
    td_chembl=pick_column(conn,'target_dictionary',['chembl_id'],required=False); td_acc=pick_column(conn,'target_dictionary',['protein_accession']); td_seq=pick_column(conn,'target_dictionary',['protein_sequence'],required=False)
    return _run_post9_query(conn=conn,endpoints=endpoints,extraction_min_confidence=extraction_min_confidence,organism=organism,target_type=native_target_type,
        a_assay=a_assay,a_mol=a_mol,a_type=a_type,a_rel=a_rel,a_val=a_val,a_unit=a_unit,a_valid=a_valid,a_dup=a_dup,
        assay_id=assay_id,assay_chembl=assay_chembl,assay_doc=assay_doc,assay_type_col=assay_type_col,assay_desc=assay_desc,
        target_join=f'''JOIN assay2target a2t ON aa."{assay_id}"=a2t."{a2t_assay}"
                        JOIN target_dictionary td ON a2t."{a2t_tid}"=td."{td_tid}"''',
        confidence_expr=f'a2t."{a2t_conf}"',relationship_expr=(f'a2t."{a2t_rel}"' if a2t_rel else 'NULL'),target_accession_expr=f'td."{td_acc}"',
        target_sequence_expr=_col_expr('td',td_seq,'target_sequence'),td_type=td_type,td_org=td_org,td_chembl=td_chembl,schema_generation='pre15_legacy')


def _doc_columns(conn):
    d_doc=pick_column(conn,'docs',['doc_id'])
    return dict(
        d_doc=d_doc,
        d_chembl=pick_column(conn,'docs',['chembl_id'],required=False),
        d_year=pick_column(conn,'docs',['year'],required=False),
        d_title=pick_column(conn,'docs',['title'],required=False),
        d_doi=pick_column(conn,'docs',['doi'],required=False),
        d_pubmed=pick_column(conn,'docs',['pubmed_id'],required=False),
        d_journal=pick_column(conn,'docs',['journal'],required=False),
        d_volume=pick_column(conn,'docs',['volume'],required=False),
        d_issue=pick_column(conn,'docs',['issue'],required=False),
        d_first=pick_column(conn,'docs',['first_page'],required=False),
    )


def _doc_select(d):
    return ',\n      '.join([
        _col_expr('doc',d['d_doc'],'native_document_id'),
        _col_expr('doc',d['d_chembl'],'document_chembl_id'), _col_expr('doc',d['d_year'],'document_year'),
        _col_expr('doc',d['d_title'],'document_title'), _col_expr('doc',d['d_doi'],'document_doi'),
        _col_expr('doc',d['d_pubmed'],'document_pubmed_id'), _col_expr('doc',d['d_journal'],'document_journal'),
        _col_expr('doc',d['d_volume'],'document_volume'), _col_expr('doc',d['d_issue'],'document_issue'),
        _col_expr('doc',d['d_first'],'document_first_page'),
    ])


def _run_post9_query(*,conn,endpoints,extraction_min_confidence,organism,target_type,a_assay,a_mol,a_type,a_rel,a_val,a_unit,a_valid,a_dup,
                     assay_id,assay_chembl,assay_doc,assay_type_col,assay_desc,target_join,confidence_expr,relationship_expr,target_accession_expr,
                     target_sequence_expr,td_type,td_org,td_chembl,schema_generation):
    cs_smiles=pick_column(conn,'compound_structures',['canonical_smiles']); cs_mol=pick_column(conn,'compound_structures',['molregno'])
    cs_molfile=pick_column(conn,'compound_structures',['molfile'],required=False)
    md_mol=pick_column(conn,'molecule_dictionary',['molregno']); md_chembl=pick_column(conn,'molecule_dictionary',['chembl_id']); md_pref=pick_column(conn,'molecule_dictionary',['pref_name'],required=False)
    d=_doc_columns(conn)
    placeholders=','.join('?' for _ in endpoints)
    assay_chem_expr=_col_expr('aa',assay_chembl,'assay_chembl_id')
    assay_src_id=pick_column(conn,'assays',['src_id'],required=False)
    assay_src_assay=pick_column(conn,'assays',['src_assay_id','source_assay_id'],required=False)
    sql=f'''SELECT aa."{assay_id}" AS native_assay_id,
      {assay_chem_expr}, {_col_expr('aa',assay_desc,'assay_description')},
      {_col_expr('aa',assay_src_id,'assay_source_id')}, {_col_expr('aa',assay_src_assay,'assay_source_assay_id')},
      md."{md_chembl}" AS molecule_chembl_id, {_col_expr('md',md_pref,'molecule_pref_name')},
      cs."{cs_smiles}" AS canonical_smiles, {_col_expr('cs',cs_molfile,'molfile')},
      {_col_expr('td',td_chembl,'target_chembl_id')}, {target_accession_expr} AS target_accession,
      {target_sequence_expr}, td."{td_type}" AS target_type, td."{td_org}" AS organism,
      {confidence_expr} AS confidence_score, {relationship_expr} AS target_relationship_type,
      aa."{assay_type_col}" AS assay_type, act."{a_type}" AS standard_type, act."{a_rel}" AS standard_relation,
      act."{a_val}" AS standard_value, act."{a_unit}" AS standard_units,
      {_col_expr('act',a_valid,'data_validity_comment')}, {_col_expr('act',a_dup,'potential_duplicate')},
      {_doc_select(d)}
    FROM activities act JOIN assays aa ON act."{a_assay}"=aa."{assay_id}"
    {target_join}
    JOIN compound_structures cs ON act."{a_mol}"=cs."{cs_mol}"
    JOIN molecule_dictionary md ON act."{a_mol}"=md."{md_mol}"
    LEFT JOIN docs doc ON aa."{assay_doc}"=doc."{d['d_doc']}"
    WHERE act."{a_type}" IN ({placeholders}) AND {confidence_expr} >= ?
      AND td."{td_type}"=? AND td."{td_org}"=? AND {target_accession_expr} IS NOT NULL'''
    df=pd.read_sql_query(sql,conn,params=[*endpoints,extraction_min_confidence,target_type,organism])
    df['schema_generation']=schema_generation
    df['target_identity_source']='native_accession'
    df['requested_scientific_target_type']='SINGLE PROTEIN'
    df['historical_native_target_type_equivalent']=target_type
    return df


def _extract_pre9(conn,endpoints,extraction_min_confidence,organism,target_type,target_bridge: pd.DataFrame | None) -> pd.DataFrame:
    """Extract v1-v8 without pretending the later molecule/target schema already existed."""
    require_tables(conn,PRE9_REQUIRED)
    native_target_type = _native_target_type_for_generation('pre9_early', target_type)
    act_cols=columns(conn,'activities'); assay_cols=columns(conn,'assays'); a2t_cols=columns(conn,'assay2target'); td_cols=columns(conn,'target_dictionary'); comp_cols=columns(conn,'compounds')
    a_assay=pick_column(conn,'activities',['assay_id']); a_mol=pick_column(conn,'activities',['molregno']); a_type=pick_column(conn,'activities',['standard_type','activity_type'])
    a_rel=pick_column(conn,'activities',['standard_relation','relation']); a_val=pick_column(conn,'activities',['standard_value']); a_unit=pick_column(conn,'activities',['standard_units'])
    a_valid=pick_column(conn,'activities',['data_validity_comment'],required=False); a_dup=pick_column(conn,'activities',['potential_duplicate'],required=False)
    assay_id=pick_column(conn,'assays',['assay_id']); assay_chembl=pick_column(conn,'assays',['chembl_id','assay_chembl_id'],required=False); assay_doc=pick_column(conn,'assays',['doc_id'])
    assay_type_col=pick_column(conn,'assays',['assay_type']); assay_desc=pick_column(conn,'assays',['description'],required=False)
    assay_src_id=pick_column(conn,'assays',['src_id'],required=False); assay_src_assay=pick_column(conn,'assays',['src_assay_id','source_assay_id'],required=False)
    a2t_assay=pick_column(conn,'assay2target',['assay_id']); a2t_tid=pick_column(conn,'assay2target',['tid']); confmeta=_legacy_confidence_descriptor(conn,'pre9_early'); a2t_conf=confmeta['confidence_column']
    a2t_rel=pick_column(conn,'assay2target',['relationship_type'],required=False)
    td_tid=pick_column(conn,'target_dictionary',['tid']); td_org=pick_column(conn,'target_dictionary',['organism']); td_type=pick_column(conn,'target_dictionary',['target_type'],required=False)
    td_acc=pick_column(conn,'target_dictionary',['protein_accession'],required=False); td_seq=pick_column(conn,'target_dictionary',['protein_sequence'],required=False); td_chembl=pick_column(conn,'target_dictionary',['chembl_id'],required=False)
    c_mol=pick_column(conn,'compounds',['molregno']); c_chembl=pick_column(conn,'compounds',['chembl_id'],required=False); c_pref=pick_column(conn,'compounds',['pref_name','name'],required=False)
    c_smiles=pick_column(conn,'compounds',['canonical_smiles'],required=False); c_molfile=pick_column(conn,'compounds',['molfile'],required=False)
    if not c_smiles and not c_molfile:
        raise HistoricalSchemaSemanticsError('pre9_early: compounds exposes neither canonical_smiles nor molfile')
    if not td_acc and not td_seq:
        raise HistoricalSchemaSemanticsError('pre9_early: target_dictionary exposes neither protein_accession nor protein_sequence')
    d=_doc_columns(conn); placeholders=','.join('?' for _ in endpoints)
    # The earliest releases used a historical 0-7 confidence field; later pre-9 releases may
    # already expose confidence_score on the modern 0-9 scale. Apply the frozen evidence floor
    # according to the release's actual confidence regime, not the schema-family label alone.
    if confmeta['confidence_regime']=='early_0_7_high_confidence':
        confidence_predicate=f'a2t."{a2t_conf}"=?'
        confidence_param=confmeta['direct_primary_value']  # historical high-confidence 7
    else:
        confidence_predicate=f'a2t."{a2t_conf}">=?'
        confidence_param=extraction_min_confidence          # modern 0-9 scale in legacy location

    sql=f'''SELECT aa."{assay_id}" AS native_assay_id, {_col_expr('aa',assay_chembl,'assay_chembl_id')}, {_col_expr('aa',assay_desc,'assay_description')},
      {_col_expr('aa',assay_src_id,'assay_source_id')}, {_col_expr('aa',assay_src_assay,'assay_source_assay_id')},
      {_col_expr('cmp',c_chembl,'molecule_chembl_id')}, {_col_expr('cmp',c_pref,'molecule_pref_name')}, {_col_expr('cmp',c_smiles,'canonical_smiles')}, {_col_expr('cmp',c_molfile,'molfile')},
      {_col_expr('td',td_chembl,'target_chembl_id')}, {_col_expr('td',td_acc,'target_accession')}, {_col_expr('td',td_seq,'target_sequence')}, {_col_expr('td',td_type,'target_type')}, td."{td_org}" AS organism,
      a2t."{a2t_conf}" AS confidence_score, {(f'a2t."{a2t_rel}"' if a2t_rel else 'NULL')} AS target_relationship_type,
      aa."{assay_type_col}" AS assay_type, act."{a_type}" AS standard_type, act."{a_rel}" AS standard_relation,
      act."{a_val}" AS standard_value, act."{a_unit}" AS standard_units, {_col_expr('act',a_valid,'data_validity_comment')}, {_col_expr('act',a_dup,'potential_duplicate')},
      {_doc_select(d)}
    FROM activities act JOIN assays aa ON act."{a_assay}"=aa."{assay_id}"
    JOIN assay2target a2t ON aa."{assay_id}"=a2t."{a2t_assay}"
    JOIN target_dictionary td ON a2t."{a2t_tid}"=td."{td_tid}"
    JOIN compounds cmp ON act."{a_mol}"=cmp."{c_mol}"
    LEFT JOIN docs doc ON aa."{assay_doc}"=doc."{d['d_doc']}"
    WHERE act."{a_type}" IN ({placeholders}) AND td."{td_org}"=? AND {confidence_predicate}'''
    df=pd.read_sql_query(sql,conn,params=[*endpoints,organism,confidence_param])
    df['schema_generation']='pre9_early'; df['early_confidence_regime']=confmeta['confidence_regime']; df['early_direct_primary_value']=confmeta['direct_primary_value']

    # Early releases may lack either UniProt accession, target_type, or both. A fixed historical
    # reference bridge (CHEMBL09 in the locked protocol) may certify identity/type from exact
    # sequence + organism metadata only. It never supplies activity evidence.
    needs_bridge = (not td_acc) or (not td_type)
    if needs_bridge:
        if target_bridge is None or target_bridge.empty:
            missing=[]
            if not td_acc: missing.append('protein_accession')
            if not td_type: missing.append('target_type')
            raise HistoricalSchemaSemanticsError(
                'pre9_early requires a non-empty exact-sequence target bridge because historical '
                f'target metadata are missing: {missing}'
            )
        df['sequence_normalized']=df['target_sequence'].map(_normalize_sequence)
        df['organism_normalized']=df['organism'].map(_norm_text)
        bridge=target_bridge[target_bridge.bridge_status.eq('unique')].copy()
        bridge=bridge[['sequence_normalized','organism_normalized','target_accession','target_type_bridge','bridge_status']].rename(
            columns={'target_accession':'bridge_target_accession'}
        )
        df=df.merge(bridge,on=['sequence_normalized','organism_normalized'],how='left')
        if td_acc:
            native=df['target_accession'].fillna('').astype(str)
            bridge_acc=df['bridge_target_accession'].fillna('').astype(str)
            consistent=native.ne('') & bridge_acc.ne('') & native.eq(bridge_acc)
            df['target_identity_source']=consistent.map({True:'native_accession_bridge_confirmed',False:'unmapped_or_inconsistent_bridge'})
            df.loc[~consistent,'target_accession']=None
        else:
            df['target_accession']=df['bridge_target_accession']
            df['target_identity_source']=df['target_accession'].notna().map({True:'exact_sequence_bridge',False:'unmapped_sequence'})
        if not td_type:
            df['target_type']=df['target_type_bridge']
    else:
        df['target_identity_source']='native_accession'

    # Never silently widen to non-protein targets after native/bridge certification.
    # For ChEMBL 1-14, the historical native label corresponding to the frozen
    # scientific single-protein concept is PROTEIN; the explicit SINGLE PROTEIN
    # controlled vocabulary is a ChEMBL-15+ data-model feature.
    if 'target_type' in df.columns:
        df=df[df['target_type'].fillna('').astype(str).str.upper().eq(native_target_type)].copy()
    df=df[df.target_accession.notna()].copy()
    df['requested_scientific_target_type']='SINGLE PROTEIN'
    df['historical_native_target_type_equivalent']=native_target_type
    return df


def _postprocess(df: pd.DataFrame, generation: str, extraction_min_confidence: int, primary_confidence: int,
                 primary_assay_type: str, release_label: str | None, *,
                 potential_duplicate_schema_available: bool, data_validity_schema_available: bool,
                 structure_cache_path=None, standardization_progress: bool=False):
    if df.empty:
        # Preserve a schema-complete empty frame so zero-row historical releases remain explicit
        # processed archives rather than crashing or disappearing from the temporal audit.
        empty_cols = {
            'native_document_id':'object','document_identity_key':'object','assay_identity_key':'object','standard_value_molar':'float64',
            'pactivity':'float64','unit_convertible':'bool','relation_eligible':'bool',
            'potential_duplicate_field_available':'bool','data_validity_field_available':'bool',
            'potential_duplicate_flagged':'bool','quality_eligible':'bool','canonical_isomeric_smiles':'object',
            'full_inchikey':'object','murcko_scaffold_smiles':'object','std_status':'object','std_error':'object',
            'std_input_format':'object','structure_eligible':'bool','legacy_mapping_consistent':'bool',
            'historical_confidence_equivalence':'object','primary_mapping_certified':'bool',
            'primary_record_eligible':'bool','confidence8_B_record_eligible':'bool',
            'historical_high_confidence_B_sensitivity_eligible':'bool','relaxed_confidence_B_record_eligible':'bool',
            'confidence9_all_assay_eligible':'bool','primary_direct_all_assay_eligible':'bool',
            'broad_prior_knowledge_eligible':'bool','release_label':'object',
            'requested_scientific_target_type':'object','historical_native_target_type_equivalent':'object'
        }
        for col,dtype in empty_cols.items():
            if col not in df.columns:
                df[col]=pd.Series(index=df.index,dtype=dtype)
        df['potential_duplicate_field_available']=bool(potential_duplicate_schema_available)
        df['data_validity_field_available']=bool(data_validity_schema_available)
        df['release_label']=release_label
        df.attrs['standardization_stats']={
            'input_rows':0,'unique_smiles_inputs':0,'unique_molblock_fallback_inputs':0,
            'persistent_cache_hits':0,'standardized_new':0,'standardization_seconds':0.0,
        }
        return df
    df['document_identity_key']=_document_identity(df)
    df['assay_identity_key']=_assay_identity(df)
    df['standard_relation']=df['standard_relation'].fillna('').astype(str).str.strip()
    df['relation_eligible']=df['standard_relation'].isin(['=','<','>'])
    df['standard_value_molar']=[to_molar(v,u) for v,u in zip(df.standard_value,df.standard_units)]
    df['unit_convertible']=df.standard_value_molar.notna(); df['pactivity']=df.standard_value_molar.map(pactivity_from_molar)
    # Schema availability is a release property, not inferred from whether a selected row happens to be non-null.
    df['potential_duplicate_field_available']=bool(potential_duplicate_schema_available)
    df['data_validity_field_available']=bool(data_validity_schema_available)
    duplicate_flag=_potential_duplicate_flag(df['potential_duplicate']); validity_ok=_validity_eligible(df['data_validity_comment'])
    df['potential_duplicate_flagged']=duplicate_flag; df['quality_eligible']=validity_ok & ~duplicate_flag

    smiles_values=df.get('canonical_smiles',pd.Series([None]*len(df),index=df.index)).tolist()
    molblock_values=df.get('molfile',pd.Series([None]*len(df),index=df.index)).tolist()
    std,std_stats=standardize_structure_batch(
        smiles_values,molblock_values,cache_db_path=structure_cache_path,
        progress_label=(release_label if standardization_progress else None),
    )
    df.attrs['standardization_stats']=std_stats
    sdf=pd.DataFrame(std,index=df.index)
    for c in ['canonical_isomeric_smiles','full_inchikey','murcko_scaffold_smiles','status','error','input_format']:
        out='std_'+c if c in {'status','error','input_format'} else c
        df[out]=sdf[c].values
    df['structure_eligible']=df.std_status.eq('ok')

    conf=pd.to_numeric(df.confidence_score,errors='coerce'); assay_type=df.assay_type.fillna('').astype(str).str.strip().str.upper(); reltype=df.target_relationship_type.fillna('').astype(str).str.strip().str.upper()
    base=df.quality_eligible & df.structure_eligible & df.relation_eligible & df.unit_convertible

    if generation == 'pre9_early':
        relationship_available=reltype.ne('').any()
        regime=str(df.get('early_confidence_regime',pd.Series('',index=df.index)).dropna().astype(str).iloc[0]) if len(df) else ''
        if regime=='early_0_7_high_confidence':
            # Historical 0-7 scale: high-confidence 7 is not numerically relabelled as modern 9.
            high=conf.eq(7)
            direct=reltype.eq('D') if relationship_available else pd.Series(False,index=df.index)
            primary_mapping=high & direct
            sensitivity_mapping=high & (reltype.isin({'D','H'}) if relationship_available else True)
            df['legacy_mapping_consistent']=primary_mapping | sensitivity_mapping
            df['historical_confidence_equivalence']='early7+D_required_for_primary' if relationship_available else 'uncertified_no_relationship_type'
            df['primary_mapping_certified']=primary_mapping
            df['primary_record_eligible']=base & primary_mapping & assay_type.eq(primary_assay_type)
            df['confidence8_B_record_eligible']=False
            df['historical_high_confidence_B_sensitivity_eligible']=base & sensitivity_mapping & assay_type.eq(primary_assay_type)
            df['relaxed_confidence_B_record_eligible']=df['historical_high_confidence_B_sensitivity_eligible']
            df['confidence9_all_assay_eligible']=base & primary_mapping  # compatibility alias only; historical score remains 7
            df['primary_direct_all_assay_eligible']=base & primary_mapping
            df['broad_prior_knowledge_eligible']=base & sensitivity_mapping
        elif regime=='modern_0_9_legacy_location':
            # Some pre-9-schema releases already store modern 0-9 confidence_score in assay2target.
            # Use the actual confidence regime rather than hard-coding 7 from the schema generation.
            direct=reltype.eq('D') if relationship_available else pd.Series(False,index=df.index)
            sens=reltype.isin({'D','H'}) if relationship_available else pd.Series(True,index=df.index)
            primary_mapping=conf.eq(primary_confidence) & direct
            sensitivity_mapping=conf.ge(extraction_min_confidence) & sens
            df['legacy_mapping_consistent']=sensitivity_mapping
            df.loc[conf.eq(primary_confidence),'legacy_mapping_consistent']=direct[conf.eq(primary_confidence)]
            df['historical_confidence_equivalence']='modern_0_9_legacy_location' if relationship_available else 'modern_0_9_uncertified_no_relationship_type'
            df['primary_mapping_certified']=primary_mapping
            df['primary_record_eligible']=base & primary_mapping & assay_type.eq(primary_assay_type)
            df['confidence8_B_record_eligible']=base & sensitivity_mapping & assay_type.eq(primary_assay_type)
            df['historical_high_confidence_B_sensitivity_eligible']=False
            df['relaxed_confidence_B_record_eligible']=df['confidence8_B_record_eligible']
            df['confidence9_all_assay_eligible']=base & primary_mapping
            df['primary_direct_all_assay_eligible']=base & primary_mapping
            df['broad_prior_knowledge_eligible']=base & sensitivity_mapping
        else:
            raise HistoricalSchemaSemanticsError(f'pre9_early: unsupported confidence regime {regime!r}')
    else:
        legacy=df.schema_generation.eq('pre15_legacy'); direct=(~legacy)|reltype.eq('D'); sens=(~legacy)|reltype.isin({'D','H'})
        df['legacy_mapping_consistent']=True
        df.loc[legacy & conf.eq(9),'legacy_mapping_consistent']=reltype[legacy & conf.eq(9)].eq('D')
        df.loc[legacy & conf.eq(8),'legacy_mapping_consistent']=reltype[legacy & conf.eq(8)].eq('H')
        df['historical_confidence_equivalence']='modern_0_9'
        df['primary_mapping_certified']=conf.eq(primary_confidence) & direct
        df['primary_record_eligible']=base & conf.eq(primary_confidence) & assay_type.eq(primary_assay_type) & direct
        df['confidence8_B_record_eligible']=base & conf.ge(extraction_min_confidence) & assay_type.eq(primary_assay_type) & sens
        df['historical_high_confidence_B_sensitivity_eligible']=False
        df['relaxed_confidence_B_record_eligible']=df['confidence8_B_record_eligible']
        df['confidence9_all_assay_eligible']=base & conf.eq(primary_confidence) & direct
        df['primary_direct_all_assay_eligible']=df['confidence9_all_assay_eligible']
        df['broad_prior_knowledge_eligible']=base & conf.ge(extraction_min_confidence) & sens

    df['release_label']=release_label
    return df


def extract_release(db_path: Path,endpoints=('IC50','Ki'),extraction_min_confidence=8,organism='Homo sapiens',target_type='SINGLE PROTEIN',
                    primary_confidence=9,primary_assay_type='B',release_label: str|None=None,multicomponent_max_fraction:float=0.001,
                    return_audit:bool=False,target_bridge:pd.DataFrame|None=None,structure_cache_path=None,
                    standardization_progress:bool=False):
    """Extract one historical ChEMBL release under the frozen multi-schema F0 rules."""
    schema_info=inspect_schema(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        generation=detect_schema_generation(conn)
        if generation=='post15':
            df=_extract_post15(conn,endpoints,extraction_min_confidence,organism,target_type)
        elif generation=='pre15_legacy':
            df=_extract_legacy(conn,endpoints,extraction_min_confidence,organism,target_type)
        else:
            df=_extract_pre9(conn,endpoints,extraction_min_confidence,organism,target_type,target_bridge)
    raw_candidate_rows=len(df)

    # Resolve within-release target ambiguity BEFORE expensive structure standardization. Pre-15
    # assay2target one-to-many mappings are a historical data-model feature and are excluded
    # deterministically from the single-protein evidence universe. Post-15 multi-accession
    # SINGLE PROTEIN targets remain governed by the preregistered 0.1% fail-closed threshold.
    pre15_multi=pd.DataFrame()
    multi=pd.DataFrame()
    if generation in {'pre9_early','pre15_legacy'}:
        df,pre15_multi=_pre15_multitarget_assay_audit_and_filter(df,release_label)
    else:
        df,multi=_multicomponent_audit_and_filter(df,release_label,float(multicomponent_max_fraction))
    rows_after_target_ambiguity_filter=len(df)

    if standardization_progress:
        unique_smiles_n=df.loc[df.get('canonical_smiles',pd.Series('',index=df.index)).fillna('').astype(str).str.strip().ne(''),'canonical_smiles'].astype(str).nunique() if 'canonical_smiles' in df else 0
        unique_mol_n=df.loc[df.get('molfile',pd.Series('',index=df.index)).fillna('').astype(str).str.strip().ne(''),'molfile'].astype(str).nunique() if 'molfile' in df else 0
        excluded_pre15=int(pre15_multi['excluded_assays'].iloc[0]) if not pre15_multi.empty else 0
        excluded_post15=int(multi['excluded_assays'].iloc[0]) if not multi.empty else 0
        print(f'{release_label}: candidate rows after historical extraction floor={raw_candidate_rows}; '
              f'rows after target-ambiguity filter={rows_after_target_ambiguity_filter}; '
              f'pre15 multi-target assays excluded={excluded_pre15}; post15 multicomponent assays excluded={excluded_post15}; '
              f'unique raw smiles={unique_smiles_n}; unique raw molblocks={unique_mol_n}')
    df=_postprocess(
        df,generation,extraction_min_confidence,primary_confidence,primary_assay_type,release_label,
        potential_duplicate_schema_available=bool(schema_info['potential_duplicate_available']),
        data_validity_schema_available=bool(schema_info['data_validity_comment_available']),
        structure_cache_path=structure_cache_path,standardization_progress=standardization_progress,
    )
    standardization_stats=dict(df.attrs.get('standardization_stats',{}))
    standardization_stats['release_label']=release_label
    standardization_stats['schema_generation']=generation
    standardization_stats['raw_candidate_rows']=raw_candidate_rows
    standardization_stats['rows_after_target_ambiguity_filter']=rows_after_target_ambiguity_filter
    standardization_stats['pre15_multitarget_assays_excluded']=int(pre15_multi['excluded_assays'].iloc[0]) if not pre15_multi.empty else 0
    standardization_stats['post15_multicomponent_assays_excluded']=int(multi['excluded_assays'].iloc[0]) if not multi.empty else 0
    mapping=(df.groupby(['schema_generation','confidence_score','target_relationship_type','historical_confidence_equivalence'],dropna=False)
             .agg(rows=('assay_identity_key','size'),unique_assays=('assay_identity_key','nunique')).reset_index()) if len(df) else pd.DataFrame()
    if len(mapping): mapping['release_label']=release_label
    bridge_audit=pd.DataFrame()
    if generation=='pre9_early':
        bridge_audit=(df.groupby(['target_identity_source'],dropna=False).agg(rows=('assay_identity_key','size'),unique_targets=('target_accession','nunique')).reset_index())
        bridge_audit['release_label']=release_label
    audit={'schema_generation':generation,'multicomponent_exclusions':multi,
           'pre15_multitarget_assay_exclusions':pre15_multi,
           'legacy_target_mapping':mapping,'pre9_target_bridge':bridge_audit,
           'standardization_stats':pd.DataFrame([standardization_stats])}
    return (df,audit) if return_audit else df


def extract_molecule_name_index(db_path: Path) -> pd.DataFrame:
    """Newest-release name/synonym index for identity resolution only; never historical evidence."""
    with sqlite3.connect(str(db_path)) as conn:
        require_tables(conn,['molecule_dictionary','compound_structures'])
        md_mol=pick_column(conn,'molecule_dictionary',['molregno']); md_chembl=pick_column(conn,'molecule_dictionary',['chembl_id']); md_pref=pick_column(conn,'molecule_dictionary',['pref_name'],required=False)
        cs_mol=pick_column(conn,'compound_structures',['molregno']); cs_smiles=pick_column(conn,'compound_structures',['canonical_smiles']); cs_molfile=pick_column(conn,'compound_structures',['molfile'],required=False)
        parts=[]
        if md_pref:
            q=f'''SELECT md."{md_chembl}" AS molecule_chembl_id, md."{md_pref}" AS name, cs."{cs_smiles}" AS canonical_smiles, {_col_expr('cs',cs_molfile,'molfile')}, 'pref_name' AS name_source
                  FROM molecule_dictionary md JOIN compound_structures cs ON md."{md_mol}"=cs."{cs_mol}" WHERE md."{md_pref}" IS NOT NULL'''
            parts.append(pd.read_sql_query(q,conn))
        if 'molecule_synonyms' in table_names(conn):
            ms_mol=pick_column(conn,'molecule_synonyms',['molregno']); ms_syn=pick_column(conn,'molecule_synonyms',['synonyms','molecule_synonym','synonym'])
            q=f'''SELECT md."{md_chembl}" AS molecule_chembl_id, ms."{ms_syn}" AS name, cs."{cs_smiles}" AS canonical_smiles, {_col_expr('cs',cs_molfile,'molfile')}, 'synonym' AS name_source
                  FROM molecule_synonyms ms JOIN molecule_dictionary md ON ms."{ms_mol}"=md."{md_mol}" JOIN compound_structures cs ON md."{md_mol}"=cs."{cs_mol}" WHERE ms."{ms_syn}" IS NOT NULL'''
            parts.append(pd.read_sql_query(q,conn))
    if not parts:
        return pd.DataFrame(columns=['molecule_chembl_id','name','canonical_smiles','name_source','full_inchikey','canonical_isomeric_smiles'])
    raw=pd.concat(parts,ignore_index=True); std=[standardize_structure(s,m) for s,m in zip(raw.canonical_smiles,raw.molfile)]
    sdf=pd.DataFrame(std); raw['full_inchikey']=sdf.full_inchikey.values; raw['canonical_isomeric_smiles']=sdf.canonical_isomeric_smiles.values; raw['std_status']=sdf.status.values
    return raw[raw.std_status.eq('ok')].drop_duplicates()


def audit_pre9_target_bridge(db_path: Path, target_bridge: pd.DataFrame, organism: str='Homo sapiens') -> pd.DataFrame:
    """Audit exact sequence+organism target identity resolution for a pre-v9 archive.

    This uses target metadata only. It does not inspect activities and therefore cannot alter
    historical evidence counts. Ambiguous reference sequences are never auto-resolved.
    """
    with sqlite3.connect(str(db_path)) as conn:
        if detect_schema_generation(conn) != 'pre9_early':
            raise ValueError('audit_pre9_target_bridge requires a pre9_early database')
        td=columns(conn,'target_dictionary')
        seq=pick_column(conn,'target_dictionary',['protein_sequence'])
        org=pick_column(conn,'target_dictionary',['organism'])
        tid=pick_column(conn,'target_dictionary',['tid'])
        acc=pick_column(conn,'target_dictionary',['protein_accession'],required=False)
        typ=pick_column(conn,'target_dictionary',['target_type'],required=False)
        q=f'''SELECT td."{tid}" AS historical_tid, td."{org}" AS organism,
                    td."{seq}" AS target_sequence, {_col_expr('td',acc,'native_target_accession')},
                    {_col_expr('td',typ,'native_target_type')}
             FROM target_dictionary td WHERE td."{org}"=? AND td."{seq}" IS NOT NULL'''
        early=pd.read_sql_query(q,conn,params=[organism])
    early['sequence_normalized']=early['target_sequence'].map(_normalize_sequence)
    early['organism_normalized']=early['organism'].map(_norm_text)
    bridge=target_bridge.copy()
    cols=['sequence_normalized','organism_normalized','bridge_status','target_accession','target_type_bridge','n_accessions','target_accessions']
    cols=[c for c in cols if c in bridge.columns]
    m=early.merge(bridge[cols],on=['sequence_normalized','organism_normalized'],how='left')
    m['bridge_status']=m['bridge_status'].fillna('unmapped')
    native=m['native_target_accession'].fillna('').astype(str)
    bridged=m.get('target_accession',pd.Series('',index=m.index)).fillna('').astype(str)
    m['native_bridge_accession_consistent']=(native.eq('') | bridged.eq('') | native.eq(bridged))
    return m
