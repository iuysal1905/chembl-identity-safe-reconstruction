import sqlite3
from pathlib import Path
import pytest
import pandas as pd
from rdkit import Chem

from ad_f0.chembl_adapter import (
    build_target_sequence_bridge,
    detect_schema_generation,
    extract_release,
    schema_probe_row,
    inspect_schema,
)


def _molblock(smiles='CCO'):
    return Chem.MolToMolBlock(Chem.MolFromSmiles(smiles))


def make_reference_pre15(path: Path):
    con=sqlite3.connect(path)
    con.executescript('''
    CREATE TABLE activities (assay_id INTEGER, molregno INTEGER, standard_type TEXT, relation TEXT, standard_value REAL, standard_units TEXT);
    CREATE TABLE assays (assay_id INTEGER, assay_type TEXT, description TEXT, doc_id INTEGER, chembl_id TEXT);
    CREATE TABLE assay2target (assay_id INTEGER, tid INTEGER, relationship_type TEXT, confidence_score INTEGER);
    CREATE TABLE target_dictionary (tid INTEGER, target_type TEXT, organism TEXT, protein_sequence TEXT, protein_accession TEXT, chembl_id TEXT);
    CREATE TABLE compound_structures (molregno INTEGER, canonical_smiles TEXT);
    CREATE TABLE molecule_dictionary (molregno INTEGER, pref_name TEXT, chembl_id TEXT);
    CREATE TABLE docs (doc_id INTEGER, year INTEGER, chembl_id TEXT, title TEXT, doi TEXT);
    ''')
    con.execute("INSERT INTO target_dictionary VALUES (51,'PROTEIN','Homo sapiens','MPEPTIDE','P56817','CHEMBL_T')")
    con.commit(); con.close()


def make_pre9(path: Path, *, relationship_column=True, relationship='D', confidence=7, accession=False, two_targets_same_sequence=False):
    con=sqlite3.connect(path)
    rel_col=', relationship_type TEXT' if relationship_column else ''
    con.executescript(f'''
    CREATE TABLE activities (
      activity_id INTEGER, assay_id INTEGER, doc_id INTEGER, record_id INTEGER, molregno INTEGER,
      relation TEXT, published_value REAL, published_units TEXT, standard_value REAL,
      standard_units TEXT, standard_flag INTEGER, activity_type TEXT, activity_comment TEXT
    );
    CREATE TABLE assays (assay_id INTEGER, assay_type TEXT, description TEXT, doc_id INTEGER, src_id INTEGER, src_assay_id TEXT);
    CREATE TABLE assay2target (assay_id INTEGER, tid INTEGER, confidence INTEGER{rel_col});
    CREATE TABLE target_dictionary (
      tid INTEGER, pref_name TEXT, organism TEXT, protein_sequence TEXT
      {', protein_accession TEXT' if accession else ''}
    );
    CREATE TABLE compounds (molregno INTEGER, molfile TEXT, pref_name TEXT);
    CREATE TABLE docs (doc_id INTEGER, year INTEGER, title TEXT, doi TEXT, pubmed_id TEXT, journal TEXT, volume TEXT, issue TEXT, first_page TEXT);
    ''')
    con.execute("INSERT INTO activities VALUES (1,11,21,31,41,'=',1000,'nM',1000,'nM',1,'IC50',NULL)")
    con.execute("INSERT INTO assays VALUES (11,'B','early direct assay',21,1,'SRC-A')")
    if relationship_column:
        con.execute("INSERT INTO assay2target VALUES (11,51,?,?)",(confidence,relationship))
    else:
        con.execute("INSERT INTO assay2target VALUES (11,51,?)",(confidence,))
    if accession:
        con.execute("INSERT INTO target_dictionary VALUES (51,'BACE1','Homo sapiens','MPEPTIDE','P56817')")
    else:
        con.execute("INSERT INTO target_dictionary VALUES (51,'BACE1','Homo sapiens','MPEPTIDE')")
    if two_targets_same_sequence:
        # Used only by separate reference-bridge ambiguity tests if needed.
        pass
    con.execute("INSERT INTO compounds VALUES (41,?,'ETHANOL')",(_molblock(),))
    con.execute("INSERT INTO docs VALUES (21,2008,'Early paper','10.1/early','123','J Med Chem','51','2','10')")
    con.commit(); con.close()


def test_pre9_exact_sequence_bridge_and_primary_mapping(tmp_path: Path):
    ref=tmp_path/'chembl09.db'; early=tmp_path/'chembl02.db'
    make_reference_pre15(ref); make_pre9(early,relationship_column=True,relationship='D',confidence=7,accession=False)
    bridge=build_target_sequence_bridge(ref)
    assert detect_schema_generation(sqlite3.connect(early))=='pre9_early'
    probe=schema_probe_row(early,'CHEMBL02')
    assert probe['schema_structural_probe_pass']
    assert probe['primary_semantics_probe_pass']
    assert probe['linked_document_count']==1
    assert probe['document_year_null_rate']==0.0
    assert probe['document_title_null_rate']==0.0
    assert probe['document_first_page_null_rate']==0.0
    assert probe['document_weak_alias_bridgeable_rate']==1.0
    d=extract_release(early,release_label='CHEMBL02',target_bridge=bridge)
    assert len(d)==1
    r=d.iloc[0]
    assert r.schema_generation=='pre9_early'
    assert r.target_accession=='P56817'
    assert r.target_identity_source=='exact_sequence_bridge'
    assert r.assay_identity_key.startswith('LEGACY_ASSAY_FP:')
    assert r.document_identity_key.startswith('DOI:')
    assert bool(r.primary_record_eligible)
    assert bool(r.relaxed_confidence_B_record_eligible)
    assert not bool(r.potential_duplicate_field_available)
    assert not bool(r.data_validity_field_available)


def test_pre9_without_relationship_type_is_readable_but_primary_uncertified(tmp_path: Path):
    early=tmp_path/'chembl02.db'; ref=tmp_path/'chembl09.db'; make_pre9(early,relationship_column=False,confidence=7,accession=True); make_reference_pre15(ref)
    bridge=build_target_sequence_bridge(ref)
    probe=schema_probe_row(early,'CHEMBL02')
    assert probe['schema_structural_probe_pass']
    assert not probe['primary_semantics_probe_pass']
    assert not probe['schema_probe_pass']
    d=extract_release(early,release_label='CHEMBL02',target_bridge=bridge)
    assert len(d)==1
    assert not bool(d.iloc[0].primary_record_eligible)
    assert bool(d.iloc[0].historical_high_confidence_B_sensitivity_eligible)
    assert d.iloc[0].historical_confidence_equivalence=='uncertified_no_relationship_type'


def test_pre9_homologue_is_not_primary(tmp_path: Path):
    early=tmp_path/'chembl02.db'; ref=tmp_path/'chembl09.db'; make_pre9(early,relationship_column=True,relationship='H',confidence=7,accession=True); make_reference_pre15(ref)
    bridge=build_target_sequence_bridge(ref)
    d=extract_release(early,release_label='CHEMBL02',target_bridge=bridge)
    assert not bool(d.iloc[0].primary_record_eligible)
    assert bool(d.iloc[0].historical_high_confidence_B_sensitivity_eligible)


def test_reference_bridge_marks_same_sequence_multiple_accessions_ambiguous(tmp_path: Path):
    ref=tmp_path/'chembl09_ambiguous.db'
    make_reference_pre15(ref)
    con=sqlite3.connect(ref)
    con.execute("INSERT INTO target_dictionary VALUES (52,'PROTEIN','Homo sapiens','MPEPTIDE','Q99999','CHEMBL_T2')")
    con.commit(); con.close()
    bridge=build_target_sequence_bridge(ref)
    hit=bridge[(bridge.sequence_normalized=='MPEPTIDE') & (bridge.organism_normalized=='HOMO SAPIENS')].iloc[0]
    assert hit.bridge_status=='ambiguous'
    assert int(hit.n_accessions)==2
    assert hit.target_accessions=='P56817|Q99999'
    assert pd.isna(hit.target_accession)


def test_pre9_ambiguous_sequence_bridge_is_excluded_not_guessed(tmp_path: Path):
    import pandas as pd
    ref=tmp_path/'chembl09_ambiguous.db'; early=tmp_path/'chembl02.db'
    make_reference_pre15(ref)
    con=sqlite3.connect(ref)
    con.execute("INSERT INTO target_dictionary VALUES (52,'PROTEIN','Homo sapiens','MPEPTIDE','Q99999','CHEMBL_T2')")
    con.commit(); con.close()
    make_pre9(early,relationship_column=True,relationship='D',confidence=7,accession=False)
    bridge=build_target_sequence_bridge(ref)
    d=extract_release(early,release_label='CHEMBL02',target_bridge=bridge)
    assert isinstance(d,pd.DataFrame)
    assert d.empty


def test_schema_probe_reports_linked_document_metadata_null_rates(tmp_path: Path):
    early=tmp_path/'chembl02_missing_meta.db'
    make_pre9(early,relationship_column=True,relationship='D',confidence=7,accession=True)
    con=sqlite3.connect(early)
    con.execute('UPDATE docs SET year=NULL, title=NULL, first_page=NULL')
    con.commit(); con.close()
    probe=schema_probe_row(early,'CHEMBL02')
    assert probe['linked_document_count']==1
    assert probe['document_year_null_rate']==1.0
    assert probe['document_title_null_rate']==1.0
    assert probe['document_first_page_null_rate']==1.0
    assert probe['document_weak_alias_bridgeable_rate']==0.0


def test_pre9_native_protein_label_is_historical_single_protein_equivalent(tmp_path: Path):
    early=tmp_path/'chembl02_native_type.db'
    make_pre9(early,relationship_column=True,relationship='D',confidence=7,accession=True)
    con=sqlite3.connect(early)
    con.execute('ALTER TABLE target_dictionary ADD COLUMN target_type TEXT')
    con.execute("UPDATE target_dictionary SET target_type='PROTEIN'")
    con.commit(); con.close()
    probe=schema_probe_row(early,'CHEMBL02')
    assert probe['historical_native_target_type_equivalent']=='PROTEIN'
    assert probe['target_type_equivalent_human_target_count']==1
    assert probe['target_type_semantics_probe_pass']
    d=extract_release(early,release_label='CHEMBL02')
    assert len(d)==1
    assert d.iloc[0].target_type=='PROTEIN'
    assert d.iloc[0].requested_scientific_target_type=='SINGLE PROTEIN'
    assert bool(d.iloc[0].primary_record_eligible)


def test_pre9_sql_applies_historical_high_confidence_floor_before_standardization(tmp_path: Path):
    """Low-confidence early activities can never enter primary/broad tracks and must not reach RDKit."""
    early=tmp_path/'chembl02_floor.db'; ref=tmp_path/'chembl09.db'
    make_pre9(early,relationship_column=True,relationship='D',confidence=7,accession=True)
    make_reference_pre15(ref)
    con=sqlite3.connect(early)
    # Same endpoint/target but historical confidence 1 and a different molecule.
    con.execute("INSERT INTO activities VALUES (2,12,21,32,42,'=',500,'nM',500,'nM',1,'IC50',NULL)")
    con.execute("INSERT INTO assays VALUES (12,'B','low confidence assay',21,1,'SRC-B')")
    con.execute("INSERT INTO assay2target VALUES (12,51,1,'D')")
    con.execute("INSERT INTO compounds VALUES (42,?,'LOWCONF')",(_molblock('CCN'),))
    con.commit(); con.close()
    bridge=build_target_sequence_bridge(ref)
    d,audit=extract_release(early,release_label='CHEMBL02',return_audit=True,target_bridge=bridge)
    assert len(d)==1
    assert set(pd.to_numeric(d.confidence_score))=={7}
    stats=audit['standardization_stats'].iloc[0]
    assert int(stats.raw_candidate_rows)==1


def make_pre9_modern_confidence(path: Path):
    con=sqlite3.connect(path)
    con.executescript('''
    CREATE TABLE activities (
      activity_id INTEGER, assay_id INTEGER, doc_id INTEGER, record_id INTEGER, molregno INTEGER,
      relation TEXT, published_value REAL, published_units TEXT, standard_value REAL,
      standard_units TEXT, standard_flag INTEGER, standard_type TEXT, activity_comment TEXT
    );
    CREATE TABLE assays (assay_id INTEGER, assay_type TEXT, description TEXT, doc_id INTEGER, src_id INTEGER, src_assay_id TEXT, chembl_id TEXT);
    CREATE TABLE assay2target (assay_id INTEGER, tid INTEGER, confidence_score INTEGER, relationship_type TEXT);
    CREATE TABLE target_dictionary (tid INTEGER, pref_name TEXT, organism TEXT, protein_sequence TEXT, protein_accession TEXT, target_type TEXT);
    CREATE TABLE compounds (molregno INTEGER, molfile TEXT, canonical_smiles TEXT, pref_name TEXT);
    CREATE TABLE docs (doc_id INTEGER, year INTEGER, title TEXT, doi TEXT, pubmed_id TEXT, journal TEXT, volume TEXT, issue TEXT, first_page TEXT);
    ''')
    con.execute("INSERT INTO target_dictionary VALUES (51,'BACE1','Homo sapiens','MPEPTIDE','P56817','PROTEIN')")
    con.execute("INSERT INTO assays VALUES (11,'B','direct score9',21,1,'S9','CHEMBL_A9')")
    con.execute("INSERT INTO assays VALUES (12,'B','homologue score8',21,1,'S8','CHEMBL_A8')")
    con.execute("INSERT INTO assay2target VALUES (11,51,9,'D')")
    con.execute("INSERT INTO assay2target VALUES (12,51,8,'H')")
    con.execute("INSERT INTO activities VALUES (1,11,21,31,41,'=',100,'nM',100,'nM',1,'IC50',NULL)")
    con.execute("INSERT INTO activities VALUES (2,12,21,32,42,'=',200,'nM',200,'nM',1,'IC50',NULL)")
    con.execute("INSERT INTO compounds VALUES (41,?,'CCO','ETHANOL')",(_molblock('CCO'),))
    con.execute("INSERT INTO compounds VALUES (42,?,'CCN','ETHYLAMINE')",(_molblock('CCN'),))
    con.execute("INSERT INTO docs VALUES (21,2010,'Paper','10.1/x','123','J','1','1','1')")
    con.commit(); con.close()


def test_pre9_schema_with_modern_confidence_score_uses_0_9_semantics(tmp_path: Path):
    db=tmp_path/'chembl08_like.db'
    make_pre9_modern_confidence(db)
    info=inspect_schema(db)
    assert info['schema_generation']=='pre9_early'
    assert info['confidence_regime']=='modern_0_9_legacy_location'
    d,audit=extract_release(db,release_label='CHEMBL08',return_audit=True,extraction_min_confidence=8)
    assert set(pd.to_numeric(d.confidence_score))=={8,9}
    r9=d.loc[pd.to_numeric(d.confidence_score).eq(9)].iloc[0]
    r8=d.loc[pd.to_numeric(d.confidence_score).eq(8)].iloc[0]
    assert bool(r9.primary_record_eligible)
    assert bool(r9.confidence9_all_assay_eligible)
    assert not bool(r8.primary_record_eligible)
    assert bool(r8.confidence8_B_record_eligible)
    assert bool(r8.broad_prior_knowledge_eligible)
    assert set(d.historical_confidence_equivalence)=={'modern_0_9_legacy_location'}
    stats=audit['standardization_stats'].iloc[0]
    assert int(stats.raw_candidate_rows)==2


def test_pre9_modern_confidence_multitarget_native_assay_is_excluded_before_primary(tmp_path: Path):
    db=tmp_path/'chembl08_multid.db'
    make_pre9_modern_confidence(db)
    con=sqlite3.connect(db)
    con.execute("INSERT INTO target_dictionary VALUES (52,'OTHER','Homo sapiens','MOTHER','Q99999','PROTEIN')")
    con.execute("INSERT INTO assay2target VALUES (11,52,9,'D')")
    con.commit(); con.close()
    d,audit=extract_release(db,release_label='CHEMBL08',return_audit=True,extraction_min_confidence=8)
    # Native assay 11 is ambiguous (two direct accessions) and is removed entirely; assay 12 remains sensitivity-only.
    assert set(pd.to_numeric(d.native_assay_id))=={12}
    assert int(d.primary_record_eligible.sum())==0
    a=audit['pre15_multitarget_assay_exclusions']
    assert len(a)==1
    r=a.iloc[0]
    assert int(r.native_assay_id)==11
    assert int(r.n_target_accessions)==2
    assert int(r.n_D_targets)==2
    assert not bool(r.threshold_applicable)
