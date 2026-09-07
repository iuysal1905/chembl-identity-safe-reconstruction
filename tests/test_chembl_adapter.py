import sqlite3
from pathlib import Path
import pandas as pd
import pytest
from ad_f0.chembl_adapter import extract_release


def make_db(path: Path, multi_component=False, duplicate_value=0, include_null_duplicate=False,
            assay_type='B', confidence=9, validity=None, units='nM'):
    con=sqlite3.connect(path)
    con.executescript('''
    CREATE TABLE activities (assay_id INTEGER, molregno INTEGER, standard_type TEXT, standard_relation TEXT,
      standard_value REAL, standard_units TEXT, data_validity_comment TEXT, potential_duplicate REAL);
    CREATE TABLE assays (assay_id INTEGER, chembl_id TEXT, tid INTEGER, doc_id INTEGER, confidence_score INTEGER, assay_type TEXT);
    CREATE TABLE target_dictionary (tid INTEGER, target_type TEXT, organism TEXT, chembl_id TEXT);
    CREATE TABLE target_components (tid INTEGER, component_id INTEGER);
    CREATE TABLE component_sequences (component_id INTEGER, accession TEXT, sequence TEXT);
    CREATE TABLE compound_structures (molregno INTEGER, canonical_smiles TEXT);
    CREATE TABLE molecule_dictionary (molregno INTEGER, chembl_id TEXT, pref_name TEXT);
    CREATE TABLE docs (doc_id INTEGER, chembl_id TEXT, year INTEGER, title TEXT, doi TEXT);
    ''')
    con.execute('INSERT INTO activities VALUES (1,1,?,?,?,?,?,?)',('IC50','=',1000,units,validity,duplicate_value))
    if include_null_duplicate:
        con.execute("INSERT INTO activities VALUES (1,2,'IC50','=',2000,'nM',NULL,NULL)")
    con.execute('INSERT INTO assays VALUES (1,\'CHEMBL_ASSAY\',10,20,?,?)',(confidence,assay_type))
    con.execute("INSERT INTO target_dictionary VALUES (10,'SINGLE PROTEIN','Homo sapiens','CHEMBL_T')")
    con.execute("INSERT INTO target_components VALUES (10,30)")
    con.execute("INSERT INTO component_sequences VALUES (30,'P56817','MPEPTIDE')")
    if multi_component:
        con.execute("INSERT INTO target_components VALUES (10,31)")
        con.execute("INSERT INTO component_sequences VALUES (31,'Q99999','MPEPTIDE2')")
    con.execute("INSERT INTO compound_structures VALUES (1,'CCO')")
    con.execute("INSERT INTO molecule_dictionary VALUES (1,'CHEMBL_M1','ETHANOL')")
    if include_null_duplicate:
        con.execute("INSERT INTO compound_structures VALUES (2,'CCN')")
        con.execute("INSERT INTO molecule_dictionary VALUES (2,'CHEMBL_M2','ETHYLAMINE')")
    con.execute("INSERT INTO docs VALUES (20,'CHEMBL_DOC',2009,'Example','10.1/x')")
    con.commit(); con.close()


def test_minimal_sqlite_extraction(tmp_path: Path):
    db=tmp_path/'chembl.db'; make_db(db)
    df=extract_release(db,endpoints=('IC50','Ki'),extraction_min_confidence=8)
    assert len(df)==1
    r=df.iloc[0]
    assert r.target_accession=='P56817'
    assert abs(r.standard_value_molar-1e-6)<1e-15
    assert r.quality_eligible and r.structure_eligible and r.primary_record_eligible
    assert r.assay_type=='B' and r.confidence_score==9


def test_potential_duplicate_float_with_null_is_rejected(tmp_path: Path):
    db=tmp_path/'chembl.db'; make_db(db,duplicate_value=1,include_null_duplicate=True)
    df=extract_release(db)
    flagged=df.loc[df.molecule_chembl_id=='CHEMBL_M1'].iloc[0]
    clean=df.loc[df.molecule_chembl_id=='CHEMBL_M2'].iloc[0]
    assert bool(flagged.potential_duplicate_flagged)
    assert not bool(flagged.quality_eligible)
    assert not bool(clean.potential_duplicate_flagged)
    assert bool(clean.quality_eligible)


def test_multicomponent_single_protein_fails_closed(tmp_path: Path):
    db=tmp_path/'chembl.db'; make_db(db,multi_component=True)
    with pytest.raises(RuntimeError,match='multiple target accessions'):
        extract_release(db)


def test_confidence9_and_binding_are_primary_only(tmp_path: Path):
    db=tmp_path/'c8.db'; make_db(db,confidence=8,assay_type='B')
    d=extract_release(db)
    assert bool(d.iloc[0].confidence8_B_record_eligible)
    assert not bool(d.iloc[0].primary_record_eligible)
    db2=tmp_path/'func.db'; make_db(db2,confidence=9,assay_type='F')
    d2=extract_release(db2)
    assert bool(d2.iloc[0].confidence9_all_assay_eligible)
    assert not bool(d2.iloc[0].primary_record_eligible)


def test_nonconvertible_units_retained_for_audit(tmp_path: Path):
    db=tmp_path/'ug.db'; make_db(db,units='ug.mL-1')
    d=extract_release(db)
    assert len(d)==1
    assert not bool(d.iloc[0].unit_convertible)
    assert not bool(d.iloc[0].primary_record_eligible)


def test_manually_validated_is_eligible(tmp_path: Path):
    db=tmp_path/'mv.db'; make_db(db,validity='Manually validated')
    d=extract_release(db)
    assert bool(d.iloc[0].quality_eligible)


def test_post15_does_not_widen_protein_label_to_single_protein(tmp_path: Path):
    db=tmp_path/'post15_wrong_type.db'; make_db(db)
    con=sqlite3.connect(db)
    con.execute("UPDATE target_dictionary SET target_type='PROTEIN' WHERE tid=10")
    con.commit(); con.close()
    df=extract_release(db)
    assert df.empty


def test_schema_probe_reports_generation_specific_target_type_semantics(tmp_path: Path):
    db=tmp_path/'post15_probe.db'; make_db(db)
    from ad_f0.chembl_adapter import schema_probe_row
    p=schema_probe_row(db,'CHEMBL17')
    assert p['requested_scientific_target_type']=='SINGLE PROTEIN'
    assert p['historical_native_target_type_equivalent']=='SINGLE PROTEIN'
    assert p['target_type_equivalent_human_target_count']==1
    assert p['target_type_semantics_probe_pass']
