import sqlite3
from pathlib import Path
import pandas as pd
import pytest

from ad_f0.chembl_adapter import extract_release, inspect_schema, schema_probe_row


def make_legacy_db(path: Path, relationship='D', confidence=9, assay_type='B'):
    con=sqlite3.connect(path)
    con.executescript('''
    CREATE TABLE activities (
      activity_id INTEGER, assay_id INTEGER, doc_id INTEGER, record_id INTEGER, molregno INTEGER,
      relation TEXT, published_value REAL, published_units TEXT, standard_value REAL,
      standard_units TEXT, standard_flag INTEGER, standard_type TEXT, activity_comment TEXT
    );
    CREATE TABLE assays (
      assay_id INTEGER, assay_type TEXT, description TEXT, doc_id INTEGER, src_id INTEGER,
      src_assay_id TEXT, chembl_id TEXT
    );
    CREATE TABLE assay2target (
      assay_id INTEGER, tid INTEGER, assay_tax_id INTEGER, assay_organism TEXT,
      relationship_type TEXT, complex INTEGER, multi INTEGER, confidence_score INTEGER,
      assay_strain TEXT, curated_by TEXT
    );
    CREATE TABLE target_dictionary (
      tid INTEGER, target_type TEXT, db_source TEXT, description TEXT, gene_names TEXT,
      pref_name TEXT, synonyms TEXT, keywords TEXT, protein_sequence TEXT,
      protein_md5sum TEXT, tax_id INTEGER, organism TEXT, tissue TEXT, strain TEXT,
      db_version TEXT, cell_line TEXT, protein_accession TEXT, ec_number TEXT, chembl_id TEXT
    );
    CREATE TABLE compound_structures (molregno INTEGER, canonical_smiles TEXT);
    CREATE TABLE molecule_dictionary (molregno INTEGER, pref_name TEXT, chembl_id TEXT);
    CREATE TABLE docs (doc_id INTEGER, year TEXT, chembl_id TEXT, title TEXT, doi TEXT);
    ''')
    con.execute("INSERT INTO activities VALUES (1,11,21,31,41,'=',1000,'nM',1000,'nM',1,'IC50',NULL)")
    con.execute("INSERT INTO assays VALUES (11,?,'legacy assay',21,1,'X','CHEMBL_ASSAY_L')",(assay_type,))
    con.execute("INSERT INTO assay2target VALUES (11,51,9606,'Homo sapiens',?,0,0,?,NULL,'Man')",(relationship,confidence))
    con.execute("INSERT INTO target_dictionary VALUES (51,'PROTEIN','SWISS-PROT','x','BACE1','BACE1',NULL,NULL,'MPEPTIDE',NULL,9606,'Homo sapiens',NULL,NULL,NULL,NULL,'P56817',NULL,'CHEMBL_T')")
    con.execute("INSERT INTO compound_structures VALUES (41,'CCO')")
    con.execute("INSERT INTO molecule_dictionary VALUES (41,'ETHANOL','CHEMBL_M')")
    con.execute("INSERT INTO docs VALUES (21,'2009','CHEMBL_DOC','Legacy','10.1/legacy')")
    con.commit(); con.close()


def test_pre15_legacy_schema_extracts_direct_mapping(tmp_path: Path):
    db=tmp_path/'legacy.db'; make_legacy_db(db, relationship='D', confidence=9)
    info=inspect_schema(db)
    assert info['schema_generation']=='pre15_legacy'
    assert info['assay_type_available']
    probe=schema_probe_row(db,'CHEMBL09')
    assert probe['schema_probe_pass']
    assert probe['historical_native_target_type_equivalent']=='PROTEIN'
    assert probe['target_type_equivalent_human_target_count']==1
    assert probe['target_type_semantics_probe_pass']
    d=extract_release(db,release_label='CHEMBL09')
    r=d.iloc[0]
    assert r.schema_generation=='pre15_legacy'
    assert r.target_accession=='P56817'
    assert r.target_relationship_type=='D'
    assert r.target_type=='PROTEIN'
    assert r.historical_native_target_type_equivalent=='PROTEIN'
    assert r.requested_scientific_target_type=='SINGLE PROTEIN'
    assert bool(r.primary_record_eligible)


def test_legacy_homologue_score8_is_sensitivity_not_primary(tmp_path: Path):
    db=tmp_path/'legacy_h.db'; make_legacy_db(db, relationship='H', confidence=8)
    d=extract_release(db,release_label='CHEMBL09')
    r=d.iloc[0]
    assert bool(r.confidence8_B_record_eligible)
    assert bool(r.broad_prior_knowledge_eligible)
    assert not bool(r.primary_record_eligible)
    assert bool(r.legacy_mapping_consistent)


def test_legacy_confidence9_not_direct_is_not_primary(tmp_path: Path):
    db=tmp_path/'legacy_badmap.db'; make_legacy_db(db, relationship='H', confidence=9)
    d=extract_release(db,release_label='CHEMBL09')
    r=d.iloc[0]
    assert not bool(r.primary_record_eligible)
    assert not bool(r.legacy_mapping_consistent)
