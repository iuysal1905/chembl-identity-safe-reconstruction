import pandas as pd
import pytest

from ad_f0.first_availability import (
    canonicalize_cross_release_identities,
    HistoricalIdentityConflictError,
    weak_alias_quarantine_summary,
)


def row(release, *, doi='', pmid='', chem='', year='2016', journal='Eur J Med Chem', volume='120',
        page='100', title='Paper title', assay='CHEMBL_A1', src='1', src_assay='S1', desc='Binding assay'):
    return dict(
        release_label=release, schema_generation='post15',
        document_journal=journal, document_year=year, document_volume=volume,
        document_issue='', document_first_page=page, document_title=title,
        document_doi=doi, document_pubmed_id=pmid, document_chembl_id=chem,
        native_document_id='', assay_description=desc, assay_type='B',
        target_accession='P22303', assay_chembl_id=assay,
        assay_source_id=src, assay_source_assay_id=src_assay,
        native_assay_id='', assay_identity_key='',
    )


def test_conflicting_page_alias_is_quarantined_not_allowed_to_merge_external_documents():
    a=pd.DataFrame([row('CHEMBL23',doi='10.1016/A',pmid='111',chem='CHEMBL_D1',title='Alpha')])
    b=pd.DataFrame([row('CHEMBL23',doi='10.1016/B',pmid='222',chem='CHEMBL_D2',title='Beta',assay='CHEMBL_A2',src_assay='S2')])
    frames,da,_=canonicalize_cross_release_identities([a,b],release_local_only_max_fraction=1.0)
    assert frames[0].iloc[0].document_identity_key != frames[1].iloc[0].document_identity_key
    q=weak_alias_quarantine_summary(da,pd.DataFrame())
    assert not q.empty
    assert q.quarantined_weak_aliases.str.contains('DOC_PAGE:').any()


def test_conflicting_title_alias_is_quarantined():
    a=pd.DataFrame([row('CHEMBL23',doi='10.1/A',pmid='111',chem='CHEMBL_D1',title='Editorial',page='10')])
    b=pd.DataFrame([row('CHEMBL24',doi='10.1/B',pmid='222',chem='CHEMBL_D2',title='Editorial',page='20',assay='CHEMBL_A2',src_assay='S2')])
    frames,da,_=canonicalize_cross_release_identities([a,b],release_local_only_max_fraction=1.0)
    assert frames[0].iloc[0].document_identity_key != frames[1].iloc[0].document_identity_key
    assert da.n_quarantined_weak_aliases.sum() >= 1


def test_safe_weak_alias_still_bridges_sparse_to_external_anchored_occurrence():
    a=pd.DataFrame([row('CHEMBL02',doi='',pmid='',chem='',year='',title='',journal='J Med Chem',volume='40',page='1200',assay='',src='',src_assay='')])
    b=pd.DataFrame([row('CHEMBL05',doi='10.1/X',pmid='123',chem='CHEMBL_D1',year='1997',title='Known paper',journal='J Med Chem',volume='40',page='1200')])
    frames,da,_=canonicalize_cross_release_identities([a,b],release_local_only_max_fraction=1.0)
    assert frames[0].iloc[0].document_identity_key == frames[1].iloc[0].document_identity_key
    assert int(da.n_quarantined_weak_aliases.sum()) == 0


def test_transitive_weak_chain_cannot_connect_distinct_external_documents():
    # A--TITLE--B--PAGE--C. Neither weak alias alone has two external anchors in a
    # direct pair, but together they would create a conflicting DOI component.
    a=pd.DataFrame([row('CHEMBL20',doi='10.1/A',pmid='111',chem='CHEMBL_D1',title='Shared title',journal='J1',volume='1',page='1')])
    b=pd.DataFrame([row('CHEMBL21',doi='',pmid='',chem='',title='Shared title',journal='J2',volume='2',page='2',assay='',src='',src_assay='')])
    c=pd.DataFrame([row('CHEMBL22',doi='10.1/C',pmid='333',chem='CHEMBL_D3',title='Different title',journal='J2',volume='2',page='2',assay='CHEMBL_A3',src_assay='S3')])
    frames,da,_=canonicalize_cross_release_identities([a,b,c],release_local_only_max_fraction=1.0)
    keys=[f.iloc[0].document_identity_key for f in frames]
    assert keys[0] != keys[2]
    assert da.n_quarantined_weak_aliases.sum() >= 2


def test_strong_internal_alias_conflicting_external_ids_still_fails_closed():
    a=pd.DataFrame([row('CHEMBL23',doi='10.1/A',pmid='111',chem='CHEMBL_D1',title='Alpha',page='10')])
    b=pd.DataFrame([row('CHEMBL24',doi='10.1/B',pmid='222',chem='CHEMBL_D1',title='Beta',page='20')])
    with pytest.raises(HistoricalIdentityConflictError):
        canonicalize_cross_release_identities([a,b],release_local_only_max_fraction=1.0)


def test_assay_semantic_alias_cannot_merge_conflicting_source_assay_ids():
    # Same document/description/target/type but distinct authoritative source-assay IDs.
    a=pd.DataFrame([row('CHEMBL23',doi='10.1/X',pmid='111',chem='CHEMBL_D1',assay='',src='SRC',src_assay='A1')])
    b=pd.DataFrame([row('CHEMBL24',doi='10.1/X',pmid='111',chem='CHEMBL_D1',assay='',src='SRC',src_assay='A2')])
    frames,_,aa=canonicalize_cross_release_identities([a,b],release_local_only_max_fraction=1.0)
    assert frames[0].iloc[0].assay_identity_key != frames[1].iloc[0].assay_identity_key
    q=weak_alias_quarantine_summary(pd.DataFrame(),aa)
    assert not q.empty
    assert q.quarantined_weak_aliases.str.contains('ASSAY_SEM:').any()
