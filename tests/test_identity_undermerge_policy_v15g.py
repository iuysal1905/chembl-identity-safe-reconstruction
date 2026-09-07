import pandas as pd
import pytest

from ad_f0.first_availability import (
    canonicalize_cross_release_identities,
    HistoricalIdentityConflictError,
    identity_reconciliation_policy_summary,
    release_local_identity_rate_by_schema,
)


def row(release, *, year='', volume='40', page='1200', title='', doc_native=55, assay_native=101,
        doc_chem='', assay_chem='', src='', src_assay='', desc='Inhibition of AChE'):
    return dict(
        release_label=release, schema_generation='pre9_early',
        document_journal='J Med Chem', document_year=year, document_volume=volume,
        document_issue='', document_first_page=page, document_title=title,
        document_doi='', document_pubmed_id='', document_chembl_id=doc_chem,
        native_document_id=doc_native, assay_description=desc, assay_type='B',
        target_accession='P22303', assay_chembl_id=assay_chem,
        assay_source_id=src, assay_source_assay_id=src_assay, native_assay_id=assay_native,
    )


def test_year_independent_journal_volume_page_bridges_missing_year_to_enriched_release():
    a=pd.DataFrame([row('CHEMBL02',year='')])
    b=pd.DataFrame([row('CHEMBL05',year='1997')])
    frames,da,aa=canonicalize_cross_release_identities([a,b])
    assert frames[0].document_identity_key.iloc[0]==frames[1].document_identity_key.iloc[0]
    assert frames[0].assay_identity_key.iloc[0]==frames[1].assay_identity_key.iloc[0]
    assert not bool(da.release_local_only_component.iloc[0])
    assert not bool(aa.release_local_only_component.iloc[0])


def test_release_local_only_fraction_above_one_percent_fails_when_family_has_multiple_releases():
    # No longitudinally admissible alias at all. Each occurrence remains release-local.
    a=pd.DataFrame([row('CHEMBL02',year='',volume='',page='',title='',doc_native=1,assay_native=1)])
    b=pd.DataFrame([row('CHEMBL05',year='',volume='',page='',title='',doc_native=1,assay_native=1)])
    with pytest.raises(HistoricalIdentityConflictError) as ei:
        canonicalize_cross_release_identities([a,b],release_local_only_max_fraction=0.01)
    audit=ei.value.audit
    assert 'release_local_only_fraction' in audit.columns
    assert float(audit.release_local_only_fraction.max()) > 0.01


def test_release_local_components_do_not_dilute_weak_internal_id_overmerge_rate():
    # One weakly bridged document with two different internal ChEMBL IDs is suspicious.
    a=[row('CHEMBL07',year='2005',title='Same paper',doc_chem='CHEMBL_D1',doc_native='',assay_chem='CHEMBL_A1')]
    b=[row('CHEMBL08',year='2005',title='Same paper',doc_chem='CHEMBL_D2',doc_native='',assay_chem='CHEMBL_A1')]
    # Add many local-only components that would dilute the old denominator below 1%.
    for i in range(200):
        a.append(row('CHEMBL07',year='',volume='',page='',title='',doc_native=1000+i,assay_native=2000+i,desc=f'a{i}'))
    for i in range(200):
        b.append(row('CHEMBL08',year='',volume='',page='',title='',doc_native=5000+i,assay_native=6000+i,desc=f'b{i}'))
    with pytest.raises(HistoricalIdentityConflictError):
        canonicalize_cross_release_identities(
            [pd.DataFrame(a),pd.DataFrame(b)],
            internal_id_weak_only_max_fraction=0.01,
            release_local_only_max_fraction=1.0,  # isolate the over-merge denominator test
        )


def test_release_local_rate_summary_is_symmetric_for_documents_and_assays():
    a=pd.DataFrame([row('CHEMBL02',year='1997')])
    b=pd.DataFrame([row('CHEMBL05',year='1997')])
    _,da,aa=canonicalize_cross_release_identities([a,b])
    out=release_local_identity_rate_by_schema(da,aa)
    assert set(out.kind)=={'document','assay'}
    assert set(out.release_local_only_fraction)=={0.0}
    summary=identity_reconciliation_policy_summary(da,aa)
    assert 'n_overmerge_denominator_components' in summary.columns
    assert 'release_local_policy_pass' in summary.columns
