import pandas as pd

from ad_f0.chembl_adapter import _pre15_multitarget_assay_audit_and_filter


def _rows():
    return pd.DataFrame([
        dict(native_assay_id=10, assay_chembl_id='A10', assay_type='B', target_accession='P1',
             target_relationship_type='D', confidence_score=9, schema_generation='pre9_early'),
        dict(native_assay_id=10, assay_chembl_id='A10', assay_type='B', target_accession='P2',
             target_relationship_type='D', confidence_score=9, schema_generation='pre9_early'),
        dict(native_assay_id=11, assay_chembl_id='A11', assay_type='B', target_accession='P3',
             target_relationship_type='D', confidence_score=9, schema_generation='pre9_early'),
    ])


def test_pre15_multitarget_native_assay_is_excluded_without_threshold():
    out,audit=_pre15_multitarget_assay_audit_and_filter(_rows(),'CHEMBL08')
    assert set(out.native_assay_id)=={11}
    assert len(audit)==1
    r=audit.iloc[0]
    assert r.native_assay_id==10
    assert r.n_target_accessions==2
    assert r.n_D_targets==2
    assert r.n_H_targets==0
    assert r.exclusion_reason=='historical_assay2target_multitarget'
    assert not bool(r.threshold_applicable)
    assert r.candidate_assays==2
    assert r.excluded_assays==1


def test_pre15_multitarget_detection_does_not_use_cross_release_assay_key():
    d=_rows()
    # Deliberately make each target row look like a different longitudinal assay fingerprint.
    d['assay_identity_key']=['LEGACY_ASSAY_FP:X','LEGACY_ASSAY_FP:Y','LEGACY_ASSAY_FP:Z']
    out,audit=_pre15_multitarget_assay_audit_and_filter(d,'CHEMBL07')
    assert set(out.native_assay_id)=={11}
    assert len(audit)==1
    assert audit.iloc[0].native_assay_id==10
