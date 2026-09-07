import pandas as pd
import pytest
from ad_f0.chembl_adapter import _multicomponent_audit_and_filter, MultiComponentThresholdError


def frame(n_good=1000):
    rows=[]
    for i in range(n_good):
        rows.append({'assay_chembl_id':f'A{i}','target_accession':'P1','schema_generation':'post15'})
    rows += [
        {'assay_chembl_id':'BAD','target_accession':'P1','schema_generation':'post15'},
        {'assay_chembl_id':'BAD','target_accession':'P2','schema_generation':'post15'},
    ]
    return pd.DataFrame(rows)


def test_multicomponent_below_or_equal_threshold_is_excluded_and_logged():
    d,a=_multicomponent_audit_and_filter(frame(1000),'CHEMBL17',0.001)
    assert 'BAD' not in set(d.assay_chembl_id)
    assert len(a)==1
    assert a.iloc[0].excluded_assays==1
    assert a.iloc[0].candidate_assays==1001
    assert a.iloc[0].exclusion_fraction <= 0.001


def test_multicomponent_above_threshold_fails_closed_with_audit():
    with pytest.raises(MultiComponentThresholdError) as e:
        _multicomponent_audit_and_filter(frame(10),'CHEMBL17',0.001)
    assert len(e.value.audit)==1
