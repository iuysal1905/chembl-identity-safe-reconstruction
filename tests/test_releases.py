import pandas as pd
from ad_f0.releases import _extract_chembl_release_dates, resolve_exact_release_dates, validate_exact_dates


def test_chembl_release_payload_extracts_creation_dates():
    payload = {
        'chembl_releases': [
            {'chembl_release': 'CHEMBL_2', 'creation_date': '2009-11-30T00:00:00'},
            {'chembl_release': 'CHEMBL17', 'creation_date': '2013-08-29'},
        ]
    }
    out = _extract_chembl_release_dates(payload)
    assert out['CHEMBL02'] == '2009-11-30'
    assert out['CHEMBL17'] == '2013-08-29'


class _Resp:
    status_code = 200
    def raise_for_status(self):
        pass
    def json(self):
        return {'chembl_releases': [{'chembl_release': 'CHEMBL_2', 'creation_date': '2009-11-30'}]}


class _Session:
    def get(self, url, timeout=30):
        return _Resp()


def test_manual_patch_override_has_priority_over_official_resource():
    manifest = pd.DataFrame([
        {'release': '2', 'release_label': 'CHEMBL02'},
        {'release': '22.1', 'release_label': 'CHEMBL22.1'},
    ])
    overrides = pd.DataFrame([
        {'release': '22.1', 'exact_release_date': '2016-11-17', 'source': 'Official ChEMBL patch announcement',
         'date_type': 'official_patch_announcement_date', 'date_field': 'ChEMBL-og publication date'}
    ])
    out = resolve_exact_release_dates(manifest, overrides=overrides, session=_Session())
    a = out.set_index('release_label')
    assert a.loc['CHEMBL02', 'exact_release_date'] == '2009-11-30'
    assert a.loc['CHEMBL02', 'exact_date_type'] == 'creation_date'
    assert a.loc['CHEMBL22.1', 'exact_release_date'] == '2016-11-17'
    assert a.loc['CHEMBL22.1', 'exact_date_type'] == 'official_patch_announcement_date'
    assert validate_exact_dates(out) == []
