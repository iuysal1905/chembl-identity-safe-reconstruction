import io
import tarfile
from pathlib import Path

from ad_f0.archive_download import format_release_token, sqlite_archive_urls, _safe_extract_single_db


def test_release_tokens_and_official_sqlite_urls():
    assert format_release_token(1)=='01'
    assert format_release_token('CHEMBL09')=='09'
    assert format_release_token('22.1')=='22_1'
    urls=sqlite_archive_urls('CHEMBL02')
    assert urls[0].endswith('/chembl_02/chembl_02_sqlite.tar.gz')
    assert urls[1].endswith('/chembl_02/archived/chembl_02_sqlite.tar.gz')


def test_safe_extract_single_sqlite(tmp_path: Path):
    tar=tmp_path/'x.tar.gz'
    payload=b'SQLite format 3\\x00FAKE'
    with tarfile.open(tar,'w:gz') as tf:
        info=tarfile.TarInfo('nested/chembl_02.db'); info.size=len(payload)
        tf.addfile(info,io.BytesIO(payload))
    out=_safe_extract_single_db(tar,tmp_path/'out')
    assert out.name=='chembl_02.db'
    assert out.read_bytes()==payload
