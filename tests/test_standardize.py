from ad_f0.standardize import standardize_smiles


def test_salt_parent_and_full_inchikey():
    a=standardize_smiles('CC(=O)O.[Na+]')
    b=standardize_smiles('CC(=O)O')
    assert a['status']=='ok' and b['status']=='ok'
    assert a['full_inchikey']==b['full_inchikey']
    assert len(a['full_inchikey'].split('-'))==3


def test_stereo_preserved():
    a=standardize_smiles('N[C@@H](C)C(=O)O')
    b=standardize_smiles('N[C@H](C)C(=O)O')
    assert a['full_inchikey'] != b['full_inchikey']


def test_molblock_standardization_supported():
    from rdkit import Chem
    from ad_f0.standardize import standardize_molblock
    mb=Chem.MolToMolBlock(Chem.MolFromSmiles('C[C@H](O)Cl'))
    r=standardize_molblock(mb)
    assert r['status']=='ok'
    assert r['full_inchikey']


def test_batch_standardizes_unique_input_once_and_reuses_persistent_cache(tmp_path):
    from ad_f0.standardize import standardize_structure_batch, clear_standardization_caches
    cache = tmp_path / 'structure_cache.sqlite'
    out1, stats1 = standardize_structure_batch(['CCO','CCO','CCN'], ['', '', ''], cache_db_path=cache)
    assert [r['status'] for r in out1] == ['ok','ok','ok']
    assert stats1['unique_smiles_inputs'] == 2
    assert stats1['standardized_new'] == 2
    assert stats1['persistent_cache_hits'] == 0
    clear_standardization_caches()
    out2, stats2 = standardize_structure_batch(['CCO','CCN'], ['', ''], cache_db_path=cache)
    assert [r['full_inchikey'] for r in out2] == [out1[0]['full_inchikey'], out1[2]['full_inchikey']]
    assert stats2['standardized_new'] == 0
    assert stats2['persistent_cache_hits'] == 2


def test_batch_preserves_smiles_then_molblock_fallback(tmp_path):
    from rdkit import Chem
    from ad_f0.standardize import standardize_structure, standardize_structure_batch
    mb = Chem.MolToMolBlock(Chem.MolFromSmiles('CCO'))
    expected = standardize_structure('not-a-smiles', mb)
    out, _ = standardize_structure_batch(['not-a-smiles'], [mb], cache_db_path=tmp_path/'c.sqlite')
    assert out[0]['status'] == expected['status'] == 'ok'
    assert out[0]['full_inchikey'] == expected['full_inchikey']
