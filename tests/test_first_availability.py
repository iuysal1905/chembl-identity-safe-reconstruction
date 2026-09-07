import pandas as pd
from ad_f0.first_availability import reconstruct_first_availability


def row(release, assay='CHEMBL_A', endpoint='IC50', val=1e-6, doc='DOC', relation='='):
    return pd.DataFrame([dict(release_label=release,assay_chembl_id=assay,full_inchikey='AAAA-BBBB-C',
        standard_type=endpoint,standard_relation=relation,standard_value_molar=val,target_accession='P56817',
        molecule_pref_name='X',molecule_chembl_id='CHEMBL_M',canonical_isomeric_smiles='CC',murcko_scaffold_smiles='',
        quality_eligible=True,structure_eligible=True,unit_convertible=True,primary_record_eligible=True,
        document_year=2009,document_chembl_id=doc)])


def manifest():
    return pd.DataFrame({'release_label':['CHEMBL01','CHEMBL02'],
                         'exact_release_date':['2009-10-15','2015-12-15']})


def test_first_release_not_activity_id():
    first,audit=reconstruct_first_availability([row('CHEMBL01'),row('CHEMBL02')],manifest())
    assert len(first)==1
    assert first.iloc[0].first_available_release=='CHEMBL01'
    assert audit.iloc[0].n_release_occurrences==2


def test_same_release_pair_tie_has_one_representative_but_two_first_rows():
    a=row('CHEMBL02',assay='A2',doc='D2',val=1e-6)
    b=row('CHEMBL02',assay='A1',doc='D1',val=2e-6)
    first,_=reconstruct_first_availability([pd.concat([a,b],ignore_index=True)],
        pd.DataFrame({'release_label':['CHEMBL02'],'exact_release_date':['2015-12-15']}))
    assert len(first)==2
    assert first.is_pair_first_release.sum()==2
    assert first.is_pair_first_representative.sum()==1
    assert set(first.pair_first_tie_count)=={2}
    assert first.pair_key.nunique()==1


def test_endpoint_specific_pair_keys_remain_distinct():
    ki=row('CHEMBL01',endpoint='Ki',assay='AKI')
    ic=row('CHEMBL02',endpoint='IC50',assay='AIC')
    first,_=reconstruct_first_availability([ki,ic],manifest())
    assert first.pair_key.nunique()==2
    assert first.cross_endpoint_pair_key.nunique()==1


def test_same_release_identical_composite_key_collapses_with_multiplicity():
    a=row('CHEMBL01',doc='DOC')
    b=row('CHEMBL01',doc='DOC')
    first,audit=reconstruct_first_availability([pd.concat([a,b],ignore_index=True)],
        pd.DataFrame({'release_label':['CHEMBL01'],'exact_release_date':['2009-10-15']}))
    assert len(first)==1
    assert int(first.iloc[0].within_release_duplicate_multiplicity)==2
    assert int(audit.iloc[0].within_release_duplicate_multiplicity)==2


def test_earliest_archive_is_left_censor_boundary_even_if_later_records_exist():
    # CHEMBL01 is processed and contains an eligible record; that record is left-censored.
    f1=row('CHEMBL01',assay='A0')
    f2=row('CHEMBL02',assay='A2',val=2e-6)
    first,audit=reconstruct_first_availability([f1,f2],manifest())
    r0=first[first.first_available_release.eq('CHEMBL01')].iloc[0]
    assert bool(r0.first_availability_left_censored)
    assert bool(r0.pair_first_left_censored)
    later=first[first.first_available_release.eq('CHEMBL02')]
    if len(later):
        assert not bool(later.iloc[0].first_availability_left_censored)


def test_broad_prior_other_endpoint_can_come_from_nonprimary_record():
    from ad_f0.first_availability import annotate_broad_prior_other_endpoint
    # Primary IC50 appears in release 2. Earlier Ki is absent from primary track but present in broad track.
    ic=row('CHEMBL02',endpoint='IC50',assay='AIC')
    ki=row('CHEMBL01',endpoint='Ki',assay='AKI')
    ki['primary_record_eligible']=False
    ki['broad_prior_knowledge_eligible']=True
    ic['broad_prior_knowledge_eligible']=True
    # primary first only sees IC50
    primary,_=reconstruct_first_availability([ki,ic],manifest(),eligibility_col='primary_record_eligible')
    broad,_=reconstruct_first_availability([ki,ic],manifest(),eligibility_col='broad_prior_knowledge_eligible')
    annotated=annotate_broad_prior_other_endpoint(primary,broad)
    assert bool(annotated.iloc[0].prior_other_endpoint_exposure)
    assert pd.Timestamp(annotated.iloc[0].prior_other_endpoint_date)==pd.Timestamp('2009-10-15')


def test_activity_identity_prefers_frozen_assay_identity_key_over_missing_native_id():
    a=row('CHEMBL01',assay='')
    b=row('CHEMBL02',assay='')
    a['assay_identity_key']='LEGACY_ASSAY_FP:ABC'
    b['assay_identity_key']='LEGACY_ASSAY_FP:ABC'
    first,audit=reconstruct_first_availability([a,b],manifest())
    assert len(first)==1
    assert int(audit.iloc[0].n_release_occurrences)==2


def test_empty_earliest_release_does_not_move_left_censor_boundary_forward():
    # A zero-row but successfully processed earliest release still fixes the archive boundary.
    empty=row('CHEMBL01').iloc[0:0].copy()
    later=row('CHEMBL02',assay='A2')
    first,_=reconstruct_first_availability([empty,later],manifest())
    assert len(first)==1
    r=first.iloc[0]
    assert pd.Timestamp(r.archive_left_censor_release_date)==pd.Timestamp('2009-10-15')
    assert r.archive_left_censor_release_label=='CHEMBL01'
    assert not bool(r.first_availability_left_censored)
    assert not bool(r.pair_first_left_censored)


def test_later_added_chembl_document_and_assay_ids_do_not_create_new_identity():
    from ad_f0.first_availability import canonicalize_cross_release_identities
    base=dict(
        full_inchikey='AAAA-BBBB-C',standard_type='IC50',standard_relation='=',standard_value_molar=1e-6,
        target_accession='P56817',assay_description='same biochemical assay',assay_type='B',
        document_year=2009,document_journal='J MED CHEM',document_volume='52',document_issue='1',
        document_first_page='100',document_title='A stable historical paper',document_doi='10.1/STABLE',
        document_pubmed_id='12345',primary_record_eligible=True,
    )
    early=pd.DataFrame([{**base,'release_label':'CHEMBL07','assay_chembl_id':None,'document_chembl_id':None}])
    later=pd.DataFrame([{**base,'release_label':'CHEMBL08','assay_chembl_id':'CHEMBL_A8','document_chembl_id':'CHEMBL_D8'}])
    frames,doc_audit,assay_audit=canonicalize_cross_release_identities([early,later])
    assert frames[0].iloc[0].document_identity_key==frames[1].iloc[0].document_identity_key
    assert frames[0].iloc[0].assay_identity_key==frames[1].iloc[0].assay_identity_key
    assert not doc_audit.empty and not assay_audit.empty


def test_conflicting_weak_bibliographic_alias_is_quarantined_by_external_ids():
    from ad_f0.first_availability import canonicalize_cross_release_identities
    # Same weak bibliographic fingerprint but incompatible DOI identities must split;
    # the weak bridge is quarantined rather than allowed to override external IDs.
    common=dict(full_inchikey='AAAA-BBBB-C',standard_type='IC50',standard_relation='=',standard_value_molar=1e-6,
        target_accession='P56817',assay_description='assay',assay_type='B',document_year=2009,
        document_journal='J',document_volume='1',document_issue='1',document_first_page='10',document_title='TITLE',
        primary_record_eligible=True,assay_chembl_id='CHEMBL_A')
    a=pd.DataFrame([{**common,'release_label':'CHEMBL07','document_doi':'10.1/A','document_chembl_id':'CHEMBL_D1'}])
    b=pd.DataFrame([{**common,'release_label':'CHEMBL08','document_doi':'10.1/B','document_chembl_id':'CHEMBL_D2'}])
    frames,da,_=canonicalize_cross_release_identities([a,b],release_local_only_max_fraction=1.0)
    assert frames[0].iloc[0].document_identity_key != frames[1].iloc[0].document_identity_key
    assert int(da.n_quarantined_weak_aliases.sum()) > 0
