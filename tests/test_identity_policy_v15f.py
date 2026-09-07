import pandas as pd
import pytest

from ad_f0.first_availability import (
    canonicalize_cross_release_identities,
    HistoricalIdentityConflictError,
    weak_alias_document_rate_by_schema,
)


def base_row(**kw):
    d=dict(
        document_journal='J Med Chem',document_year=2005,document_volume='48',document_issue='1',
        document_first_page='100',document_title='A paper',document_doi='',document_pubmed_id='',
        document_chembl_id='',native_document_id='',assay_description='enzyme inhibition',assay_type='B',
        target_accession='P56817',assay_chembl_id='',assay_source_id='',assay_source_assay_id='',
        native_assay_id='1',schema_generation='pre9_early',release_label='CHEMBL07'
    )
    d.update(kw)
    return d


def test_weak_only_internal_document_renumbering_above_preregistered_fraction_fails():
    # Same weak bibliographic component, no DOI/PMID support, two internal ChEMBL doc IDs.
    a=pd.DataFrame([base_row(document_chembl_id='CHEMBL_D1')])
    b=pd.DataFrame([base_row(document_chembl_id='CHEMBL_D2',release_label='CHEMBL08')])
    with pytest.raises(HistoricalIdentityConflictError):
        canonicalize_cross_release_identities([a,b], internal_id_weak_only_max_fraction=0.01)


def test_assay_type_drift_is_audited_when_strong_internal_alias_bridges_recuration():
    a=pd.DataFrame([base_row(document_doi='10.1/X',assay_chembl_id='CHEMBL_A1',assay_type='B')])
    b=pd.DataFrame([base_row(document_doi='10.1/X',assay_chembl_id='CHEMBL_A1',assay_type='F',release_label='CHEMBL08')])
    _,_,aa=canonicalize_cross_release_identities([a,b])
    row=aa[aa.assay_type_drift].iloc[0]
    assert row.n_assay_types==2
    assert set(row.assay_types.split('|'))=={'B','F'}


def test_weak_alias_rate_is_reported_by_schema_family():
    a=pd.DataFrame([base_row(document_chembl_id='')])
    _,da,_=canonicalize_cross_release_identities([a])
    out=weak_alias_document_rate_by_schema(da)
    assert len(out)==1
    assert out.iloc[0].schema_generation=='PRE9_EARLY'
    assert out.iloc[0].weak_only_fraction==1.0


def test_doi_url_and_bare_doi_reconcile_as_same_external_identity():
    a=pd.DataFrame([base_row(document_doi='https://doi.org/10.1000/Test',document_chembl_id='CHEMBL_D1')])
    b=pd.DataFrame([base_row(document_doi='10.1000/test',document_chembl_id='CHEMBL_D2',release_label='CHEMBL08')])
    frames,da,_=canonicalize_cross_release_identities([a,b])
    assert frames[0].iloc[0].document_identity_key==frames[1].iloc[0].document_identity_key
    assert int(da.iloc[0].n_doi_aliases)==1

def test_numeric_document_doi_is_not_external_doi_authority():
    a = pd.DataFrame([
        base_row(
            document_doi="24044434",
            document_chembl_id="CHEMBL_D1",
        )
    ])

    _, da, _ = canonicalize_cross_release_identities([a])

    assert int(da.iloc[0].n_doi_aliases) == 0
    assert "DOI:24044434" not in da.iloc[0].aliases


def test_valid_doi_still_remains_external_authority():
    a = pd.DataFrame([
        base_row(
            document_doi="10.1021/jm4011753",
            document_chembl_id="CHEMBL_D1",
        )
    ])

    _, da, _ = canonicalize_cross_release_identities([a])

    assert int(da.iloc[0].n_doi_aliases) == 1
    assert "DOI:10.1021/JM4011753" in da.iloc[0].aliases

def test_weak_title_alias_survives_nonessential_bibliographic_enrichment():
    a=pd.DataFrame([base_row(document_doi='',document_chembl_id='',document_journal='',document_volume='',document_issue='')])
    b=pd.DataFrame([base_row(document_doi='',document_chembl_id='',document_journal='J Med Chem',document_volume='48',document_issue='2',release_label='CHEMBL08')])
    frames,_,_=canonicalize_cross_release_identities([a,b])
    assert frames[0].iloc[0].document_identity_key==frames[1].iloc[0].document_identity_key



def test_shared_contaminated_doi_collision_is_quarantined_and_split():
    """A shared DOI must not merge two deterministically distinct documents."""

    a = base_row(
        release_label='CHEMBL23',
        schema_generation='post15',
        document_doi='10.1021/COLLIDE',
        document_pubmed_id='111',
        document_chembl_id='CHEMBL_D1',
        document_first_page='100',
        document_title='Paper alpha',
    )

    b = base_row(
        release_label='CHEMBL23',
        schema_generation='post15',
        document_doi='10.1021/COLLIDE',
        document_pubmed_id='222',
        document_chembl_id='CHEMBL_D2',
        document_first_page='200',
        document_title='Paper beta',
    )

    frames, da, _ = canonicalize_cross_release_identities(
        [pd.DataFrame([a, b])],
        release_local_only_max_fraction=1.0,
    )

    assert (
        frames[0].iloc[0].document_identity_key
        != frames[0].iloc[1].document_identity_key
    )

    assert int(
        da['n_quarantined_external_aliases'].sum()
    ) == 2

    assert all(
        'DOI:10.1021/COLLIDE'
        in str(v)
        for v in da[
            'quarantined_external_aliases'
        ]
    )


def test_ambiguous_shared_doi_without_bibliographic_separation_still_fails():
    """The collision exception must remain narrow and fail closed otherwise."""

    a = base_row(
        release_label='CHEMBL23',
        schema_generation='post15',
        document_doi='10.1021/AMBIGUOUS',
        document_pubmed_id='111',
        document_chembl_id='CHEMBL_D1',
        document_first_page='100',
        document_title='Same paper',
    )

    b = base_row(
        release_label='CHEMBL23',
        schema_generation='post15',
        document_doi='10.1021/AMBIGUOUS',
        document_pubmed_id='222',
        document_chembl_id='CHEMBL_D2',
        document_first_page='100',
        document_title='Same paper',
    )

    with pytest.raises(
        HistoricalIdentityConflictError
    ):
        canonicalize_cross_release_identities(
            [pd.DataFrame([a, b])],
            release_local_only_max_fraction=1.0,
        )




def test_temporal_pmid_recuration_one_way_passes():
    """Replicated one-way PMID recuration may retain one document lineage."""

    def make_row(release, pmid):
        return base_row(
            release_label=release,
            schema_generation='pre15',
            document_doi='10.1000/STABLE-PMID',
            document_pubmed_id=pmid,
            document_chembl_id='CHEMBL_D1',
            document_year='2004',
            document_journal='J TEST',
            document_volume='1',
            document_issue='1',
            document_first_page='10',
            document_title='Stable publication',
        )

    input_frames = [
        pd.DataFrame([make_row('CHEMBL01', '111')]),
        pd.DataFrame([make_row('CHEMBL02', '111')]),
        pd.DataFrame([make_row('CHEMBL10', '222')]),
        pd.DataFrame([make_row('CHEMBL11', '222')]),
    ]

    frames, da, _ = canonicalize_cross_release_identities(
        input_frames,
        release_local_only_max_fraction=1.0,
    )

    keys = [
        f.iloc[0].document_identity_key
        for f in frames
    ]

    assert len(set(keys)) == 1

    temporal = da[
        da['temporal_recuration_pass']
    ]

    assert len(temporal) == 1

    row = temporal.iloc[0]

    assert row.temporal_external_type == 'PMID'
    assert row.temporal_old_external_id == '111'
    assert row.temporal_new_external_id == '222'

    assert int(
        row.temporal_old_state_n_frames
    ) == 2

    assert int(
        row.temporal_new_state_n_frames
    ) == 2

    assert bool(
        row.temporal_globally_exclusive
    )

    assert bool(
        row.temporal_bibliographically_compatible
    )


def test_temporal_doi_recuration_one_way_passes():
    """Replicated one-way DOI recuration may retain one document lineage."""

    def make_row(release, doi):
        return base_row(
            release_label=release,
            schema_generation='post15',
            document_doi=doi,
            document_pubmed_id='999',
            document_chembl_id='CHEMBL_D1',
            document_year='1993',
            document_journal='J TEST',
            document_volume='2',
            document_issue='1',
            document_first_page='20',
            document_title='Stable DOI publication',
        )

    input_frames = [
        pd.DataFrame([
            make_row(
                'CHEMBL09',
                '10.1000/OLD-DOI',
            )
        ]),
        pd.DataFrame([
            make_row(
                'CHEMBL10',
                '10.1000/OLD-DOI',
            )
        ]),
        pd.DataFrame([
            make_row(
                'CHEMBL23',
                '10.1000/NEW-DOI',
            )
        ]),
        pd.DataFrame([
            make_row(
                'CHEMBL24',
                '10.1000/NEW-DOI',
            )
        ]),
    ]

    frames, da, _ = canonicalize_cross_release_identities(
        input_frames,
        release_local_only_max_fraction=1.0,
    )

    keys = [
        f.iloc[0].document_identity_key
        for f in frames
    ]

    assert len(set(keys)) == 1

    temporal = da[
        da['temporal_recuration_pass']
    ]

    assert len(temporal) == 1

    row = temporal.iloc[0]

    assert row.temporal_external_type == 'DOI'

    assert (
        row.temporal_old_external_id
        == '10.1000/OLD-DOI'
    )

    assert (
        row.temporal_new_external_id
        == '10.1000/NEW-DOI'
    )

    assert int(
        row.temporal_old_state_n_frames
    ) == 2

    assert int(
        row.temporal_new_state_n_frames
    ) == 2

def test_temporal_external_ids_simultaneous_still_fail_closed():
    """Two competing external IDs in one release are not temporal recuration."""

    a = base_row(
        release_label='CHEMBL23',
        schema_generation='post15',
        document_doi='10.1000/SAME',
        document_pubmed_id='111',
        document_chembl_id='CHEMBL_D1',
        document_year='2004',
        document_journal='J TEST',
        document_volume='1',
        document_first_page='10',
        document_title='Same publication',
    )

    b = base_row(
        release_label='CHEMBL23',
        schema_generation='post15',
        document_doi='10.1000/SAME',
        document_pubmed_id='222',
        document_chembl_id='CHEMBL_D1',
        document_year='2004',
        document_journal='J TEST',
        document_volume='1',
        document_first_page='10',
        document_title='Same publication',
    )

    with pytest.raises(
        HistoricalIdentityConflictError
    ):
        canonicalize_cross_release_identities(
            [
                pd.DataFrame([a, b]),
            ],
            release_local_only_max_fraction=1.0,
        )


def test_temporal_external_id_reversion_still_fails_closed():
    """A -> B -> A is not a one-way historical recuration."""

    rows = []

    for release, pmid in [
        ('CHEMBL01', '111'),
        ('CHEMBL10', '222'),
        ('CHEMBL20', '111'),
    ]:
        rows.append(
            base_row(
                release_label=release,
                schema_generation='pre15',
                document_doi='10.1000/STABLE-REV',
                document_pubmed_id=pmid,
                document_chembl_id='CHEMBL_D1',
                document_year='2004',
                document_journal='J TEST',
                document_volume='1',
                document_first_page='10',
                document_title='Stable publication',
            )
        )

    with pytest.raises(
        HistoricalIdentityConflictError
    ):
        canonicalize_cross_release_identities(
            [
                pd.DataFrame([rows[0]]),
                pd.DataFrame([rows[1]]),
                pd.DataFrame([rows[2]]),
            ],
            release_local_only_max_fraction=1.0,
        )


def test_temporal_recuration_with_bibliographic_contradiction_fails_closed():
    """One-way chronology alone cannot override incompatible bibliography."""

    early = base_row(
        release_label='CHEMBL01',
        schema_generation='pre15',
        document_doi='10.1000/STABLE-BIB',
        document_pubmed_id='111',
        document_chembl_id='CHEMBL_D1',
        document_year='2004',
        document_journal='J TEST',
        document_volume='1',
        document_first_page='10',
        document_title='Publication alpha',
    )

    late = base_row(
        release_label='CHEMBL10',
        schema_generation='pre15',
        document_doi='10.1000/STABLE-BIB',
        document_pubmed_id='222',
        document_chembl_id='CHEMBL_D1',
        document_year='2005',
        document_journal='J TEST',
        document_volume='1',
        document_first_page='99',
        document_title='Publication beta',
    )

    with pytest.raises(
        HistoricalIdentityConflictError
    ):
        canonicalize_cross_release_identities(
            [
                pd.DataFrame([early]),
                pd.DataFrame([late]),
            ],
            release_local_only_max_fraction=1.0,
        )
