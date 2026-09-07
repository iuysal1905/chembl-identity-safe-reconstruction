"""Regression test for cross-release identity UNDER-merge in F0 v1.3.

v1.3 correctly guards against over-merging (weak-only internal-ID multiplicity,
>1% per schema family fails closed). There is no symmetric guard against
under-merging.

A document that lacks `document_year` (or lacks both title and first page) and
carries no DOI / PMID / ChEMBL document id falls back to
`LOCAL_DOC_ID:{release}:{native_document_id}`, which is release-scoped and can
never bridge two releases. Because the assay semantic alias embeds
`document_identity_key`, the assay fragments too -- even when its description is
present and identical across releases.

One physical measurement then becomes N activity identities with N distinct
first-availability dates. Endpoint pair identity is unaffected, so primary-P
*pair* counts stay correct, but R/C `exact_observations` and every layer's
`unique_documents` inflate by roughly N. The bias direction is toward a false
F0 PASS, and it is concentrated in `pre9_early` -- Backtest A.

This test FAILS against v1.3.
"""
import pandas as pd

from ad_f0.first_availability import (
    canonicalize_cross_release_identities,
    reconstruct_first_availability,
)

MANIFEST = pd.DataFrame({
    "release_label": ["CHEMBL02", "CHEMBL05", "CHEMBL09"],
    "exact_release_date": ["2009-12-01", "2010-07-01", "2011-02-01"],
})


def _release(label, year):
    """The same physical AChE IC50, re-published in three consecutive archives.

    No DOI, no PMID, no ChEMBL document id, no ChEMBL assay id, no source-assay
    id -- i.e. the sparse-metadata situation typical of the earliest archives.
    """
    return pd.DataFrame([dict(
        release_label=label, schema_generation="pre9_early",
        document_journal="J Med Chem", document_year=year,
        document_volume="40", document_issue="", document_first_page="1200",
        document_title="", document_doi="", document_pubmed_id="",
        document_chembl_id="", native_document_id=55,
        assay_description="Inhibition of AChE", assay_type="B",
        target_accession="P22303", assay_chembl_id="",
        assay_source_id="", assay_source_assay_id="", native_assay_id=101,
        full_inchikey="AAAAAAAAAAAAAA-BBBBBBBBBB-C", standard_type="IC50",
        standard_relation="=", standard_value_molar=1e-7,
        primary_record_eligible=True, quality_eligible=True,
        structure_eligible=True, unit_convertible=True,
    )])


def _reconstruct(year):
    frames, _, assay_audit = canonicalize_cross_release_identities(
        [_release(label, year) for label in MANIFEST.release_label])
    first, _ = reconstruct_first_availability(frames, MANIFEST)
    return first, assay_audit


def test_sparse_document_metadata_does_not_fragment_identity():
    """Control: with `document_year` present the three archives reconcile to one identity."""
    first, _ = _reconstruct("1997")
    assert first.activity_key.nunique() == 1
    assert first.document_identity_key.nunique() == 1


def test_unbridgeable_document_must_not_inflate_row_and_document_counts():
    """Same measurement, `document_year` missing: v1.3 yields 3 of everything."""
    first, _ = _reconstruct("")
    assert first.pair_key.nunique() == 1, "pair identity is expected to hold"
    assert first.activity_key.nunique() == 1, (
        "one physical measurement must not become several activity identities; "
        "R/C exact_observations inflate directly from this"
    )
    assert first.document_identity_key.nunique() == 1, (
        "document gates count document_identity_key, so fragmentation inflates them"
    )


def test_release_local_only_components_are_audited():
    """An under-merge rate metric must exist, symmetric to the over-merge policy."""
    _, assay_audit = _reconstruct("")
    expected = {"n_release_local_only_components", "release_local_only_fraction"}
    assert expected & set(assay_audit.columns), (
        "no audit column reports components whose alias set is release-scoped only"
    )
