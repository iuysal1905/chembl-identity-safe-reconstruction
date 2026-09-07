"""Regression tests for two identity-reconciliation defects in F0 v1.2.

Both FAIL against v1.2 and should pass after the fixes described in review.

  1. Internal ChEMBL IDs are documented as "just an alias", but the fail-closed
     check treats >1 CHEMBL_DOC / CHEMBL_ASSAY alias in a component as fatal.
     Document renumbering/merging is exactly the case this subsystem exists to
     absorb, so it must be logged, not fatal. Only conflicting EXTERNAL ids
     (DOI, PMID, source-assay) justify failing closed.

  2. The weak bibliographic alias accepts journal+year+volume as
     "sufficiently specific". Two distinct papers sharing those fields with no
     title, no first page and no strong id collapse into one document identity,
     which cascades into merged assay and activity identities.
"""
import pandas as pd
import pytest

from ad_f0.first_availability import (
    canonicalize_cross_release_identities,
    HistoricalIdentityConflictError,
)


def _row(doc_id, doc_num, title="Series", page="1000", doi=None, pmid=""):
    return dict(
        document_journal="J Med Chem",
        document_year=2005,
        document_volume=str(doc_num),
        document_issue="1",
        document_first_page=page,
        document_title=f"{title} {doc_num}" if title else "",
        document_doi=doi if doi is not None else f"10.1021/jm{doc_num:06d}",
        document_pubmed_id=pmid,
        document_chembl_id=doc_id,
        assay_description=f"Inhibition assay {doc_num}",
        assay_type="B",
        target_accession="P56817",
        assay_chembl_id=f"CHEMBL_A{doc_num}",
        assay_source_id="1",
        assay_source_assay_id=f"S{doc_num}",
    )


def test_internal_chembl_id_renumbering_is_not_fatal():
    """Same paper, same DOI, different ChEMBL document id in a later release.

    This is ChEMBL curation renumbering/merging. It must reconcile to one
    canonical identity and be recorded in the audit, not abort the run.
    """
    early = pd.DataFrame([_row("CHEMBL_D1", 1)])
    late = pd.DataFrame([_row("CHEMBL_D999", 1)])  # renumbered, DOI unchanged

    frames, doc_audit, _ = canonicalize_cross_release_identities([early, late])

    assert frames[0].loc[0, "document_identity_key"] == frames[1].loc[0, "document_identity_key"]
    merged = doc_audit[doc_audit.n_chembl_aliases > 1]
    assert len(merged) == 1, "internal-id merge must be retained in the audit"


def test_conflicting_external_ids_still_fail_closed():
    """Two different DOIs pulled into one component must remain fatal."""
    a = pd.DataFrame([_row("CHEMBL_D1", 1, doi="10.1021/jm000001")])
    b = pd.DataFrame([_row("CHEMBL_D1", 1, doi="10.1021/jm000002")])
    with pytest.raises(HistoricalIdentityConflictError):
        canonicalize_cross_release_identities([a, b])


def test_weak_alias_does_not_merge_distinct_papers():
    """Journal + year + volume alone is not specific enough.

    Simulates an early release where titles, page numbers and strong ids are
    absent. Two genuinely different documents must not collapse.
    """
    r1 = _row("", 1, title="", page="", doi="")
    r2 = _row("", 1, title="", page="", doi="")
    r2["assay_description"] = "Inhibition assay 2"
    r2["assay_chembl_id"] = ""
    r2["assay_source_assay_id"] = "S2"
    r1["assay_chembl_id"] = ""
    r1["assay_source_assay_id"] = "S1"
    frame = pd.DataFrame([r1, r2])

    frames, _, _ = canonicalize_cross_release_identities([frame])
    assert frames[0]["document_identity_key"].nunique() == 2
