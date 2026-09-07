import pandas as pd

from ad_f0.paper_analysis import (
    attribute_mechanisms,
    build_release_local_bridge,
    derive_document_events,
    release_local_first_from_bridge,
    source_policy_from_final_assay_audit,
    source_policy_summary,
)


def _base(release, assay_identity="SAFE:A", source_assay="S1"):
    return {
        "release_label": release,
        "primary_record_eligible": True,
        "assay_identity_key": assay_identity,
        "assay_source_id": "SRC",
        "assay_source_assay_id": source_assay,
        "assay_chembl_id": "",
        "native_assay_id": "1",
        "document_identity_key": "DOC:A",
        "assay_description": "binding assay",
        "assay_type": "B",
        "full_inchikey": "AAAAAAAAAAAAAA-BBBBBBBBBB-C",
        "standard_type": "IC50",
        "standard_relation": "=",
        "standard_value_molar": 1e-6,
        "target_accession": "P12345",
    }


def test_release_local_comparator_does_not_force_release_prefix():
    frames = [
        pd.DataFrame([_base("R1")]),
        pd.DataFrame([_base("R2")]),
    ]
    manifest = pd.DataFrame(
        {"release_label": ["R1", "R2"], "exact_release_date": ["2020-01-01", "2020-02-01"]}
    )
    bridge = build_release_local_bridge(frames, manifest, 12)
    assert bridge["release_local_activity_key"].nunique() == 1
    local_first, _ = release_local_first_from_bridge(bridge, manifest)
    assert len(local_first) == 1
    assert local_first.iloc[0]["first_available_release"] == "R1"


def test_source_policy_mixed_tags_are_resolved_per_family():
    audit = pd.DataFrame(
        {
            "canonical_key": ["AT", "AP1", "AP2"],
            "assay_source_policy_types": [
                "TEMPORAL_RENAME",
                "AMBIGUOUS_TEMPORAL_RENAME|PARTITION_DESCENDANT|PARTITION_QUARANTINE",
                "PARTITION_DESCENDANT|PARTITION_QUARANTINE",
            ],
            "assay_source_policy_families": ["37:T_1", "37:P_1|37:A_1", "37:P_1"],
            "renamed_external_aliases": [
                "SRCASSAY:37:T_1=>SRCASSAY:CAN|SRCASSAY:37:T_1_POL_9=>SRCASSAY:CAN",
                "",
                "",
            ],
            "quarantined_external_aliases": [
                "",
                "SRCASSAY:37:P_1|SRCASSAY:37:A_1|SRCASSAY:37:A_1_POL_9",
                "SRCASSAY:37:P_1",
            ],
        }
    )
    policy = source_policy_from_final_assay_audit(audit)
    got = policy.set_index("family_id")["policy"].to_dict()
    assert got == {
        "37:A_1": "AMBIGUOUS_TEMPORAL_RENAME",
        "37:P_1": "PARTITION_QUARANTINE",
        "37:T_1": "TEMPORAL_RENAME",
    }
    summary = source_policy_summary(policy)
    assert summary["source_policy_temporal_rename_families"] == 1
    assert summary["source_policy_partition_quarantine_families"] == 1
    assert summary["source_policy_ambiguous_temporal_families"] == 1
    assert summary["source_policy_reconciled_raw_aliases"] == 2
    assert summary["source_policy_quarantined_aliases"] == 3


def test_shared_doi_collision_counts_only_simultaneous_release_groups():
    manifest = pd.DataFrame(
        {"release_label": ["R1", "R2"], "exact_release_date": ["2020-01-01", "2020-02-01"]}
    )
    frames = [
        pd.DataFrame(
            {
                "release_label": ["R1", "R1"],
                "document_identity_key": ["D1", "D2"],
                "document_doi": ["10.1/X", "10.1/X"],
                "document_pubmed_id": ["", ""],
                "document_chembl_id": ["C1", "C2"],
            }
        ),
        pd.DataFrame(
            {
                "release_label": ["R2"],
                "document_identity_key": ["D1"],
                "document_doi": ["10.1/X"],
                "document_pubmed_id": [""],
                "document_chembl_id": ["C1"],
            }
        ),
    ]
    _, summary = derive_document_events(
        frames,
        manifest,
        norm_text=lambda x: "" if pd.isna(x) else str(x).strip(),
        norm_doi=lambda x: "" if pd.isna(x) else str(x).strip().upper(),
    )
    assert summary["shared_doi_collision_aliases"] == 1
    assert summary["shared_doi_collision_release_groups"] == 1


def test_mechanism_priority_uses_temporal_then_partition():
    displacement = pd.DataFrame(
        {
            "safe_activity_key": ["a", "b", "c"],
            "assay_identity_key": ["A1", "A2", "A3"],
            "has_later_branch": [True, True, True],
            "max_delay_releases": [3, 2, 1],
        }
    )
    assay_audit = pd.DataFrame(
        {
            "canonical_key": ["A1", "A2", "A3"],
            "assay_source_policy_types": [
                "TEMPORAL_RENAME|PARTITION_QUARANTINE",
                "PARTITION_QUARANTINE",
                "",
            ],
            "n_quarantined_weak_aliases": [0, 1, 0],
        }
    )
    out = attribute_mechanisms(displacement, assay_audit)
    assert out["mechanism"].tolist() == [
        "STRICT_TEMPORAL_SOURCE_ASSAY_RENAME",
        "PARTITION_NONINJECTIVE_PROTECTION",
        "ORDINARY_DETERMINISTIC_ALIAS_BRIDGING",
    ]
    assert out["weak_alias_quarantine_safeguard"].tolist() == [False, True, False]
