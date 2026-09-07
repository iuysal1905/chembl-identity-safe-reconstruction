#!/usr/bin/env python3
from __future__ import annotations

"""Reproduce the main JCIM identity/F0 results from extracted ChEMBL releases.

This script expects the release extraction stage to have completed already.
It intentionally recomputes identity reconciliation and the controlled
release-local comparator from those extracted release tables.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _root(path: str | None) -> Path:
    root = Path(path).expanduser().resolve() if path else Path.cwd().resolve()
    if root.name.lower() == "scripts":
        root = root.parent
    if not (root / "src" / "ad_f0").exists():
        raise FileNotFoundError(f"Not a repository root: {root}")
    return root


PAPER_EXPECTED = {
    "document_identities": 29385,
    "assay_identities": 124410,
    "malformed_numeric_doi_release_occurrences": 41,
    "malformed_numeric_doi_distinct_values": 21,
    "shared_doi_collision_aliases": 3,
    "shared_doi_collision_release_groups": 28,
    "document_temporal_recurations": 8,
    "document_temporal_recurations_pmid": 7,
    "document_temporal_recurations_doi": 1,
    "source_policy_temporal_rename_families": 921,
    "source_policy_partition_quarantine_families": 198,
    "source_policy_ambiguous_temporal_families": 19,
    "source_policy_reconciled_raw_aliases": 1842,
    "source_policy_quarantined_aliases": 236,
    "primary_activities": 884186,
    "primary_pairs": 723433,
    "primary_left_censored": 108337,
    "release_local_activities": 1018565,
    "release_local_inflation": 134379,
    "fragmented_safe_activities": 134432,
    "overmerged_release_local_identities": 129,
    "mechanism_ordinary": 67046,
    "mechanism_temporal_rename": 49079,
    "mechanism_partition": 18307,
    "nested_weak_alias": 656,
    "nested_ambiguous_temporal": 784,
    "relaxed_activities": 1199578,
    "relaxed_pairs": 963770,
    "all_direct_activities": 982444,
    "all_direct_pairs": 798829,
}

FIRST_AVAILABILITY_COLUMNS = [
    "release_label", "schema_generation", "native_document_id", "document_chembl_id",
    "document_year", "document_title", "document_doi", "document_pubmed_id",
    "document_journal", "document_volume", "document_issue", "document_first_page",
    "native_assay_id", "assay_chembl_id", "assay_description", "assay_source_id",
    "assay_source_assay_id", "assay_type", "target_accession", "assay_identity_key",
    "molecule_chembl_id", "molecule_pref_name", "canonical_isomeric_smiles",
    "full_inchikey", "murcko_scaffold_smiles", "standard_type", "standard_relation",
    "standard_value_molar", "confidence_score", "target_relationship_type",
    "historical_confidence_equivalence", "primary_record_eligible",
    "relaxed_confidence_B_record_eligible", "primary_direct_all_assay_eligible",
    "broad_prior_knowledge_eligible", "unit_convertible", "relation_eligible",
    "potential_duplicate_flagged", "data_validity_comment", "data_validity_eligible",
    "structure_eligible",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--validate-paper-locks", action="store_true")
    args = parser.parse_args()
    root = _root(args.root)
    sys.path.insert(0, str(root / "src"))

    from ad_f0.io import load_config
    from ad_f0.first_availability import (
        canonicalize_cross_release_identities,
        reconstruct_first_availability,
        _norm_identity_text,
        _norm_doi,
    )
    from ad_f0.paper_analysis import (
        PaperAnalysisError,
        attribute_mechanisms,
        build_release_local_bridge,
        comparator_analysis,
        derive_document_events,
        release_local_first_from_bridge,
        source_policy_from_final_assay_audit,
        source_policy_summary,
    )

    cfg = load_config(root)
    manifest_path = root / "config" / "paper_release_manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype={"release": str})
    manifest["exact_release_date"] = pd.to_datetime(manifest["exact_release_date"])
    manifest = manifest.sort_values("exact_release_date", kind="mergesort").reset_index(drop=True)

    extracted = root / cfg["paths"]["extracted_dir"]
    frames: list[pd.DataFrame] = []
    for release in manifest["release_label"].astype(str):
        path = extracted / f"{release}_activities.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing extracted release: {path}. Run notebooks 00-03 first."
            )
        available = set(pq.ParquetFile(path).schema.names)
        cols = [c for c in FIRST_AVAILABILITY_COLUMNS if c in available]
        frames.append(pd.read_parquet(path, columns=cols))

    identity_cfg = cfg["identity_reconciliation"]
    frames, document_audit, assay_audit = canonicalize_cross_release_identities(
        frames,
        internal_id_weak_only_max_fraction=float(
            identity_cfg["internal_id_weak_only_multiplicity_max_fraction"]
        ),
        release_local_only_max_fraction=float(
            identity_cfg["release_local_only_max_fraction"]
        ),
        release_local_policy_min_releases=int(
            identity_cfg["release_local_policy_min_releases_per_family"]
        ),
        copy_frames=False,
    )

    sigfigs = int(cfg["release_matching"]["value_round_sigfigs"])
    primary, _ = reconstruct_first_availability(
        frames, manifest, sigfigs=sigfigs, eligibility_col="primary_record_eligible"
    )
    relaxed, _ = reconstruct_first_availability(
        frames,
        manifest,
        sigfigs=sigfigs,
        eligibility_col="relaxed_confidence_B_record_eligible",
    )
    all_direct, _ = reconstruct_first_availability(
        frames,
        manifest,
        sigfigs=sigfigs,
        eligibility_col="primary_direct_all_assay_eligible",
    )

    bridge = build_release_local_bridge(frames, manifest, sigfigs)
    local_first, local_release_audit = release_local_first_from_bridge(bridge, manifest)
    displacement, overmerge, comparator_summary = comparator_analysis(
        primary, local_first, bridge, manifest
    )
    mechanisms = attribute_mechanisms(displacement, assay_audit)

    document_events, document_summary = derive_document_events(
        frames, manifest, _norm_identity_text, _norm_doi
    )
    source_policy = source_policy_from_final_assay_audit(assay_audit)
    source_summary = source_policy_summary(source_policy)

    out = root / "results" / "paper"
    out.mkdir(parents=True, exist_ok=True)
    document_audit.to_csv(out / "document_identity_audit.csv", index=False)
    assay_audit.to_csv(out / "assay_identity_audit.csv", index=False)
    document_events.to_csv(out / "document_identity_events.csv", index=False)
    source_policy.to_csv(out / "assay_source_policy.csv", index=False)
    comparator_summary.to_csv(out / "release_local_comparator_summary.csv", index=False)
    local_release_audit.to_csv(out / "release_local_F0_by_release.csv", index=False)
    displacement.to_parquet(out / "F0_displacement.parquet", index=False)
    overmerge.to_csv(out / "release_local_overmerge.csv", index=False)
    mechanisms.to_parquet(out / "shifted_activity_mechanisms.parquet", index=False)

    delay = pd.to_numeric(mechanisms["max_delay_releases"], errors="raise")
    mechanism_counts = mechanisms["mechanism"].value_counts().to_dict()
    observed = {
        "document_identities": len(document_audit),
        "assay_identities": len(assay_audit),
        **document_summary,
        **source_summary,
        "primary_activities": len(primary),
        "primary_pairs": int(primary["pair_key"].nunique()),
        "primary_left_censored": int(primary["first_availability_left_censored"].sum()),
        "release_local_activities": len(local_first),
        "release_local_inflation": len(local_first) - len(primary),
        "fragmented_safe_activities": int(
            comparator_summary.iloc[0]["fragmented_safe_activities"]
        ),
        "overmerged_release_local_identities": len(overmerge),
        "mechanism_ordinary": int(
            mechanism_counts.get("ORDINARY_DETERMINISTIC_ALIAS_BRIDGING", 0)
        ),
        "mechanism_temporal_rename": int(
            mechanism_counts.get("STRICT_TEMPORAL_SOURCE_ASSAY_RENAME", 0)
        ),
        "mechanism_partition": int(
            mechanism_counts.get("PARTITION_NONINJECTIVE_PROTECTION", 0)
        ),
        "nested_weak_alias": int(mechanisms["weak_alias_quarantine_safeguard"].sum()),
        "nested_ambiguous_temporal": int(
            mechanisms["ambiguous_temporal_safeguard"].sum()
        ),
        "relaxed_activities": len(relaxed),
        "relaxed_pairs": int(relaxed["pair_key"].nunique()),
        "all_direct_activities": len(all_direct),
        "all_direct_pairs": int(all_direct["pair_key"].nunique()),
        "delay_min": int(delay.min()),
        "delay_q25": float(delay.quantile(0.25)),
        "delay_median": float(delay.median()),
        "delay_mean": float(delay.mean()),
        "delay_q75": float(delay.quantile(0.75)),
        "delay_q90": float(delay.quantile(0.90)),
        "delay_q95": float(delay.quantile(0.95)),
        "delay_q99": float(delay.quantile(0.99)),
        "delay_max": int(delay.max()),
        "delay_ge_3_pct": 100.0 * float(delay.ge(3).mean()),
        "delay_ge_5_pct": 100.0 * float(delay.ge(5).mean()),
        "delay_ge_7_pct": 100.0 * float(delay.ge(7).mean()),
        "delay_ge_10_pct": 100.0 * float(delay.ge(10).mean()),
    }

    validation_rows = []
    for key, expected in PAPER_EXPECTED.items():
        if key not in observed:
            continue
        actual = observed[key]
        passed = actual == expected
        validation_rows.append(
            {"check": key, "observed": actual, "expected": expected, "passed": passed}
        )
        if args.validate_paper_locks and not passed:
            raise PaperAnalysisError(
                f"Paper lock failed for {key}: observed={actual}, expected={expected}"
            )

    # Continuous-valued displacement locks are checked separately.
    continuous = {
        "delay_mean": (5.295391, 5e-6),
        "delay_ge_3_pct": (91.12786, 5e-5),
        "delay_ge_5_pct": (51.23111, 5e-5),
        "delay_ge_7_pct": (26.68859, 5e-5),
        "delay_ge_10_pct": (5.001785, 5e-5),
    }
    for key, (expected, tolerance) in continuous.items():
        actual = float(observed[key])
        passed = abs(actual - expected) <= tolerance
        validation_rows.append(
            {
                "check": key,
                "observed": actual,
                "expected": expected,
                "passed": passed,
                "tolerance": tolerance,
            }
        )
        if args.validate_paper_locks and not passed:
            raise PaperAnalysisError(
                f"Paper lock failed for {key}: observed={actual}, expected={expected}"
            )

    pd.DataFrame(validation_rows).to_csv(out / "paper_validation.csv", index=False)
    (out / "paper_result_summary.json").write_text(
        json.dumps(observed, indent=2, default=str), encoding="utf-8"
    )

    print("Reconstruction complete.")
    print(f"  identity-safe activities: {len(primary):,}")
    print(f"  release-local activities: {len(local_first):,}")
    print(f"  displaced activities: {len(mechanisms):,}")
    print("  mechanisms:", mechanism_counts)
    print("  outputs:", out)
    if args.validate_paper_locks:
        print("All publication locks passed.")


if __name__ == "__main__":
    main()
