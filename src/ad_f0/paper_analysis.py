from __future__ import annotations

"""Publication analysis helpers for the historical ChEMBL identity study.

The functions in this module implement the controlled release-local comparator,
identity-event summaries, and mechanism attribution used in the manuscript.
They deliberately fail closed when required provenance fields are absent.
"""

import re
from typing import Callable

import numpy as np
import pandas as pd

from .first_availability import add_activity_key, _assay_local_keys


NUMERIC_LIKE_RE = re.compile(r"^[+-]?\d+(?:\.0+)?$")


class PaperAnalysisError(RuntimeError):
    """Raised when a publication analysis invariant is violated."""


def _release_order(manifest: pd.DataFrame) -> dict[str, int]:
    if "release_label" not in manifest.columns:
        raise PaperAnalysisError("Manifest lacks release_label")
    return {str(r): i for i, r in enumerate(manifest["release_label"].astype(str))}


def _assert_frames_align_manifest(frames: list[pd.DataFrame], manifest: pd.DataFrame) -> None:
    if len(frames) != len(manifest):
        raise PaperAnalysisError(
            f"Frame/manifest length mismatch: {len(frames)} vs {len(manifest)}"
        )
    for i, (frame, expected_rel) in enumerate(
        zip(frames, manifest["release_label"].astype(str))
    ):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"frames[{i}] is not a pandas DataFrame")
        if "release_label" not in frame.columns:
            raise PaperAnalysisError(f"frames[{i}] lacks release_label")
        observed = set(frame["release_label"].dropna().astype(str).unique())
        if observed and observed != {expected_rel}:
            raise PaperAnalysisError(
                f"frames[{i}] release mismatch: observed={sorted(observed)}, "
                f"manifest={expected_rel}"
            )


def build_release_local_bridge(
    frames: list[pd.DataFrame],
    manifest: pd.DataFrame,
    sigfigs: int,
) -> pd.DataFrame:
    """Build the occurrence bridge for the controlled release-local comparator.

    ``_assay_local_keys`` already applies the intended local-key hierarchy:
    source-assay ID -> ChEMBL assay ID -> semantic raw fingerprint ->
    release-scoped native-ID fallback -> release-scoped row fallback.

    Crucially, this function does *not* prepend the release label to every key.
    Unchanged unreconciled assay identifiers may therefore recur across releases;
    what is omitted is canonical cross-release reconciliation.
    """
    _assert_frames_align_manifest(frames, manifest)
    parts: list[pd.DataFrame] = []

    for frame_index, frame in enumerate(frames):
        if "primary_record_eligible" not in frame.columns:
            raise PaperAnalysisError(
                f"frames[{frame_index}] lacks primary_record_eligible"
            )
        eligible = (
            frame["primary_record_eligible"]
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )
        safe = frame.loc[eligible].copy()
        if safe.empty:
            continue

        safe["historical_identity_canonicalized"] = True
        safe_keyed = add_activity_key(safe, sigfigs)

        local_ids = _assay_local_keys(safe, frame_index).astype(str)
        local = safe.copy()
        local["assay_identity_key"] = "RELEASE_LOCAL_ASSAY:" + local_ids
        local["historical_identity_canonicalized"] = True
        local_keyed = add_activity_key(local, sigfigs)

        part = pd.DataFrame(
            {
                "release_label": safe["release_label"].astype(str).to_numpy(copy=False),
                "safe_activity_key": safe_keyed["activity_key"].astype(str).to_numpy(copy=False),
                "release_local_activity_key": local_keyed["activity_key"].astype(str).to_numpy(copy=False),
                "safe_assay_identity_key": safe["assay_identity_key"].astype(str).to_numpy(copy=False),
                "release_local_assay_identity_key": local["assay_identity_key"].astype(str).to_numpy(copy=False),
            }
        ).drop_duplicates()
        parts.append(part)

    if not parts:
        raise PaperAnalysisError("Release-local occurrence bridge is empty")

    return (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )


def release_local_first_from_bridge(
    bridge: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive release-local F0 from the eligible occurrence bridge."""
    required = {
        "release_label",
        "release_local_activity_key",
        "release_local_assay_identity_key",
    }
    missing = required - set(bridge.columns)
    if missing:
        raise PaperAnalysisError(
            "Comparator bridge lacks columns: " + ", ".join(sorted(missing))
        )

    order = _release_order(manifest)
    date_map = {
        str(r.release_label): pd.Timestamp(r.exact_release_date)
        for r in manifest.itertuples(index=False)
    }

    x = bridge[
        [
            "release_label",
            "release_local_activity_key",
            "release_local_assay_identity_key",
        ]
    ].drop_duplicates().copy()
    x["_release_order"] = x["release_label"].astype(str).map(order)
    if x["_release_order"].isna().any():
        bad = sorted(set(x.loc[x["_release_order"].isna(), "release_label"].astype(str)))
        raise PaperAnalysisError(f"Unknown release labels in comparator bridge: {bad[:20]}")

    x = x.sort_values(
        ["release_local_activity_key", "_release_order", "release_label"],
        kind="mergesort",
    )
    first = (
        x.drop_duplicates("release_local_activity_key", keep="first")
        .rename(
            columns={
                "release_local_activity_key": "activity_key",
                "release_local_assay_identity_key": "assay_identity_key",
                "release_label": "first_available_release",
            }
        )
        .reset_index(drop=True)
    )
    first["first_available_date"] = (
        first["first_available_release"].astype(str).map(date_map)
    )
    first["first_availability_left_censored"] = first["_release_order"].eq(0)
    first = first.drop(columns=["_release_order"])

    observed = (
        bridge.groupby("release_label")["release_local_activity_key"]
        .nunique()
        .rename("unique_release_local_identities_observed")
    )
    newly = (
        first.groupby("first_available_release")
        .size()
        .rename("new_release_local_identities_at_F0")
    )
    audit = (
        manifest[["release_label", "exact_release_date"]]
        .copy()
        .set_index("release_label")
        .join(observed, how="left")
        .join(newly, how="left")
        .fillna(
            {
                "unique_release_local_identities_observed": 0,
                "new_release_local_identities_at_F0": 0,
            }
        )
        .reset_index()
    )
    audit[
        [
            "unique_release_local_identities_observed",
            "new_release_local_identities_at_F0",
        ]
    ] = audit[
        [
            "unique_release_local_identities_observed",
            "new_release_local_identities_at_F0",
        ]
    ].astype("int64")
    return first, audit


def comparator_analysis(
    safe_first: pd.DataFrame,
    local_first: pd.DataFrame,
    bridge: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare identity-safe and release-local topology and F0."""
    order = _release_order(manifest)
    if safe_first["activity_key"].duplicated().any():
        raise PaperAnalysisError("safe_first activity_key is not unique")
    if local_first["activity_key"].duplicated().any():
        raise PaperAnalysisError("local_first activity_key is not unique")

    topology = bridge[["safe_activity_key", "release_local_activity_key"]].drop_duplicates()
    safe_meta = safe_first[["activity_key", "first_available_release"]].rename(
        columns={
            "activity_key": "safe_activity_key",
            "first_available_release": "safe_first_available_release",
        }
    )
    local_meta = local_first[["activity_key", "first_available_release"]].rename(
        columns={
            "activity_key": "release_local_activity_key",
            "first_available_release": "release_local_first_available_release",
        }
    )
    merged = (
        topology.merge(safe_meta, on="safe_activity_key", how="left", validate="many_to_one")
        .merge(local_meta, on="release_local_activity_key", how="left", validate="many_to_one")
    )
    if merged["safe_first_available_release"].isna().any():
        raise PaperAnalysisError("Comparator bridge contains rows without identity-safe F0")
    if merged["release_local_first_available_release"].isna().any():
        raise PaperAnalysisError("Comparator bridge contains rows without release-local F0")

    bridged_safe = int(topology["safe_activity_key"].nunique())
    if bridged_safe != len(safe_first):
        raise PaperAnalysisError(
            f"Comparator bridge covers {bridged_safe} safe identities; expected {len(safe_first)}"
        )

    safe_ord = merged["safe_first_available_release"].astype(str).map(order)
    local_ord = merged["release_local_first_available_release"].astype(str).map(order)
    if safe_ord.isna().any() or local_ord.isna().any():
        raise PaperAnalysisError("Comparator contains a release absent from the verified manifest")
    merged["shift_releases"] = local_ord.astype(np.int32) - safe_ord.astype(np.int32)

    displacement = (
        merged.groupby("safe_activity_key", sort=False, observed=True)
        .agg(
            safe_first_available_release=("safe_first_available_release", "first"),
            n_release_local_branches=("release_local_activity_key", "nunique"),
            min_shift_releases=("shift_releases", "min"),
            max_delay_releases=("shift_releases", "max"),
        )
        .reset_index()
    )
    displacement["safe_first_release_ordinal"] = (
        displacement["safe_first_available_release"].astype(str).map(order).astype(np.int32)
    )
    displacement["any_f0_difference"] = (
        (displacement["min_shift_releases"] != 0)
        | (displacement["max_delay_releases"] != 0)
    )
    displacement["has_later_branch"] = displacement["max_delay_releases"] > 0

    keep = [
        c
        for c in [
            "activity_key",
            "assay_identity_key",
            "full_inchikey",
            "target_accession",
            "standard_type",
            "standard_relation",
            "standard_value_molar",
        ]
        if c in safe_first.columns
    ]
    meta = safe_first[keep].drop_duplicates("activity_key").rename(
        columns={"activity_key": "safe_activity_key"}
    )
    displacement = displacement.merge(
        meta, on="safe_activity_key", how="left", validate="one_to_one"
    )

    local_counts = topology.groupby("release_local_activity_key", sort=False)[
        "safe_activity_key"
    ].nunique()
    over_keys = local_counts.index[local_counts.gt(1)]
    if len(over_keys):
        over_rows = topology[topology["release_local_activity_key"].isin(over_keys)]
        overmerge = (
            over_rows.groupby("release_local_activity_key", sort=False)["safe_activity_key"]
            .agg(lambda s: "|".join(sorted(set(map(str, s)))))
            .reset_index(name="safe_activity_keys")
        )
        overmerge["n_safe_activity_keys"] = overmerge["safe_activity_keys"].str.count(r"\|") + 1
        overmerge = overmerge[
            ["release_local_activity_key", "n_safe_activity_keys", "safe_activity_keys"]
        ]
    else:
        overmerge = pd.DataFrame(
            columns=["release_local_activity_key", "n_safe_activity_keys", "safe_activity_keys"]
        )

    fragmented = int((displacement["n_release_local_branches"] > 1).sum())
    summary = pd.DataFrame(
        [
            {
                "identity_safe_activities": int(len(safe_first)),
                "release_local_activities": int(len(local_first)),
                "absolute_inflation": int(len(local_first) - len(safe_first)),
                "inflation_percent_of_safe": 100.0
                * (len(local_first) - len(safe_first))
                / len(safe_first),
                "fragmented_safe_activities": fragmented,
                "fragmented_percent_of_safe": 100.0 * fragmented / len(safe_first),
                "safe_activities_with_any_f0_difference": int(
                    displacement["any_f0_difference"].sum()
                ),
                "safe_activities_with_later_branch": int(
                    displacement["has_later_branch"].sum()
                ),
                "overmerged_release_local_identities": int(len(overmerge)),
                "overmerged_percent_of_release_local": 100.0
                * len(overmerge)
                / len(local_first),
            }
        ]
    )
    return displacement, overmerge, summary


def _split_pipe_list(value: object) -> list[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    text = str(value).strip()
    return [x.strip() for x in text.split("|") if x.strip()] if text else []


def _source_alias_belongs_to_family(alias_token: str, family_id: str) -> bool:
    raw = str(alias_token).split("=>", 1)[0]
    prefix = "SRCASSAY:" + str(family_id)
    return raw == prefix or raw.startswith(prefix + "_POL_")


def source_policy_from_final_assay_audit(assay_audit: pd.DataFrame) -> pd.DataFrame:
    """Recover family-level source policy from the final canonical assay audit."""
    required = {
        "canonical_key",
        "assay_source_policy_types",
        "assay_source_policy_families",
        "renamed_external_aliases",
        "quarantined_external_aliases",
    }
    missing = sorted(required - set(assay_audit.columns))
    if missing:
        raise PaperAnalysisError(
            "Final assay audit lacks source-policy provenance columns: "
            + ", ".join(missing)
        )

    evidence: dict[str, dict] = {}
    cols = list(required)
    for row in assay_audit[cols].itertuples(index=False):
        families = _split_pipe_list(row.assay_source_policy_families)
        if not families:
            continue
        tags = set(_split_pipe_list(row.assay_source_policy_types))
        renamed = _split_pipe_list(row.renamed_external_aliases)
        quarantined = _split_pipe_list(row.quarantined_external_aliases)
        canonical_key = str(row.canonical_key)
        for family in families:
            item = evidence.setdefault(
                family,
                {
                    "components": set(),
                    "row_policy_tags": set(),
                    "renamed_aliases": set(),
                    "quarantined_aliases": set(),
                },
            )
            item["components"].add(canonical_key)
            item["row_policy_tags"].update(tags)
            item["renamed_aliases"].update(
                a for a in renamed if _source_alias_belongs_to_family(a, family)
            )
            item["quarantined_aliases"].update(
                a for a in quarantined if _source_alias_belongs_to_family(a, family)
            )

    if not evidence:
        raise PaperAnalysisError("Final assay audit contains no source-policy families")

    principal_tags = {
        "TEMPORAL_RENAME",
        "PARTITION_QUARANTINE",
        "AMBIGUOUS_TEMPORAL_RENAME",
    }
    rows = []
    for family in sorted(evidence):
        item = evidence[family]
        tags = set(item["row_policy_tags"])
        principal = tags & principal_tags
        renamed = sorted(item["renamed_aliases"])
        quarantined = sorted(item["quarantined_aliases"])
        components = sorted(item["components"])

        if renamed:
            if "TEMPORAL_RENAME" not in principal:
                raise PaperAnalysisError(
                    f"Family {family} has renamed aliases but no TEMPORAL_RENAME tag"
                )
            policy = "TEMPORAL_RENAME"
            resolution = "family_specific_renamed_alias_audit"
        elif principal == {"PARTITION_QUARANTINE"}:
            policy = "PARTITION_QUARANTINE"
            resolution = "unambiguous_final_audit_tag"
        elif principal == {"AMBIGUOUS_TEMPORAL_RENAME"}:
            policy = "AMBIGUOUS_TEMPORAL_RENAME"
            resolution = "unambiguous_final_audit_tag"
        elif principal == {"PARTITION_QUARANTINE", "AMBIGUOUS_TEMPORAL_RENAME"}:
            raw_q = [a.split("=>", 1)[0] for a in quarantined]
            prefix = "SRCASSAY:" + family
            has_plain = prefix in raw_q
            has_pol = any(a.startswith(prefix + "_POL_") for a in raw_q)
            if len(components) > 1:
                policy = "PARTITION_QUARANTINE"
                resolution = "multi_component_family_specific_partition_evidence"
            elif len(components) == 1 and has_plain and has_pol:
                policy = "AMBIGUOUS_TEMPORAL_RENAME"
                resolution = "single_component_plain_plus_pol_quarantine_evidence"
            else:
                raise PaperAnalysisError(
                    "Cannot resolve mixed PARTITION/AMBIGUOUS family: "
                    f"family={family}, components={len(components)}, "
                    f"quarantined={quarantined}, tags={sorted(tags)}"
                )
        else:
            raise PaperAnalysisError(
                f"Family {family} has unresolved principal policy tags {sorted(principal)}"
            )

        source_id, base_id = (family.split(":", 1) + [""])[:2]
        rows.append(
            {
                "source_id": source_id,
                "base_source_assay_id": base_id,
                "family_id": family,
                "policy": policy,
                "policy_resolution": resolution,
                "all_row_policy_tags": "|".join(sorted(tags)),
                "n_canonical_assay_components": len(components),
                "safe_assay_identities": "|".join(components),
                "renamed_external_aliases": "|".join(renamed),
                "quarantined_external_aliases": "|".join(quarantined),
                "n_reconciled_raw_aliases": len(renamed),
                "n_quarantined_raw_aliases": len(quarantined),
            }
        )

    out = pd.DataFrame(rows).sort_values(
        ["source_id", "base_source_assay_id", "family_id"], kind="mergesort"
    ).reset_index(drop=True)

    global_renamed: set[str] = set()
    global_quarantined: set[str] = set()
    for value in assay_audit["renamed_external_aliases"]:
        global_renamed.update(_split_pipe_list(value))
    for value in assay_audit["quarantined_external_aliases"]:
        global_quarantined.update(_split_pipe_list(value))

    table_renamed: set[str] = set()
    table_quarantined: set[str] = set()
    for value in out["renamed_external_aliases"]:
        table_renamed.update(_split_pipe_list(value))
    for value in out["quarantined_external_aliases"]:
        table_quarantined.update(_split_pipe_list(value))
    if table_renamed != global_renamed:
        raise PaperAnalysisError("Family policy table does not preserve renamed aliases")
    if table_quarantined != global_quarantined:
        raise PaperAnalysisError("Family policy table does not preserve quarantined aliases")
    return out


def source_policy_summary(policy: pd.DataFrame) -> dict[str, int]:
    counts = policy["policy"].value_counts().to_dict()
    renamed: set[str] = set()
    quarantined: set[str] = set()
    for value in policy["renamed_external_aliases"]:
        renamed.update(_split_pipe_list(value))
    for value in policy["quarantined_external_aliases"]:
        quarantined.update(_split_pipe_list(value))
    return {
        "source_policy_temporal_rename_families": int(counts.get("TEMPORAL_RENAME", 0)),
        "source_policy_partition_quarantine_families": int(counts.get("PARTITION_QUARANTINE", 0)),
        "source_policy_ambiguous_temporal_families": int(counts.get("AMBIGUOUS_TEMPORAL_RENAME", 0)),
        "source_policy_reconciled_raw_aliases": int(len(renamed)),
        "source_policy_quarantined_aliases": int(len(quarantined)),
    }


def _split_pipe(value: object) -> set[str]:
    return set(_split_pipe_list(value))


def build_assay_policy_map_from_final_audit(
    assay_audit: pd.DataFrame,
) -> dict[str, set[str]]:
    required = {"canonical_key", "assay_source_policy_types"}
    missing = sorted(required - set(assay_audit.columns))
    if missing:
        raise PaperAnalysisError(
            "Final assay audit lacks mechanism policy columns: " + ", ".join(missing)
        )
    return {
        str(row.canonical_key): _split_pipe(row.assay_source_policy_types)
        for row in assay_audit[["canonical_key", "assay_source_policy_types"]].itertuples(index=False)
    }


def attribute_mechanisms(
    displacement: pd.DataFrame,
    assay_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Assign one principal mechanism to each activity with a later branch."""
    affected = displacement[displacement["has_later_branch"]].copy()
    policy_map = build_assay_policy_map_from_final_audit(assay_audit)
    weak_keys: set[str] = set()
    if "n_quarantined_weak_aliases" in assay_audit.columns:
        weak_keys = set(
            assay_audit.loc[
                pd.to_numeric(
                    assay_audit["n_quarantined_weak_aliases"], errors="coerce"
                ).fillna(0).gt(0),
                "canonical_key",
            ].astype(str)
        )

    def classify(assay_key: str) -> tuple[str, bool, bool]:
        policy = policy_map.get(str(assay_key), set())
        if "TEMPORAL_RENAME" in policy:
            mechanism = "STRICT_TEMPORAL_SOURCE_ASSAY_RENAME"
        elif "PARTITION_QUARANTINE" in policy:
            mechanism = "PARTITION_NONINJECTIVE_PROTECTION"
        else:
            mechanism = "ORDINARY_DETERMINISTIC_ALIAS_BRIDGING"
        return (
            mechanism,
            "AMBIGUOUS_TEMPORAL_RENAME" in policy,
            str(assay_key) in weak_keys,
        )

    values = affected["assay_identity_key"].astype(str).map(classify)
    affected["mechanism"] = [x[0] for x in values]
    affected["ambiguous_temporal_safeguard"] = [x[1] for x in values]
    affected["weak_alias_quarantine_safeguard"] = [x[2] for x in values]
    return affected


def _state_release_stats(
    group: pd.DataFrame,
    value_col: str,
    order: dict[str, int],
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for value, sub in group[group[value_col].astype(str).ne("")].groupby(
        value_col, dropna=False
    ):
        releases = sorted(set(sub["release_label"].astype(str)), key=lambda x: order[x])
        out[str(value)] = {
            "releases": releases,
            "n_releases": len(releases),
            "min_order": min(order[r] for r in releases),
            "max_order": max(order[r] for r in releases),
        }
    return out


def derive_document_events(
    frames: list[pd.DataFrame],
    manifest: pd.DataFrame,
    norm_text: Callable[[object], str],
    norm_doi: Callable[[object], str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Summarize malformed DOI metadata, shared DOI collisions and recurations."""
    order = _release_order(manifest)
    rows: list[pd.DataFrame] = []
    for frame in frames:
        if frame.empty:
            continue
        cols = [
            c
            for c in [
                "release_label",
                "document_identity_key",
                "document_doi",
                "document_pubmed_id",
                "document_chembl_id",
                "document_year",
                "document_title",
                "document_journal",
                "document_volume",
                "document_first_page",
            ]
            if c in frame.columns
        ]
        part = frame[cols].drop_duplicates().copy()
        if "document_identity_key" not in part.columns:
            raise PaperAnalysisError("Canonical document_identity_key is required")
        part["doi_raw_text"] = part.get(
            "document_doi", pd.Series("", index=part.index)
        ).map(norm_text)
        part["doi_norm"] = part.get(
            "document_doi", pd.Series("", index=part.index)
        ).map(norm_doi)
        part["pmid_norm"] = part.get(
            "document_pubmed_id", pd.Series("", index=part.index)
        ).map(norm_text)
        rows.append(part)
    if not rows:
        raise PaperAnalysisError("No document occurrences available")
    occurrences = pd.concat(rows, ignore_index=True)

    malformed = occurrences[
        occurrences["doi_raw_text"].ne("")
        & occurrences["doi_raw_text"].map(
            lambda s: bool(NUMERIC_LIKE_RE.fullmatch(str(s)))
        )
    ].copy()
    malformed_cols = [
        c
        for c in [
            "release_label",
            "document_identity_key",
            "document_chembl_id",
            "doi_raw_text",
        ]
        if c in malformed.columns
    ]
    malformed = (
        malformed[malformed_cols]
        .drop_duplicates()
        .sort_values(
            ["release_label", "document_identity_key", "doi_raw_text"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    events: list[dict] = []
    for row in malformed.itertuples(index=False):
        events.append(
            {
                "event_type": "MALFORMED_NUMERIC_DOI_LIKE",
                "document_identity_key": str(getattr(row, "document_identity_key", "")),
                "external_id_type": "DOI",
                "earlier_value": str(getattr(row, "doi_raw_text", "")),
                "later_value": "",
                "release_labels": str(getattr(row, "release_label", "")),
                "n_release_groups": 1,
                "decision": "EXCLUDE_FROM_DOI_IDENTITY_EVIDENCE",
            }
        )

    collision_aliases: list[str] = []
    collision_release_groups = 0
    doi_nonempty = occurrences[occurrences["doi_norm"].ne("")].copy()
    for doi, group in doi_nonempty.groupby("doi_norm"):
        identities = sorted(set(group["document_identity_key"].astype(str)))
        if len(identities) <= 1:
            continue
        collision_aliases.append(str(doi))
        affected_releases = []
        for release, release_group in group.groupby("release_label", sort=False):
            release_ids = sorted(set(release_group["document_identity_key"].astype(str)))
            if len(release_ids) > 1:
                affected_releases.append(str(release))
        affected_releases = sorted(set(affected_releases), key=lambda x: order[x])
        collision_release_groups += len(affected_releases)
        events.append(
            {
                "event_type": "SHARED_DOI_COLLISION",
                "document_identity_key": "|".join(identities),
                "external_id_type": "DOI",
                "earlier_value": str(doi),
                "later_value": "",
                "release_labels": "|".join(affected_releases),
                "n_release_groups": len(affected_releases),
                "decision": "QUARANTINE_SHARED_DOI_ALIAS",
            }
        )

    recurations: list[dict] = []
    for document_key, group in occurrences.groupby("document_identity_key", dropna=False):
        doi_stats = _state_release_stats(group, "doi_norm", order)
        pmid_stats = _state_release_stats(group, "pmid_norm", order)
        for external_type, stats, other in [
            ("DOI", doi_stats, pmid_stats),
            ("PMID", pmid_stats, doi_stats),
        ]:
            if len(stats) != 2 or len(other) > 1:
                continue
            states = sorted(stats.items(), key=lambda item: item[1]["min_order"])
            (old_value, old_state), (new_value, new_state) = states
            repeated = old_state["n_releases"] >= 2 and new_state["n_releases"] >= 2
            chronological = old_state["max_order"] < new_state["min_order"]
            if repeated and chronological:
                releases = sorted(
                    set(old_state["releases"] + new_state["releases"]),
                    key=lambda x: order[x],
                )
                event = {
                    "event_type": "STRICT_TEMPORAL_EXTERNAL_ID_RECURATION",
                    "document_identity_key": str(document_key),
                    "external_id_type": external_type,
                    "earlier_value": old_value,
                    "later_value": new_value,
                    "release_labels": "|".join(releases),
                    "n_release_groups": len(releases),
                    "decision": "ACCEPT_TEMPORAL_RECURATION",
                }
                recurations.append(event)
                events.append(event)

    event_table = pd.DataFrame(events)
    summary = {
        "malformed_numeric_doi_release_occurrences": int(len(malformed)),
        "malformed_numeric_doi_distinct_values": int(malformed["doi_raw_text"].nunique())
        if not malformed.empty
        else 0,
        "malformed_numeric_doi_document_identities": int(
            malformed["document_identity_key"].nunique()
        )
        if not malformed.empty
        else 0,
        "shared_doi_collision_aliases": len(collision_aliases),
        "shared_doi_collision_release_groups": int(collision_release_groups),
        "document_temporal_recurations": len(recurations),
        "document_temporal_recurations_pmid": sum(
            e["external_id_type"] == "PMID" for e in recurations
        ),
        "document_temporal_recurations_doi": sum(
            e["external_id_type"] == "DOI" for e in recurations
        ),
    }
    return event_table, summary
