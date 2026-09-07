# Reproducibility notes

## Paper corpus

The manuscript analysis uses historical ChEMBL releases 1–31. The exact chronological release list used by the public analysis is stored in `config/paper_release_manifest.csv`.

## Fail-closed design

The public code retains the analysis safeguards rather than replacing them with permissive fallbacks:

- conflicting strong document/assay evidence is not silently merged;
- weak aliases implicated in a strong-evidence conflict are quarantined and the component graph is rebuilt;
- source-assay `_POL_` transitions are not normalized by blanket suffix stripping;
- non-injective source identifiers are protected from global merging;
- identity under/over-merge diagnostics retain the frozen 1% fail-closed thresholds;
- malformed or scientifically meaningful values are not silently coerced merely to keep a pipeline running.

## Comparator semantics

The release-local comparator differs from the identity-safe representation in assay identity only. Molecule identity, target, endpoint, relation, standardized molar value, and activity eligibility are unchanged.

The comparator local-key hierarchy is inherited from `_assay_local_keys`. Only native/row fallbacks are release-scoped by that helper. The comparator therefore must **not** prepend `release_label` to every assay key.

## Frozen publication validation

`python scripts/reproduce_jcim_paper.py --validate-paper-locks` checks the final manuscript values after recomputation. The expected values are assertions only; they are not used to construct identities, F0, mechanisms, or sensitivity sets.
