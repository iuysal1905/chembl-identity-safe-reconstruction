# Historical Identifier Drift in ChEMBL — reproducibility code

This repository contains the code used for the manuscript **“Historical Identifier Drift Distorts Longitudinal Bioactivity Analysis: Identity-Safe Reconstruction of ChEMBL.”**

The repository reconstructs document and assay identity across historical ChEMBL releases and measures how cross-release identity treatment changes the apparent activity universe and first availability (F0).

## Scope

The paper analysis uses historical **ChEMBL 1 through ChEMBL 31** SQLite releases. It keeps document identity, assay identity, molecule standardization, activity eligibility, first availability, the controlled release-local comparator, and mechanism attribution separate so that each step can be audited.

The internal Python package remains named `ad_f0` because that was the validated import path used during the analysis. The repository name and documentation have been simplified for publication; the scientific code path is not renamed merely for presentation.

## Main reproducibility checks

A successful paper run reproduces the following locked results:

| Quantity | Final value |
|---|---:|
| Canonical historical document identities | 29,385 |
| Canonical historical assay identities | 124,410 |
| Primary identity-safe activity identities | 884,186 |
| Primary molecule–target–endpoint pairs | 723,433 |
| Release-local activity identities | 1,018,565 |
| Net activity-identity inflation | 134,379 (15.20%) |
| Identity-safe activities fragmented under release-local representation | 134,432 (15.20%) |
| Overmerged release-local identities | 129 (0.013%) |
| Median maximum F0 displacement | 5 releases |
| P95 maximum F0 displacement | 10 releases |
| Maximum F0 displacement | 31 releases |
| Ordinary deterministic alias bridging | 67,046 (49.87%) |
| Strict temporal source-assay rename | 49,079 (36.51%) |
| Partition / non-injective protection | 18,307 (13.62%) |

The source-policy audit additionally resolves **921** temporal-rename families, **198** partition-quarantine families, and **19** ambiguous temporal-rename families; **1,842** raw aliases are reconciled and **236** unique raw aliases are quarantined.

## What is not stored in Git

Historical ChEMBL SQLite archives, extracted Parquet tables, caches, and large row-level result files are deliberately excluded. They are recreated locally from the official ChEMBL archives. This keeps the repository reviewable and avoids embedding machine-specific paths or multi-gigabyte data.

See `DATA_PROVENANCE.md` for the release corpus and `config/paper_release_manifest.csv` for exact dates used by the paper analysis.

## Installation

Python 3.10+ is required. A fresh environment is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev]
pytest -q
```

## Reproduction workflow

### 1. Prepare and probe historical releases

Run the extraction notebooks in order:

1. `notebooks/00_F0_setup_and_lock.ipynb`
2. `notebooks/01_F0_schema_probe.ipynb`
3. `notebooks/02_F0_release_manifest.ipynb`
4. `notebooks/03_F0_extract_archived_chembl.ipynb`

The schema probe explicitly covers the early, legacy, and modern ChEMBL layouts. The extraction stage writes one Parquet table per verified release.

### 2. Reconstruct identity and reproduce the paper comparator

```bash
python scripts/reproduce_jcim_paper.py --root . --validate-paper-locks
```

The script:

- reconstructs document and assay identity from the extracted historical releases;
- applies the primary, relaxed-confidence, and all-direct-assay eligibility definitions;
- reconstructs identity-safe F0;
- builds the controlled release-local comparator from the same eligible observations;
- derives F0 displacement and overmerge topology;
- attributes each displaced activity to the final mutually exclusive mechanism categories;
- writes row-level audits and summary files to `results/paper/`;
- fails if any frozen manuscript result changes when `--validate-paper-locks` is used.

The release-local comparator does **not** make every release occurrence unique. Unchanged unreconciled assay identifiers may recur across releases. What is removed is the canonical cross-release reconciliation used by the identity-safe representation.

## Identity policy in brief

Document identity uses DOI and PMID as strong external evidence, ChEMBL document IDs as internal aliases, and three predefined weak bibliographic aliases. Weak bridges that create an incompatible external-ID component are quarantined and the graph is rebuilt. Residual external conflicts fail closed except for strictly supported temporal document recuration.

Assay identity uses source + source-assay ID as external evidence, ChEMBL assay IDs as internal aliases, and a semantic fingerprint based on canonical document identity, normalized assay description, assay type, and target accession. Source-assay `_POL_` transitions are accepted only when the temporal policy supports continuity; `_POL_` is never stripped globally.

The 1% weak-only internal-ID multiplicity and 1% release-local-only component safeguards remain fail-closed checks.

## Activity identity

The activity key is the SHA-256 serialization of:

`canonical assay identity × full InChIKey × endpoint × relation × molar value × target accession`

Molecule–target–endpoint pair identity is:

`full InChIKey × target accession × endpoint`

## Repository integrity

`SOURCE_CODE_SHA256.txt` records the SHA-256 of the exact `src/ad_f0/first_availability.py` exported from the validated analysis checkout. The release builder refuses to package the older pre-publication identity engine that lacks the final temporal source-policy and document-recuration functions.

## Tests

The unit tests use synthetic fixtures and do not require the historical ChEMBL archives. Full historical reproduction is intentionally a separate, data-dependent step.

A GitHub Actions workflow runs the unit test suite on pushes and pull requests.

## License

No software license is added automatically by the release builder. Choose a license deliberately before making the repository public. For a manuscript code repository, MIT or BSD-3-Clause are common choices, but this is an author decision.
