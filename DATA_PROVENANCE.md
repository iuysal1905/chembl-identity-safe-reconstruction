# Data provenance

## ChEMBL archives

The analysis uses the retrospective SQLite distributions of ChEMBL releases 1–31. The database archives themselves are not redistributed in this repository.

`config/paper_release_manifest.csv` records the release label, exact release date, release DOI, and schema generation used by the paper.

The code recognizes three historical schema families:

- `pre9_early`: ChEMBL 1–8
- `pre15_legacy`: ChEMBL 9–14
- `post15`: ChEMBL 15–31

The scientific target concept is human single protein. Historical native target semantics are adapted by schema rather than assuming that the modern label existed in every release.

## Release dates

Release chronology is based on the verified historical release metadata. Patch releases 22.1 and 24.1 use the explicit official announcement-date overrides used in the final analysis.

## Checksums

Archive and extracted-file checksums are generated locally during acquisition/extraction. Machine-specific database paths are not committed.

## Molecule standardization

Structure standardization is deterministic and uses full InChIKey as molecular identity. The project preserves stereochemistry and applies the same standardization policy to all eligibility and comparator representations.
