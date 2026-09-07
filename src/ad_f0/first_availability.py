from __future__ import annotations
import re
import hashlib
import pandas as pd


class HistoricalIdentityConflictError(RuntimeError):
    """Raised when identity-only aliases imply an ambiguous cross-release entity merge."""
    def __init__(self, message: str, audit: pd.DataFrame):
        super().__init__(message)
        self.audit = audit


def _norm_identity_text(v) -> str:
    if v is None or pd.isna(v):
        return ''
    import re
    # SQLite integer columns with NULLs may arrive as floats. Prevent 123 -> '123.0'
    # alias drift across releases while leaving genuinely decimal/string tokens unchanged.
    if isinstance(v, float) and v.is_integer():
        v=int(v)
    return re.sub(r'\s+', ' ', str(v).strip()).upper()


def _norm_doi(v) -> str:
    s = _norm_identity_text(v)
    if not s:
        return ""

    # Canonical DOI presentation
    s = re.sub(
        r"^HTTPS?://(DX\.)?DOI\.ORG/",
        "",
        s,
        flags=re.I,
    )
    s = re.sub(
        r"^DOI\s*:\s*",
        "",
        s,
        flags=re.I,
    )
    s = s.strip()

    # A value that cannot syntactically be a DOI must not
    # become an authoritative DOI identity edge.
    #
    # Deliberately permissive so historical/synthetic DOI
    # forms such as 10.1/A remain supported.
    if not re.fullmatch(
        r"10\.[^/\s]+/\S+",
        s,
        flags=re.I,
    ):
        return ""

    return s.upper()




def _norm_pmid(v) -> str:
    """Canonicalize PubMed identifiers as unsigned integer strings.

    Representation-only floating forms introduced by dataframe dtype
    coercion (for example ``15177484.0``) are accepted and reduced to
    their integer representation. Non-integer values are not accepted
    as authoritative PMID aliases.
    """

    s = _norm_identity_text(v)

    if not s:
        return ''

    if re.fullmatch(r'\d+\.0+', s):
        s = s.split('.', 1)[0]

    if not re.fullmatch(r'\d+', s):
        return ''

    return str(int(s))

def _norm_identity_series(s: pd.Series) -> pd.Series:
    # Vectorised normalisation for the large historical metadata pass.
    out=s.astype('string').fillna('').str.strip().str.replace(r'\s+',' ',regex=True).str.upper()
    return out.astype(str)


def _identity_sha(prefix: str, parts: list[str]) -> str:
    return prefix + hashlib.sha256('\x1f'.join(parts).encode('utf-8')).hexdigest()


DOC_BIB_FIELDS=['document_journal','document_year','document_volume','document_issue','document_first_page','document_title']
DOC_ALIAS_FIELDS=DOC_BIB_FIELDS+['document_doi','document_pubmed_id','document_chembl_id','native_document_id','schema_generation','release_label']
ASSAY_ALIAS_FIELDS=['assay_description','assay_type','target_accession','assay_chembl_id','assay_source_id','assay_source_assay_id','native_assay_id','assay_identity_key','schema_generation','release_label']


def _document_weak_aliases(r: pd.Series) -> list[str]:
    """Return only preregistered sufficiently-specific bibliographic aliases.

    Three aliases are allowed independently so sparse early-release metadata can bridge
    longitudinally without making journal+year+volume sufficient:
      - year + normalized title
      - year + journal + first page
      - journal + volume + first page (year-independent fallback)

    The last alias is emitted whenever its fields exist, including rows that also have a
    publication year, so a release with missing year can bridge to a later enriched release.
    Journal+year+volume alone remains prohibited.
    """
    vals={c:_norm_identity_text(r.get(c)) for c in DOC_BIB_FIELDS}
    out=[]
    if vals['document_year'] and vals['document_title']:
        out.append(_identity_sha('DOC_TITLE:', [vals['document_year'],vals['document_title']]))
    if vals['document_year'] and vals['document_journal'] and vals['document_first_page']:
        out.append(_identity_sha('DOC_PAGE:', [vals['document_year'],vals['document_journal'],vals['document_first_page']]))
    if vals['document_journal'] and vals['document_volume'] and vals['document_first_page']:
        out.append(_identity_sha('DOC_VOLPAGE:', [vals['document_journal'],vals['document_volume'],vals['document_first_page']]))
    return sorted(set(out))


def _document_weak_alias(r: pd.Series) -> str:
    """Compatibility helper returning the first admissible weak alias, if any."""
    a=_document_weak_aliases(r)
    return a[0] if a else ''


class _UnionFind:
    def __init__(self):
        self.parent={}
    def add(self,x):
        if x and x not in self.parent: self.parent[x]=x
    def find(self,x):
        p=self.parent[x]
        if p!=x:
            self.parent[x]=self.find(p)
        return self.parent[x]
    def union(self,a,b):
        self.add(a); self.add(b)
        ra,rb=self.find(a),self.find(b)
        if ra!=rb:
            lo,hi=sorted([ra,rb])
            self.parent[hi]=lo
    def groups(self):
        out={}
        for x in list(self.parent):
            out.setdefault(self.find(x),[]).append(x)
        return {k:sorted(v) for k,v in out.items()}


def _canonical_alias_components(alias_sets: list[list[str]], kind: str) -> tuple[list[str], pd.DataFrame]:
    uf=_UnionFind()
    for aliases in alias_sets:
        a=sorted(set(x for x in aliases if x))
        for x in a: uf.add(x)
        for x in a[1:]: uf.union(a[0],x)
    groups=uf.groups()
    alias_to_canon={}
    rows=[]
    for aliases in groups.values():
        canon=_identity_sha(f'{kind}_CANON:',aliases)
        for a in aliases: alias_to_canon[a]=canon
        prefixes={
            'chembl':[a for a in aliases if a.startswith('CHEMBL_')],
            'doi':[a for a in aliases if a.startswith('DOI:')],
            'pmid':[a for a in aliases if a.startswith('PMID:')],
            'source':[a for a in aliases if a.startswith('SRCASSAY:')],
            'weak':[a for a in aliases if a.startswith('DOC_TITLE:') or a.startswith('DOC_PAGE:') or a.startswith('DOC_VOLPAGE:') or a.startswith('ASSAY_SEM:')],
            'local':[a for a in aliases if a.startswith('LOCAL_')],
        }
        rows.append({'canonical_key':canon,'aliases':'|'.join(aliases),'n_aliases':len(aliases),
                     'n_chembl_aliases':len(prefixes['chembl']),'n_doi_aliases':len(prefixes['doi']),
                     'n_pmid_aliases':len(prefixes['pmid']),'n_source_aliases':len(prefixes['source']),
                     'n_weak_aliases':len(prefixes['weak']),'n_local_aliases':len(prefixes['local'])})
    canon=[]
    for aliases in alias_sets:
        a=[x for x in aliases if x]
        canon.append(alias_to_canon[a[0]] if a else '')
    return canon,pd.DataFrame(rows)




def _is_weak_alias(alias: str, kind: str) -> bool:
    """Return whether an alias is a non-authoritative semantic/bibliographic bridge."""
    if kind == 'document':
        return str(alias).startswith(('DOC_TITLE:', 'DOC_PAGE:', 'DOC_VOLPAGE:'))
    if kind == 'assay':
        return str(alias).startswith('ASSAY_SEM:')
    raise ValueError(f'Unknown identity kind: {kind}')


def _external_conflict_mask(base: pd.DataFrame, kind: str) -> pd.Series:
    """External identifiers are hard constraints; weak aliases may never override them."""
    if base.empty:
        return pd.Series(False, index=base.index, dtype=bool)
    if kind == 'document':
        return base['n_doi_aliases'].gt(1) | base['n_pmid_aliases'].gt(1)
    if kind == 'assay':
        return base['n_source_aliases'].gt(1)
    raise ValueError(f'Unknown identity kind: {kind}')



_TEMPORAL_DOC_BIB_FIELDS = (
    'document_year',
    'document_journal',
    'document_volume',
    'document_issue',
    'document_first_page',
    'document_title',
)


def _norm_temporal_integer_like(v) -> str:
    """Canonicalize integer-like bibliographic values without semantic inference."""

    s = _norm_identity_text(v)

    if not s:
        return ''

    if re.fullmatch(r'[+-]?\d+\.0+', s):
        return s.split('.', 1)[0]

    if re.fullmatch(r'[+-]?\d+', s):
        return str(int(s))

    return s


def _norm_temporal_bib_value(column: str, value) -> str:
    """Normalize only representation-level differences used by temporal validation."""

    s = _norm_identity_text(value)

    if not s:
        return ''

    if column in {
        'document_year',
        'document_volume',
        'document_issue',
        'document_first_page',
    }:
        return _norm_temporal_integer_like(s)

    return re.sub(r'\s+', ' ', s).strip()


def _document_temporal_recuration_context(
    release_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    """Build compact release-ordered document metadata for temporal validation.

    Activity, potency, target and model-output fields are intentionally excluded.

    DOI aliases already identified by the strict within-release shared-DOI
    collision policy are excluded from active temporal identity edges, while
    their raw normalized DOI values are retained separately for the global
    exclusivity safeguard.
    """

    parts = []

    for frame_index, df in enumerate(release_frames):

        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        quarantined_dois = (
            _document_local_doi_collision_quarantine(df)
        )

        part = pd.DataFrame(index=df.index)

        part['_frame_index'] = frame_index

        part['release_label'] = _series(
            df,
            'release_label',
        )

        part['_chembl'] = _series(
            df,
            'document_chembl_id',
        )

        if 'document_doi' in df.columns:
            doi_raw = df['document_doi'].map(
                _norm_doi
            )
        else:
            doi_raw = pd.Series(
                '',
                index=df.index,
                dtype='object',
            )

        part['_doi_raw'] = doi_raw
        part['_doi'] = doi_raw.copy()

        if quarantined_dois:
            part.loc[
                part['_doi'].isin(
                    quarantined_dois
                ),
                '_doi',
            ] = ''

        if 'document_pubmed_id' in df.columns:
            pmid_raw = df['document_pubmed_id'].map(
                _norm_pmid
            )
        else:
            pmid_raw = pd.Series(
                '',
                index=df.index,
                dtype='object',
            )

        part['_pmid'] = pmid_raw
        part['_pmid_raw'] = pmid_raw

        for col in _TEMPORAL_DOC_BIB_FIELDS:

            if col in df.columns:
                part[col] = df[col].map(
                    lambda v, c=col:
                        _norm_temporal_bib_value(
                            c,
                            v,
                        )
                )
            else:
                part[col] = ''

        parts.append(
            part.drop_duplicates()
        )

    if not parts:
        return pd.DataFrame(
            columns=[
                '_frame_index',
                'release_label',
                '_chembl',
                '_doi',
                '_doi_raw',
                '_pmid',
                '_pmid_raw',
                *_TEMPORAL_DOC_BIB_FIELDS,
            ]
        )

    return (
        pd.concat(
            parts,
            ignore_index=True,
            sort=False,
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )


def _classify_temporal_external_recuration_components(
    bad: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the preregistered strict temporal-recuration predicate.

    A residual document component passes only when all conditions hold:

      1. exactly one ChEMBL document ID;
      2. exactly two conflicting external IDs of one authority type;
      3. the two IDs are never simultaneous in one release/frame;
      4. exactly one one-way state transition occurs;
      5. no reversion occurs;
      6. normalized bibliography is fieldwise compatible;
      7. both external IDs are globally exclusive to the target lineage.

    No winner is selected between the historical external IDs.
    """

    output = []

    if bad.empty:
        return pd.DataFrame()

    if context is None or context.empty:
        context = pd.DataFrame()

    def parse_aliases(row):
        return {
            x
            for x in str(
                row.get('aliases', '')
            ).split('|')
            if x
        }

    def values_with_prefix(aliases, prefix):
        return sorted(
            x[len(prefix):]
            for x in aliases
            if x.startswith(prefix)
        )

    for _, row in bad.iterrows():

        aliases = parse_aliases(row)

        chembl_ids = values_with_prefix(
            aliases,
            'CHEMBL_DOC:',
        )

        dois = values_with_prefix(
            aliases,
            'DOI:',
        )

        pmids = values_with_prefix(
            aliases,
            'PMID:',
        )

        result = {
            'canonical_key':
                row.get('canonical_key', ''),

            'temporal_recuration_pass':
                False,

            'temporal_external_type':
                '',

            'temporal_old_external_id':
                '',

            'temporal_new_external_id':
                '',

            'temporal_transition':
                '',

            'temporal_old_state_n_frames':
                0,

            'temporal_new_state_n_frames':
                0,

            'temporal_simultaneous_conflict':
                False,

            'temporal_bibliographically_compatible':
                False,

            'temporal_globally_exclusive':
                False,

            'temporal_recuration_failure_reason':
                '',
        }


        # ----------------------------------------------------
        # 1. Exactly one internal ChEMBL document lineage
        # ----------------------------------------------------

        if len(chembl_ids) != 1:

            result[
                'temporal_recuration_failure_reason'
            ] = 'NOT_EXACTLY_ONE_CHEMBL_DOCUMENT'

            output.append(result)
            continue


        target_chembl = chembl_ids[0]


        # ----------------------------------------------------
        # 2. Exactly two IDs of one conflicting authority
        # ----------------------------------------------------

        if (
            len(pmids) == 2
            and len(dois) <= 1
        ):

            ext_type = 'PMID'
            ext_values = pmids
            ext_col = '_pmid'
            raw_ext_col = '_pmid_raw'

        elif (
            len(dois) == 2
            and len(pmids) <= 1
        ):

            ext_type = 'DOI'
            ext_values = dois
            ext_col = '_doi'
            raw_ext_col = '_doi_raw'

        else:

            result[
                'temporal_recuration_failure_reason'
            ] = (
                'NOT_EXACTLY_TWO_IDS_OF_ONE_EXTERNAL_AUTHORITY'
            )

            output.append(result)
            continue


        result[
            'temporal_external_type'
        ] = ext_type


        if context.empty:

            result[
                'temporal_recuration_failure_reason'
            ] = 'NO_ARCHIVE_EVIDENCE'

            output.append(result)
            continue


        # ----------------------------------------------------
        # Relevant historical lineage
        # ----------------------------------------------------

        alias_mask = context[
            ext_col
        ].isin(
            ext_values
        )

        target_mask = context[
            '_chembl'
        ].eq(
            target_chembl
        )

        relevant = context[
            alias_mask | target_mask
        ].copy()


        if relevant.empty:

            result[
                'temporal_recuration_failure_reason'
            ] = 'NO_ARCHIVE_EVIDENCE'

            output.append(result)
            continue


        # ----------------------------------------------------
        # 3–5. Release chronology:
        #      no simultaneity, one transition, no reversion
        # ----------------------------------------------------

        states = []
        simultaneous = False

        for frame_index in sorted(
            relevant[
                '_frame_index'
            ].unique()
        ):

            g = relevant[
                relevant[
                    '_frame_index'
                ].eq(
                    frame_index
                )
            ]

            vals = sorted({
                v
                for v in g[ext_col]
                if v in ext_values
            })

            if len(vals) > 1:
                simultaneous = True

            if len(vals) == 1:
                states.append(
                    (
                        int(frame_index),
                        vals[0],
                    )
                )


        result[
            'temporal_simultaneous_conflict'
        ] = simultaneous


        if simultaneous:

            result[
                'temporal_recuration_failure_reason'
            ] = (
                'EXTERNAL_IDS_SIMULTANEOUS_IN_RELEASE'
            )

            output.append(result)
            continue


        compressed = []

        for frame_index, state in states:

            if (
                not compressed
                or compressed[-1][1] != state
            ):
                compressed.append(
                    (
                        frame_index,
                        state,
                    )
                )


        if len(compressed) != 2:

            result[
                'temporal_recuration_failure_reason'
            ] = (
                'NOT_EXACTLY_ONE_ONE_WAY_TRANSITION'
            )

            output.append(result)
            continue


        first_frame, old_id = compressed[0]
        second_frame, new_id = compressed[1]


        if set(
            [
                old_id,
                new_id,
            ]
        ) != set(
            ext_values
        ):

            result[
                'temporal_recuration_failure_reason'
            ] = (
                'TRANSITION_DOES_NOT_COVER_BOTH_EXTERNAL_IDS'
            )

            output.append(result)
            continue


        old_state_frames = sorted({
            frame_index
            for frame_index, state in states
            if state == old_id
        })

        new_state_frames = sorted({
            frame_index
            for frame_index, state in states
            if state == new_id
        })

        result[
            'temporal_old_state_n_frames'
        ] = len(old_state_frames)

        result[
            'temporal_new_state_n_frames'
        ] = len(new_state_frames)

        # A one-off A -> B observation is not sufficient evidence
        # of archive recuration. Both historical states must have
        # replicated support in at least two distinct release frames.
        if (
            len(old_state_frames) < 2
            or len(new_state_frames) < 2
        ):

            result[
                'temporal_recuration_failure_reason'
            ] = 'INSUFFICIENT_REPLICATED_TEMPORAL_SUPPORT'

            output.append(result)
            continue


        result[
            'temporal_old_external_id'
        ] = old_id

        result[
            'temporal_new_external_id'
        ] = new_id

        result[
            'temporal_transition'
        ] = (
            f'F{first_frame}:{old_id}'
            f' -> '
            f'F{second_frame}:{new_id}'
        )


        # ----------------------------------------------------
        # 6. Fieldwise bibliographic compatibility
        # ----------------------------------------------------

        contradictions = {}

        for col in _TEMPORAL_DOC_BIB_FIELDS:

            vals = sorted({
                v
                for v in relevant[col]
                if v
            })

            if len(vals) > 1:
                contradictions[col] = vals


        bib_ok = (
            len(contradictions) == 0
        )

        result[
            'temporal_bibliographically_compatible'
        ] = bib_ok


        if not bib_ok:

            result[
                'temporal_recuration_failure_reason'
            ] = (
                'BIBLIOGRAPHIC_CONTRADICTION:'
                + repr(
                    contradictions
                )
            )

            output.append(result)
            continue


        # ----------------------------------------------------
        # 7. Global exclusivity
        #
        # Use RAW external metadata here, deliberately stricter
        # than the active identity-edge representation.
        # ----------------------------------------------------

        external_rows = context[
            context[
                raw_ext_col
            ].isin(
                ext_values
            )
        ].copy()


        other_chems = sorted({
            v
            for v in external_rows[
                '_chembl'
            ]
            if (
                v
                and v != target_chembl
            )
        })


        global_ok = (
            len(other_chems) == 0
        )


        result[
            'temporal_globally_exclusive'
        ] = global_ok


        if not global_ok:

            result[
                'temporal_recuration_failure_reason'
            ] = (
                'EXTERNAL_ID_USED_BY_OTHER_CHEMBL_DOCUMENT:'
                + '|'.join(
                    other_chems
                )
            )

            output.append(result)
            continue


        # ----------------------------------------------------
        # PASS
        # ----------------------------------------------------

        result[
            'temporal_recuration_pass'
        ] = True

        result[
            'temporal_recuration_failure_reason'
        ] = ''

        output.append(result)


    return pd.DataFrame(output)


def _protected_alias_components(
    alias_sets: list[list[str]],
    kind_label: str,
    kind: str,
    fallback_aliases: list[str],
    document_temporal_context: pd.DataFrame | None=None,
    initial_quarantined_weak_aliases: set[str] | None=None,
) -> tuple[list[str], pd.DataFrame]:
    """Canonicalize while treating external identifiers as hard constraints.

    Weak aliases are candidate bridges. If a weak bridge participates in a
    component containing incompatible external identifiers, every weak alias
    in that conflicted component is quarantined and the graph is rebuilt.

    For documents only, after all weak aliases have been removed, a residual
    DOI/PMID conflict may pass solely through the strict temporal external-ID
    recuration predicate. Anything outside that predicate remains fail-closed.
    """

    if len(alias_sets) != len(fallback_aliases):
        raise ValueError(
            'alias_sets and fallback_aliases must have identical length'
        )

    original = [
        sorted(
            set(
                str(x)
                for x in aliases
                if x
            )
        )
        for aliases in alias_sets
    ]

    all_weak = sorted({
        a
        for aliases in original
        for a in aliases
        if _is_weak_alias(
            a,
            kind,
        )
    })

    quarantined: set[str] = {
        str(a)
        for a in (
            initial_quarantined_weak_aliases
            or set()
        )
        if (
            a
            and _is_weak_alias(
                str(a),
                kind,
            )
        )
    }

    quarantine_iteration: dict[str, int] = {
        a: 0
        for a in quarantined
    }

    temporal_meta = pd.DataFrame()


    def finalize(
        keys,
        base,
    ):

        # ----------------------------------------------------
        # Weak-alias quarantine audit
        # ----------------------------------------------------

        qrows = []

        for key, aliases in zip(
            keys,
            original,
        ):

            qq = sorted(
                a
                for a in aliases
                if a in quarantined
            )

            if qq:
                qrows.append(
                    (
                        key,
                        qq,
                    )
                )


        if qrows:

            qdf = pd.DataFrame(
                qrows,
                columns=[
                    'canonical_key',
                    '_q',
                ],
            )

            qagg = (
                qdf.groupby(
                    'canonical_key'
                )['_q']
                .agg(
                    lambda rows:
                        sorted({
                            a
                            for xs in rows
                            for a in xs
                        })
                )
                .reset_index()
            )

            qagg[
                'quarantined_weak_aliases'
            ] = qagg['_q'].map(
                lambda xs:
                    '|'.join(xs)
            )

            qagg[
                'n_quarantined_weak_aliases'
            ] = qagg['_q'].map(
                len
            )

            qagg[
                'quarantine_first_iteration'
            ] = qagg['_q'].map(
                lambda xs:
                    min(
                        (
                            quarantine_iteration.get(
                                a,
                                0,
                            )
                            for a in xs
                        ),
                        default=0,
                    )
            )

            qagg = qagg.drop(
                columns=[
                    '_q',
                ]
            )

            base = base.merge(
                qagg,
                on='canonical_key',
                how='left',
                validate='one_to_one',
            )


        if 'quarantined_weak_aliases' not in base:

            base[
                'quarantined_weak_aliases'
            ] = ''

            base[
                'n_quarantined_weak_aliases'
            ] = 0

            base[
                'quarantine_first_iteration'
            ] = 0

        else:

            base[
                'quarantined_weak_aliases'
            ] = (
                base[
                    'quarantined_weak_aliases'
                ]
                .fillna('')
            )

            base[
                'n_quarantined_weak_aliases'
            ] = (
                base[
                    'n_quarantined_weak_aliases'
                ]
                .fillna(0)
                .astype(int)
            )

            base[
                'quarantine_first_iteration'
            ] = (
                base[
                    'quarantine_first_iteration'
                ]
                .fillna(0)
                .astype(int)
            )


        # ----------------------------------------------------
        # Temporal-recuration audit
        # ----------------------------------------------------

        if (
            kind == 'document'
            and temporal_meta is not None
            and not temporal_meta.empty
        ):

            base = base.merge(
                temporal_meta,
                on='canonical_key',
                how='left',
                validate='one_to_one',
            )


        if kind == 'document':

            bool_defaults = [
                'temporal_recuration_pass',
                'temporal_simultaneous_conflict',
                'temporal_bibliographically_compatible',
                'temporal_globally_exclusive',
            ]

            text_defaults = [
                'temporal_external_type',
                'temporal_old_external_id',
                'temporal_new_external_id',
                'temporal_transition',
                'temporal_recuration_failure_reason',
            ]

            integer_defaults = [
                'temporal_old_state_n_frames',
                'temporal_new_state_n_frames',
            ]

            for col in integer_defaults:

                if col not in base:
                    base[col] = 0
                else:
                    base[col] = (
                        base[col]
                        .fillna(0)
                        .astype(int)
                    )

            for col in bool_defaults:

                if col not in base:
                    base[col] = False
                else:
                    base[col] = (
                        base[col]
                        .fillna(False)
                        .astype(bool)
                    )

            for col in text_defaults:

                if col not in base:
                    base[col] = ''
                else:
                    base[col] = (
                        base[col]
                        .fillna('')
                    )


        base[
            'weak_alias_external_constraint_pass'
        ] = True

        return keys, base


    # At least one new weak alias is removed on every
    # non-terminal iteration.
    for iteration in range(
        len(all_weak) + 2
    ):

        active = []

        for i, aliases in enumerate(
            original
        ):

            aa = [
                a
                for a in aliases
                if not (
                    _is_weak_alias(
                        a,
                        kind,
                    )
                    and a in quarantined
                )
            ]

            if not aa:
                aa = [
                    fallback_aliases[i]
                ]

            active.append(
                sorted(
                    set(
                        aa
                    )
                )
            )


        keys, base = (
            _canonical_alias_components(
                active,
                kind_label,
            )
        )

        bad = (
            base[
                _external_conflict_mask(
                    base,
                    kind,
                )
            ]
            if not base.empty
            else base
        )


        if bad.empty:

            return finalize(
                keys,
                base,
            )


        weak_in_bad = set()

        for aliases in bad[
            'aliases'
        ].fillna(''):

            weak_in_bad.update(
                a
                for a in str(
                    aliases
                ).split('|')
                if (
                    a
                    and _is_weak_alias(
                        a,
                        kind,
                    )
                )
            )


        new = sorted(
            weak_in_bad
            - quarantined
        )


        if not new:

            # ------------------------------------------------
            # v1.5m:
            # weak aliases are already exhausted.
            # Only strict document temporal recuration may
            # exempt a residual strong external conflict.
            # ------------------------------------------------

            if (
                kind == 'document'
                and document_temporal_context
                    is not None
            ):

                temporal_meta = (
                    _classify_temporal_external_recuration_components(
                        bad,
                        document_temporal_context,
                    )
                )


                passed_keys = set(
                    temporal_meta.loc[
                        temporal_meta[
                            'temporal_recuration_pass'
                        ].fillna(False),
                        'canonical_key',
                    ]
                )


                unresolved = bad[
                    ~bad[
                        'canonical_key'
                    ].isin(
                        passed_keys
                    )
                ].copy()


                if unresolved.empty:

                    return finalize(
                        keys,
                        base,
                    )


                # Attach classifier diagnostics only to the
                # unresolved rows that will fail closed.
                unresolved = unresolved.merge(
                    temporal_meta,
                    on='canonical_key',
                    how='left',
                    validate='one_to_one',
                )

                msg = (
                    'Conflicting external document aliases remain '
                    'after all weak aliases are quarantined and '
                    'strict temporal-recuration validation fails.'
                )

                raise HistoricalIdentityConflictError(
                    msg,
                    unresolved,
                )


            if kind == 'document':

                msg = (
                    'Conflicting external document aliases remain '
                    'after all weak aliases are quarantined.'
                )

            else:

                msg = (
                    'Conflicting external source-assay aliases remain '
                    'after all weak aliases are quarantined.'
                )


            raise HistoricalIdentityConflictError(
                msg,
                bad,
            )


        for a in new:

            quarantined.add(a)

            quarantine_iteration[
                a
            ] = iteration + 1


    raise RuntimeError(
        'Weak-alias quarantine did not converge'
    )

def _identity_fallback_aliases(occurrences: pd.DataFrame, kind: str) -> list[str]:
    """Stable release-scoped fallbacks used only when every weak alias is quarantined."""
    prefix='LOCAL_DOC_QUAR:' if kind=='document' else 'LOCAL_ASSAY_QUAR:'
    out=[]
    for _,r in occurrences.iterrows():
        release=_norm_identity_text(r.get('release_label'))
        occ=_norm_identity_text(r.get('occurrence_key'))
        out.append(_identity_sha(prefix,[release,occ]))
    return out


def weak_alias_quarantine_summary(doc_audit: pd.DataFrame, assay_audit: pd.DataFrame) -> pd.DataFrame:
    """Component-level audit of weak aliases disabled by external-identity constraints."""
    rows=[]
    for kind,audit in [('document',doc_audit),('assay',assay_audit)]:
        if audit.empty or 'n_quarantined_weak_aliases' not in audit:
            continue
        sub=audit[audit['n_quarantined_weak_aliases'].fillna(0).astype(int).gt(0)]
        for r in sub.itertuples(index=False):
            rows.append({
                'kind':kind,
                'canonical_key':r.canonical_key,
                'n_quarantined_weak_aliases':int(r.n_quarantined_weak_aliases),
                'quarantined_weak_aliases':str(r.quarantined_weak_aliases),
                'schema_generations':getattr(r,'schema_generations',''),
                'release_labels':getattr(r,'release_labels',''),
            })
    return pd.DataFrame(rows,columns=[
        'kind','canonical_key','n_quarantined_weak_aliases','quarantined_weak_aliases',
        'schema_generations','release_labels'
    ])

def _series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return _norm_identity_series(df[col])
    return pd.Series('',index=df.index,dtype='object')



def _document_local_doi_collision_quarantine(df: pd.DataFrame) -> set[str]:
    """Return DOI aliases that are unsafe as release-local document edges.

    A DOI is quarantined only when one release contains a strict,
    deterministic signature of two or more distinct documents:

    - >1 ChEMBL document ID;
    - >1 PubMed ID;
    - ChEMBL-ID <-> PMID mapping is one-to-one;
    - bibliography distinguishes the documents by first page or title;
    - each ChEMBL document has at most one non-empty page and title;
    - every affected row retains a strong PMID or ChEMBL-ID fallback.

    The raw DOI metadata are never altered. The DOI is excluded only
    from identity grouping / graph edges for the affected release.
    Ambiguous cases remain active and therefore fail closed downstream.
    """

    if df.empty or 'document_doi' not in df.columns:
        return set()

    doi = df['document_doi'].map(_norm_doi)
    pmid = _series(df, 'document_pubmed_id')
    chem = _series(df, 'document_chembl_id')
    page = _series(df, 'document_first_page')
    title = _series(df, 'document_title')

    meta = pd.DataFrame({
        '_doi': doi,
        '_pmid': pmid,
        '_chem': chem,
        '_page': page,
        '_title': title,
    }, index=df.index)

    meta = meta[meta['_doi'] != ''].drop_duplicates()

    if meta.empty:
        return set()

    quarantined: set[str] = set()

    for doi_value, g in meta.groupby('_doi', sort=False):

        chems = sorted({
            str(v) for v in g['_chem'] if str(v)
        })

        pmids = sorted({
            str(v) for v in g['_pmid'] if str(v)
        })

        pages = sorted({
            str(v) for v in g['_page'] if str(v)
        })

        titles = sorted({
            str(v) for v in g['_title'] if str(v)
        })

        # A collision must represent at least two strong document
        # identities and must be bibliographically distinguishable.
        if len(chems) <= 1 or len(pmids) <= 1:
            continue

        if len(pages) <= 1 and len(titles) <= 1:
            continue

        pairs = g[
            (g['_chem'] != '') &
            (g['_pmid'] != '')
        ][['_chem', '_pmid']].drop_duplicates()

        if pairs.empty:
            continue

        chem_to_pmid = (
            pairs.groupby('_chem')['_pmid']
            .agg(lambda s: sorted(set(str(v) for v in s if str(v))))
            .to_dict()
        )

        pmid_to_chem = (
            pairs.groupby('_pmid')['_chem']
            .agg(lambda s: sorted(set(str(v) for v in s if str(v))))
            .to_dict()
        )

        # Every observed ChEMBL document and PMID must participate
        # in the one-to-one mapping.
        if set(chem_to_pmid) != set(chems):
            continue

        if set(pmid_to_chem) != set(pmids):
            continue

        if not all(len(v) == 1 for v in chem_to_pmid.values()):
            continue

        if not all(len(v) == 1 for v in pmid_to_chem.values()):
            continue

        # Bibliographic metadata must not itself be internally
        # contradictory for one ChEMBL document.
        page_map = (
            g[(g['_chem'] != '') & (g['_page'] != '')]
            .groupby('_chem')['_page']
            .agg(lambda s: sorted(set(str(v) for v in s if str(v))))
            .to_dict()
        )

        title_map = (
            g[(g['_chem'] != '') & (g['_title'] != '')]
            .groupby('_chem')['_title']
            .agg(lambda s: sorted(set(str(v) for v in s if str(v))))
            .to_dict()
        )

        if any(len(v) > 1 for v in page_map.values()):
            continue

        if any(len(v) > 1 for v in title_map.values()):
            continue

        # Removing DOI must never leave an affected row without
        # a strong release-local grouping key.
        if not ((g['_pmid'] != '') | (g['_chem'] != '')).all():
            continue

        quarantined.add(str(doi_value))

    return quarantined


def _document_local_keys(df: pd.DataFrame, frame_index: int, quarantined_dois: set[str] | None=None) -> pd.Series:
    """Release-local grouping key used only to shrink the identity metadata table.

    It is not a longitudinal identity. Sparse records with no sufficiently specific
    document metadata are deliberately kept row-distinct rather than falsely merged.
    """
    doi=(df['document_doi'].map(_norm_doi) if 'document_doi' in df else pd.Series('',index=df.index,dtype='object')); pmid=(df['document_pubmed_id'].map(_norm_pmid) if 'document_pubmed_id' in df else pd.Series('',index=df.index,dtype='object')); chem=_series(df,'document_chembl_id')
    native=_series(df,'native_document_id')
    journal=_series(df,'document_journal'); year=_series(df,'document_year'); volume=_series(df,'document_volume')
    issue=_series(df,'document_issue'); first=_series(df,'document_first_page'); title=_series(df,'document_title')
    title_ok=(year!='')&(title!=''); page_ok=(year!='')&(journal!='')&(first!='')
    volpage_ok=(journal!='')&(volume!='')&(first!='')
    weak_ok=title_ok|page_ok|volpage_ok
    weak_raw=pd.Series('',index=df.index,dtype='object')
    weak_raw.loc[title_ok]='TITLE_RAW:'+year.loc[title_ok]+'|'+title.loc[title_ok]
    mpage=(~title_ok)&page_ok
    weak_raw.loc[mpage]='PAGE_RAW:'+year.loc[mpage]+'|'+journal.loc[mpage]+'|'+first.loc[mpage]
    mvol=(~title_ok)&(~page_ok)&volpage_ok
    weak_raw.loc[mvol]='VOLPAGE_RAW:'+journal.loc[mvol]+'|'+volume.loc[mvol]+'|'+first.loc[mvol]
    out=pd.Series('',index=df.index,dtype='object')
    # Local grouping precedence is deliberately not the longitudinal conflict policy.
    quarantined_dois=set(quarantined_dois or ())
    m=(doi!='')&~doi.isin(quarantined_dois)
    out.loc[m]='DOI:'+doi.loc[m]
    m=(out=='')&(pmid!=''); out.loc[m]='PMID:'+pmid.loc[m]
    m=(out=='')&(chem!=''); out.loc[m]='CHEMBL_DOC:'+chem.loc[m]
    m=(out=='')&weak_ok; out.loc[m]=weak_raw.loc[m]
    release=_series(df,'release_label')
    m=(out=='')&(native!=''); out.loc[m]='LOCAL_DOC_ID:'+release.loc[m]+':'+native.loc[m]
    if (out=='').any():
        pos=pd.Series(range(len(df)),index=df.index).astype(str)
        m=out==''; out.loc[m]='LOCAL_DOC_ROW:'+str(frame_index)+':'+pos.loc[m]
    return out


def _document_occurrences(df: pd.DataFrame, frame_index: int) -> tuple[pd.DataFrame,pd.Series]:
    quarantined_dois=_document_local_doi_collision_quarantine(df)
    local=_document_local_keys(
        df,
        frame_index,
        quarantined_dois=quarantined_dois,
    )
    cols=[c for c in DOC_ALIAS_FIELDS if c in df.columns]
    slim=df[cols].copy() if cols else pd.DataFrame(index=df.index)
    slim['_local_document_occurrence']=local.values
    # Retain distinct alias-bearing variants for an occurrence, but never any activity/potency columns.
    slim=slim.drop_duplicates().reset_index(drop=True)
    records=[]
    for occ,g in slim.groupby('_local_document_occurrence',sort=False,dropna=False):
        aliases=set()
        quarantined_external=set()
        for _,r in g.iterrows():
            for weak in _document_weak_aliases(r):
                aliases.add(weak)
            chem=_norm_identity_text(r.get('document_chembl_id'))
            doi=_norm_doi(r.get('document_doi'))
            pmid=_norm_pmid(r.get('document_pubmed_id'))
            if chem:
                aliases.add('CHEMBL_DOC:'+chem)
            if doi:
                if doi in quarantined_dois:
                    quarantined_external.add('DOI:'+doi)
                else:
                    aliases.add('DOI:'+doi)
            if pmid:
                aliases.add('PMID:'+pmid)
        if not aliases:
            aliases.add('LOCAL_DOC_OCC:'+str(occ))
        records.append({
            'occurrence_key':str(occ),'aliases':sorted(aliases),
            'schema_generation':'|'.join(sorted(set(_norm_identity_text(v) for v in g.get('schema_generation',pd.Series(dtype=object)) if _norm_identity_text(v)))),
            'release_label':'|'.join(sorted(set(_norm_identity_text(v) for v in g.get('release_label',pd.Series(dtype=object)) if _norm_identity_text(v)))),
            'quarantined_external_aliases':'|'.join(sorted(quarantined_external)),
        })
    return pd.DataFrame(records),local




def _assay_source_identity_policy(
    release_frames: list[pd.DataFrame],
) -> dict:
    """Derive assay source-ID policy solely from residual strong conflicts.

    Workflow:
      1. build baseline assay occurrences without any source-ID policy;
      2. reproduce weak-alias quarantine;
      3. inspect only residual components containing >1 SRCASSAY alias;
      4. accept either:
           a) strict one-to-one temporal rename, or
           b) strict one-to-many partition quarantine;
      5. leave every other component untouched and therefore fail-closed.

    Activity, potency, target values and model outputs are never used.
    """

    # ========================================================
    # 1. BASELINE ASSAY OCCURRENCES
    # ========================================================

    occ_parts = []

    for frame_index, df in enumerate(
        release_frames
    ):

        if (
            not isinstance(df, pd.DataFrame)
            or df.empty
        ):
            continue

        occ, _ = _assay_occurrences(
            df,
            frame_index,
            source_policy={},
        )

        if not occ.empty:

            occ = occ.copy()

            occ[
                '_frame_index'
            ] = frame_index

            occ_parts.append(
                occ
            )


    if not occ_parts:

        return {
            'rename_map': {},
            'quarantined_aliases': set(),
            'policy_type_map': {},
            'family_map': {},
            'audit': pd.DataFrame(),
        }


    baseline_occ = pd.concat(
        occ_parts,
        ignore_index=True,
        sort=False,
    )


    # ========================================================
    # 2. REPRODUCE WEAK-ALIAS QUARANTINE, BUT RETURN THE
    #    RESIDUAL STRONG CONFLICTS INSTEAD OF RAISING.
    # ========================================================

    original = [
        sorted(
            set(
                str(x)
                for x in aliases
                if x
            )
        )
        for aliases
        in baseline_occ[
            'aliases'
        ].tolist()
    ]


    fallback_aliases = (
        _identity_fallback_aliases(
            baseline_occ,
            'assay',
        )
    )


    quarantined_weak = set()


    all_weak = sorted({
        alias
        for aliases in original
        for alias in aliases
        if _is_weak_alias(
            alias,
            'assay',
        )
    })


    residual = pd.DataFrame()


    for _iteration in range(
        len(all_weak) + 2
    ):

        active = []

        for i, aliases in enumerate(
            original
        ):

            aa = [
                alias
                for alias in aliases
                if not (
                    _is_weak_alias(
                        alias,
                        'assay',
                    )
                    and alias
                    in quarantined_weak
                )
            ]


            if not aa:

                aa = [
                    fallback_aliases[i]
                ]


            active.append(
                sorted(
                    set(
                        aa
                    )
                )
            )


        _keys, base = (
            _canonical_alias_components(
                active,
                'ASSAY',
            )
        )


        if base.empty:

            residual = base
            break


        bad = base[
            _external_conflict_mask(
                base,
                'assay',
            )
        ].copy()


        if bad.empty:

            residual = bad
            break


        weak_in_bad = set()


        for alias_text in bad[
            'aliases'
        ].fillna(''):

            for alias in str(
                alias_text
            ).split('|'):

                if (
                    alias
                    and _is_weak_alias(
                        alias,
                        'assay',
                    )
                ):

                    weak_in_bad.add(
                        alias
                    )


        new_weak = (
            weak_in_bad
            - quarantined_weak
        )


        if not new_weak:

            residual = bad
            break


        quarantined_weak.update(
            new_weak
        )


    else:

        raise RuntimeError(
            'Assay policy baseline weak-alias '
            'quarantine did not converge.'
        )


    if residual.empty:

        return {
            'rename_map': {},
            'quarantined_aliases': set(),
            'policy_type_map': {},
            'family_map': {},
            'audit': pd.DataFrame(),
        }


    # ========================================================
    # 3. HELPERS
    # ========================================================

    def parse_aliases(value):

        return {
            x
            for x in str(
                value
            ).split('|')
            if x
        }


    def parse_src_alias(alias):

        payload = alias[
            len('SRCASSAY:'):
        ]

        if ':' not in payload:
            return '', payload

        return payload.split(
            ':',
            1,
        )


    def pol_base(value):

        return re.sub(
            r'_POL_\d+$',
            '',
            value,
        )


    def occurrence_has_component_alias(
        aliases,
        component_aliases,
    ):

        return bool(
            set(
                aliases
            )
            & component_aliases
        )


    # ========================================================
    # 4. POLICY OUTPUT
    # ========================================================

    rename_map = {}

    quarantined_aliases = set()

    policy_type_map = {}

    family_map = {}

    audit_rows = []


    # ========================================================
    # 5. CLASSIFY ONLY REAL RESIDUAL COMPONENTS
    # ========================================================

    for _, bad_row in residual.iterrows():

        component_aliases = (
            parse_aliases(
                bad_row.get(
                    'aliases',
                    '',
                )
            )
        )


        src_aliases = sorted(
            alias
            for alias in component_aliases
            if alias.startswith(
                'SRCASSAY:'
            )
        )


        # No source conflict means no assay-source policy.
        if len(src_aliases) < 2:
            continue


        src_pairs = [
            parse_src_alias(
                alias
            )
            for alias in src_aliases
        ]


        source_ids = sorted({
            source_id
            for source_id, _
            in src_pairs
        })


        if len(source_ids) != 1:
            # Unknown conflict shape:
            # leave unchanged -> fail closed.
            continue


        source_id = source_ids[0]


        source_assay_ids = sorted({
            assay_id
            for _, assay_id
            in src_pairs
        })


        bases = sorted({
            pol_base(
                assay_id
            )
            for assay_id
            in source_assay_ids
        })


        if len(bases) != 1:
            continue


        base_id = bases[0]


        plain_ids = [
            x
            for x in source_assay_ids
            if x == base_id
        ]


        pol_ids = sorted(
            x
            for x in source_assay_ids
            if (
                x != base_id
                and re.fullmatch(
                    re.escape(
                        base_id
                    )
                    + r'_POL_\d+',
                    x,
                )
            )
        )


        if (
            len(plain_ids) != 1
            or not pol_ids
            or (
                len(plain_ids)
                + len(pol_ids)
                != len(
                    source_assay_ids
                )
            )
        ):

            continue


        plain_alias = (
            'SRCASSAY:'
            + source_id
            + ':'
            + base_id
        )


        family_id = (
            source_id
            + ':'
            + base_id
        )


        # ----------------------------------------------------
        # Occurrences belonging specifically to this residual
        # graph component.
        # ----------------------------------------------------

        comp_occ = baseline_occ[
            baseline_occ[
                'aliases'
            ].map(
                lambda aliases:
                    occurrence_has_component_alias(
                        aliases,
                        component_aliases,
                    )
            )
        ].copy()


        if comp_occ.empty:
            continue


        # ====================================================
        # A. ONE-TO-ONE TEMPORAL RENAME
        # ====================================================

        if len(pol_ids) == 1:

            pol_id = pol_ids[0]

            pol_alias = (
                'SRCASSAY:'
                + source_id
                + ':'
                + pol_id
            )


            allowed_aliases = {
                plain_alias,
                pol_alias,
            }


            states = []

            simultaneous = False


            for frame_index, fg in (
                comp_occ.groupby(
                    '_frame_index',
                    sort=True,
                )
            ):

                frame_src = sorted({
                    alias
                    for aliases
                    in fg[
                        'aliases'
                    ]
                    for alias in aliases
                    if alias in allowed_aliases
                })


                if len(frame_src) > 1:
                    simultaneous = True


                if len(frame_src) == 1:

                    states.append(
                        (
                            int(
                                frame_index
                            ),
                            frame_src[0],
                        )
                    )


            compressed = []

            for frame_index, state in states:

                if (
                    not compressed
                    or compressed[-1][1]
                    != state
                ):

                    compressed.append(
                        (
                            frame_index,
                            state,
                        )
                    )


            plain_frames = {
                frame
                for frame, state
                in states
                if state == plain_alias
            }


            pol_frames = {
                frame
                for frame, state
                in states
                if state == pol_alias
            }


            chembl_aliases = sorted(
                alias
                for alias
                in component_aliases
                if alias.startswith(
                    'CHEMBL_ASSAY:'
                )
            )


            # First establish the temporal SHAPE independently
            # of replicated-support strength.
            temporal_shape_pass = all([
                not simultaneous,

                len(compressed) == 2,

                (
                    len(compressed) == 2
                    and compressed[0][1]
                    == plain_alias
                    and compressed[1][1]
                    == pol_alias
                ),

                # The residual component itself identifies
                # exactly one internal assay lineage.
                len(chembl_aliases) == 1,
            ])


            replicated_support = all([
                len(plain_frames) >= 2,
                len(pol_frames) >= 2,
            ])


            temporal_pass = (
                temporal_shape_pass
                and replicated_support
            )


            # ------------------------------------------------
            # Valid 1->1 temporal shape, but insufficient
            # replicated historical support.
            #
            # Do NOT merge the competing external identifiers.
            # Quarantine both external aliases and retain an
            # explicit ambiguity marker for downstream exclusion.
            # ------------------------------------------------
            if (
                temporal_shape_pass
                and not replicated_support
            ):

                quarantined_aliases.update([
                    plain_alias,
                    pol_alias,
                ])


                for raw_alias in [
                    plain_alias,
                    pol_alias,
                ]:

                    policy_type_map[
                        raw_alias
                    ] = 'AMBIGUOUS_TEMPORAL_RENAME'

                    family_map[
                        raw_alias
                    ] = family_id


                audit_rows.append({

                    'assay_source_policy':
                        'AMBIGUOUS_TEMPORAL_RENAME',

                    'source_id':
                        source_id,

                    'base_source_assay_id':
                        base_id,

                    'n_pol_descendants':
                        1,

                    'old_source_alias':
                        plain_alias,

                    'new_source_aliases':
                        pol_alias,

                    'canonical_source_alias':
                        '',

                    'n_old_state_frames':
                        len(
                            plain_frames
                        ),

                    'n_new_state_frames':
                        len(
                            pol_frames
                        ),

                    'n_chembl_assay_ids':
                        1,

                    'policy_family':
                        family_id,
                })

                continue


            if not temporal_pass:
                # Any other temporal shape remains completely
                # unresolved and therefore fail-closed.
                continue


            canonical_alias = (
                _identity_sha(
                    'SRCASSAY:',
                    [
                        'TEMPORAL_RENAME',
                        source_id,
                        base_id,
                    ],
                )
            )


            rename_map[
                plain_alias
            ] = canonical_alias

            rename_map[
                pol_alias
            ] = canonical_alias


            for raw_alias in [
                plain_alias,
                pol_alias,
            ]:

                policy_type_map[
                    raw_alias
                ] = 'TEMPORAL_RENAME'

                family_map[
                    raw_alias
                ] = family_id


            audit_rows.append({

                'assay_source_policy':
                    'TEMPORAL_RENAME',

                'source_id':
                    source_id,

                'base_source_assay_id':
                    base_id,

                'n_pol_descendants':
                    1,

                'old_source_alias':
                    plain_alias,

                'new_source_aliases':
                    pol_alias,

                'canonical_source_alias':
                    canonical_alias,

                'n_old_state_frames':
                    len(
                        plain_frames
                    ),

                'n_new_state_frames':
                    len(
                        pol_frames
                    ),

                'n_chembl_assay_ids':
                    1,

                'policy_family':
                    family_id,
            })


            continue


        # ====================================================
        # B. ONE-TO-MANY PARTITION
        # ====================================================

        # Mapping is evaluated only inside this residual
        # component, not across unrelated raw archive families.

        pol_to_chem = {}


        for pol_id in pol_ids:

            pol_alias = (
                'SRCASSAY:'
                + source_id
                + ':'
                + pol_id
            )


            hits = comp_occ[
                comp_occ[
                    'aliases'
                ].map(
                    lambda aliases:
                        pol_alias
                        in set(
                            aliases
                        )
                )
            ]


            chem_ids = sorted({
                alias[
                    len(
                        'CHEMBL_ASSAY:'
                    ):
                ]
                for aliases
                in hits[
                    'aliases'
                ]
                for alias in aliases
                if alias.startswith(
                    'CHEMBL_ASSAY:'
                )
            })


            pol_to_chem[
                pol_id
            ] = chem_ids


        each_pol_one_chem = all(
            len(chem_ids) == 1
            for chem_ids
            in pol_to_chem.values()
        )


        chem_to_pol = {}


        if each_pol_one_chem:

            for pol_id, chem_ids in (
                pol_to_chem.items()
            ):

                chem_id = (
                    chem_ids[0]
                )

                chem_to_pol.setdefault(
                    chem_id,
                    set(),
                ).add(
                    pol_id
                )


        each_chem_one_pol = (
            bool(chem_to_pol)
            and all(
                len(pol_set) == 1
                for pol_set
                in chem_to_pol.values()
            )
        )


        component_chembl_ids = {
            alias[
                len(
                    'CHEMBL_ASSAY:'
                ):
            ]
            for alias
            in component_aliases
            if alias.startswith(
                'CHEMBL_ASSAY:'
            )
        }


        mapped_chembl_ids = set(
            chem_to_pol.keys()
        )


        partition_pass = all([
            len(pol_ids) >= 2,

            each_pol_one_chem,

            each_chem_one_pol,

            len(
                component_chembl_ids
            ) == len(
                pol_ids
            ),

            mapped_chembl_ids
            == component_chembl_ids,
        ])


        if not partition_pass:
            # No exception.
            continue


        # Only the coarse/plain source-assay alias is
        # quarantined. POL descendants remain independent
        # strong source identities.
        quarantined_aliases.add(
            plain_alias
        )


        policy_type_map[
            plain_alias
        ] = 'PARTITION_QUARANTINE'

        family_map[
            plain_alias
        ] = family_id


        for pol_id in pol_ids:

            pol_alias = (
                'SRCASSAY:'
                + source_id
                + ':'
                + pol_id
            )

            policy_type_map[
                pol_alias
            ] = 'PARTITION_DESCENDANT'

            family_map[
                pol_alias
            ] = family_id


        audit_rows.append({

            'assay_source_policy':
                'PARTITION_QUARANTINE',

            'source_id':
                source_id,

            'base_source_assay_id':
                base_id,

            'n_pol_descendants':
                len(
                    pol_ids
                ),

            'old_source_alias':
                plain_alias,

            'new_source_aliases':
                '|'.join(
                    'SRCASSAY:'
                    + source_id
                    + ':'
                    + pol_id
                    for pol_id
                    in pol_ids
                ),

            'canonical_source_alias':
                '',

            'n_old_state_frames':
                int(
                    comp_occ[
                        'aliases'
                    ].map(
                        lambda aliases:
                            plain_alias
                            in set(
                                aliases
                            )
                    ).groupby(
                        comp_occ[
                            '_frame_index'
                        ]
                    ).any().sum()
                ),

            'n_new_state_frames':
                int(
                    comp_occ[
                        'aliases'
                    ].map(
                        lambda aliases:
                            any(
                                (
                                    'SRCASSAY:'
                                    + source_id
                                    + ':'
                                    + pol_id
                                )
                                in set(
                                    aliases
                                )
                                for pol_id
                                in pol_ids
                            )
                    ).groupby(
                        comp_occ[
                            '_frame_index'
                        ]
                    ).any().sum()
                ),

            'n_chembl_assay_ids':
                len(
                    component_chembl_ids
                ),

            'policy_family':
                family_id,
        })


    return {

        'rename_map':
            rename_map,

        'quarantined_aliases':
            quarantined_aliases,

        'policy_type_map':
            policy_type_map,

        'family_map':
            family_map,

        'audit':
            pd.DataFrame(
                audit_rows
            ),
    }

def _augment_assay_source_policy_audit(
    base: pd.DataFrame,
    occurrences: pd.DataFrame,
    canonical: list[str],
) -> pd.DataFrame:
    """Attach source-assay rename/quarantine evidence to assay audit."""

    if base.empty:
        return base


    tmp = occurrences.copy()

    tmp['canonical_key'] = canonical


    def flatten(values):
        out = set()

        for value in values:

            if isinstance(
                value,
                (list, tuple, set),
            ):
                out.update(
                    x
                    for x in value
                    if x
                )

            elif value:
                out.add(
                    str(value)
                )

        return sorted(out)


    rows = []

    for key, g in tmp.groupby(
        'canonical_key',
        sort=False,
    ):

        policy_types = flatten(
            g.get(
                'assay_source_policy_types',
                pd.Series(dtype=object),
            )
        )

        policy_families = flatten(
            g.get(
                'assay_source_policy_families',
                pd.Series(dtype=object),
            )
        )

        quarantined = flatten(
            g.get(
                'quarantined_external_aliases',
                pd.Series(dtype=object),
            )
        )

        renamed = flatten(
            g.get(
                'renamed_external_aliases',
                pd.Series(dtype=object),
            )
        )


        rows.append({
            'canonical_key':
                key,

            'assay_source_policy_types':
                '|'.join(
                    policy_types
                ),

            'assay_source_policy_families':
                '|'.join(
                    policy_families
                ),

            'quarantined_external_aliases':
                '|'.join(
                    quarantined
                ),

            'n_quarantined_external_aliases':
                len(
                    quarantined
                ),

            'renamed_external_aliases':
                '|'.join(
                    renamed
                ),

            'n_renamed_external_aliases':
                len(
                    renamed
                ),
        })


    extra = pd.DataFrame(
        rows
    )


    return base.merge(
        extra,
        on='canonical_key',
        how='left',
        validate='one_to_one',
    )


def _assay_local_keys(
    df: pd.DataFrame,
    frame_index: int,
    source_policy: dict | None=None,
) -> pd.Series:

    doc = _series(
        df,
        'document_identity_key',
    )

    desc = _series(
        df,
        'assay_description',
    )

    typ = _series(
        df,
        'assay_type',
    )

    target = _series(
        df,
        'target_accession',
    )

    chem = _series(
        df,
        'assay_chembl_id',
    )

    src = _series(
        df,
        'assay_source_id',
    )

    srcassay = _series(
        df,
        'assay_source_assay_id',
    )

    native = _series(
        df,
        'native_assay_id',
    )

    release = _series(
        df,
        'release_label',
    )


    source_policy = (
        source_policy
        or {}
    )

    rename_map = (
        source_policy.get(
            'rename_map',
            {},
        )
    )

    quarantined = set(
        source_policy.get(
            'quarantined_aliases',
            set(),
        )
    )


    sem_ok = (
        desc != ''
    )

    sem_raw = (
        'SEMRAW:'
        + doc
        + '|'
        + desc
        + '|'
        + typ
        + '|'
        + target
    )


    raw_source_alias = (
        'SRCASSAY:'
        + src
        + ':'
        + srcassay
    )


    mapped_source_alias = (
        raw_source_alias.map(
            lambda x:
                rename_map.get(
                    x,
                    x,
                )
        )
    )


    out = pd.Series(
        '',
        index=df.index,
        dtype='object',
    )


    # Strong source-assay identity, unless the raw historical
    # alias has been explicitly quarantined as non-injective.
    m = (
        src.ne('')
        & srcassay.ne('')
        & ~raw_source_alias.isin(
            quarantined
        )
    )

    out.loc[m] = (
        mapped_source_alias.loc[m]
    )


    m = (
        out.eq('')
        & chem.ne('')
    )

    out.loc[m] = (
        'CHEMBL_ASSAY:'
        + chem.loc[m]
    )


    m = (
        out.eq('')
        & sem_ok
    )

    out.loc[m] = (
        sem_raw.loc[m]
    )


    m = (
        out.eq('')
        & native.ne('')
    )

    out.loc[m] = (
        'LOCAL_ASSAY_ID:'
        + release.loc[m]
        + ':'
        + native.loc[m]
    )


    if out.eq('').any():

        pos = pd.Series(
            range(len(df)),
            index=df.index,
        ).astype(str)

        m = out.eq('')

        out.loc[m] = (
            'LOCAL_ASSAY_ROW:'
            + str(frame_index)
            + ':'
            + pos.loc[m]
        )


    return out


def _assay_occurrences(
    df: pd.DataFrame,
    frame_index: int,
    source_policy: dict | None=None,
) -> tuple[pd.DataFrame, pd.Series]:

    source_policy = (
        source_policy
        or {}
    )

    rename_map = (
        source_policy.get(
            'rename_map',
            {},
        )
    )

    quarantined = set(
        source_policy.get(
            'quarantined_aliases',
            set(),
        )
    )

    policy_type_map = (
        source_policy.get(
            'policy_type_map',
            {},
        )
    )

    family_map = (
        source_policy.get(
            'family_map',
            {},
        )
    )


    local = _assay_local_keys(
        df,
        frame_index,
        source_policy=source_policy,
    )


    cols = (
        ['document_identity_key']
        + [
            c
            for c in ASSAY_ALIAS_FIELDS
            if c in df.columns
        ]
    )


    slim = df[
        cols
    ].copy()

    slim[
        '_local_assay_occurrence'
    ] = local.values


    slim = (
        slim
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )


    records = []


    for occ, g in slim.groupby(
        '_local_assay_occurrence',
        sort=False,
        dropna=False,
    ):

        aliases = set()

        quarantined_here = set()

        renamed_here = set()

        policy_types_here = set()

        policy_families_here = set()


        for _, r in g.iterrows():

            desc = _norm_identity_text(
                r.get(
                    'assay_description'
                )
            )


            if desc:

                sem_parts = [
                    _norm_identity_text(
                        r.get(
                            'document_identity_key'
                        )
                    ),

                    desc,

                    _norm_identity_text(
                        r.get(
                            'assay_type'
                        )
                    ),

                    _norm_identity_text(
                        r.get(
                            'target_accession'
                        )
                    ),
                ]


                aliases.add(
                    _identity_sha(
                        'ASSAY_SEM:',
                        sem_parts,
                    )
                )


            chem = _norm_identity_text(
                r.get(
                    'assay_chembl_id'
                )
            )


            src = _norm_identity_text(
                r.get(
                    'assay_source_id'
                )
            )

            srcassay = _norm_identity_text(
                r.get(
                    'assay_source_assay_id'
                )
            )


            if chem:

                aliases.add(
                    'CHEMBL_ASSAY:'
                    + chem
                )


            if src and srcassay:

                raw_alias = (
                    'SRCASSAY:'
                    + src
                    + ':'
                    + srcassay
                )


                ptype = policy_type_map.get(
                    raw_alias,
                    '',
                )

                family = family_map.get(
                    raw_alias,
                    '',
                )


                if ptype:
                    policy_types_here.add(
                        ptype
                    )

                if family:
                    policy_families_here.add(
                        family
                    )


                if raw_alias in quarantined:

                    quarantined_here.add(
                        raw_alias
                    )

                else:

                    active_alias = (
                        rename_map.get(
                            raw_alias,
                            raw_alias,
                        )
                    )

                    aliases.add(
                        active_alias
                    )


                    if active_alias != raw_alias:

                        renamed_here.add(
                            raw_alias
                            + '=>'
                            + active_alias
                        )


            existing = _norm_identity_text(
                r.get(
                    'assay_identity_key'
                )
            )


            if existing:

                aliases.add(
                    'EXISTING_ASSAY:'
                    + existing
                )


        if not aliases:

            aliases.add(
                'LOCAL_ASSAY_OCC:'
                + str(occ)
            )


        types = sorted(
            set(
                _norm_identity_text(v)
                for v in g.get(
                    'assay_type',
                    pd.Series(
                        dtype=object
                    ),
                )
                if _norm_identity_text(v)
            )
        )


        records.append({

            'occurrence_key':
                str(occ),

            'aliases':
                sorted(
                    aliases
                ),

            'assay_types':
                types,

            'schema_generation':
                '|'.join(
                    sorted(
                        set(
                            _norm_identity_text(v)
                            for v in g.get(
                                'schema_generation',
                                pd.Series(
                                    dtype=object
                                ),
                            )
                            if _norm_identity_text(v)
                        )
                    )
                ),

            'release_label':
                '|'.join(
                    sorted(
                        set(
                            _norm_identity_text(v)
                            for v in g.get(
                                'release_label',
                                pd.Series(
                                    dtype=object
                                ),
                            )
                            if _norm_identity_text(v)
                        )
                    )
                ),

            'quarantined_external_aliases':
                sorted(
                    quarantined_here
                ),

            'renamed_external_aliases':
                sorted(
                    renamed_here
                ),

            'assay_source_policy_types':
                sorted(
                    policy_types_here
                ),

            'assay_source_policy_families':
                sorted(
                    policy_families_here
                ),
        })


    return (
        pd.DataFrame(
            records
        ),
        local,
    )

def _augment_component_audit(base: pd.DataFrame, occurrences: pd.DataFrame, canonical: list[str], kind: str) -> pd.DataFrame:
    if base.empty:
        return base
    occ=occurrences.copy(); occ['canonical_key']=canonical
    def join_unique(series):
        vals=set()
        for v in series:
            for x in str(v).split('|'):
                if x: vals.add(x)
        return '|'.join(sorted(vals))
    agg=occ.groupby('canonical_key',as_index=False).agg(
        n_occurrences=('occurrence_key','size'),
        release_labels=('release_label',join_unique),
        schema_generations=('schema_generation',join_unique),
    )
    out=base.merge(agg,on='canonical_key',how='left',validate='one_to_one')

    if kind=='document' and 'quarantined_external_aliases' in occ.columns:
        qext=(
            occ.groupby('canonical_key',as_index=False)
            .agg(
                quarantined_external_aliases=(
                    'quarantined_external_aliases',
                    join_unique,
                )
            )
        )
        qext['n_quarantined_external_aliases']=(
            qext['quarantined_external_aliases']
            .fillna('')
            .map(lambda s:len([x for x in s.split('|') if x]))
        )
        out=out.merge(
            qext,
            on='canonical_key',
            how='left',
            validate='one_to_one',
        )
        out['quarantined_external_aliases']=(
            out['quarantined_external_aliases'].fillna('')
        )
        out['n_quarantined_external_aliases']=(
            out['n_quarantined_external_aliases']
            .fillna(0)
            .astype(int)
        )
    elif kind=='document':
        out['quarantined_external_aliases']=''
        out['n_quarantined_external_aliases']=0
    out['internal_id_multiplicity']=out['n_chembl_aliases'].gt(1)
    if kind=='document':
        ext=out['n_doi_aliases'].gt(0)|out['n_pmid_aliases'].gt(0)
        out['weak_only_component']=out['n_weak_aliases'].gt(0)&~ext
    else:
        ext=out['n_source_aliases'].gt(0)
        out['weak_only_component']=out['n_weak_aliases'].gt(0)&~ext
        types=occ.explode('assay_types').groupby('canonical_key')['assay_types'].agg(lambda s:'|'.join(sorted(set(str(x) for x in s if isinstance(x,str) and x)))).rename('assay_types').reset_index()
        out=out.merge(types,on='canonical_key',how='left',validate='one_to_one')
        out['n_assay_types']=out['assay_types'].fillna('').map(lambda s:len([x for x in s.split('|') if x]))
        out['assay_type_drift']=out['n_assay_types'].gt(1)
    out['internal_id_multiplicity_external_supported']=out['internal_id_multiplicity']&ext
    out['internal_id_multiplicity_weak_only']=out['internal_id_multiplicity']&~ext
    # Symmetric under-merge diagnostic: every alias in the component is release-scoped.
    out['release_local_only_component']=out['n_aliases'].gt(0)&out['n_local_aliases'].eq(out['n_aliases'])
    return out


def _schema_generations(audit: pd.DataFrame) -> list[str]:
    return sorted({g for s in audit.get('schema_generations',pd.Series(dtype=object)).fillna('')
                   for g in str(s).split('|') if g}) or ['UNKNOWN']


def _release_local_policy_table(audit: pd.DataFrame, kind: str, max_fraction: float, min_releases: int=2) -> pd.DataFrame:
    rows=[]
    if audit.empty:
        return pd.DataFrame(columns=['kind','schema_generation','n_components','n_release_local_only_components',
                                     'release_local_only_fraction','n_processed_releases_in_family','min_releases_for_policy','policy_evaluable',
                                     'max_fraction','policy_pass'])
    for gen in _schema_generations(audit):
        sub=(audit[audit['schema_generations'].fillna('').map(lambda s: gen in str(s).split('|'))]
             if gen!='UNKNOWN' else audit)
        n=len(sub); k=int(sub['release_local_only_component'].fillna(False).sum())
        frac=k/n if n else 0.0
        rels=sorted({r for x in sub.get('release_labels',pd.Series(dtype=object)).fillna('')
                     for r in str(x).split('|') if r})
        evaluable=len(rels)>=int(min_releases)
        rows.append({'kind':kind,'schema_generation':gen,'n_components':n,
                     'n_release_local_only_components':k,'release_local_only_fraction':frac,
                     'n_processed_releases_in_family':len(rels),'min_releases_for_policy':int(min_releases),'policy_evaluable':bool(evaluable),
                     'max_fraction':float(max_fraction),
                     'policy_pass':bool((not evaluable) or frac<=float(max_fraction))})
    return pd.DataFrame(rows)


def _annotate_release_local_family_metrics(audit: pd.DataFrame, kind: str, max_fraction: float, min_releases: int=2) -> pd.DataFrame:
    if audit.empty:
        return audit
    out=audit.copy()
    policy=_release_local_policy_table(out,kind,max_fraction,min_releases)
    by_gen={r.schema_generation:r for r in policy.itertuples(index=False)}
    counts=[]; fracs=[]; passes=[]
    for gens in out['schema_generations'].fillna(''):
        gg=[g for g in str(gens).split('|') if g] or ['UNKNOWN']
        rs=[by_gen[g] for g in gg if g in by_gen]
        counts.append(max((int(r.n_release_local_only_components) for r in rs),default=0))
        fracs.append(max((float(r.release_local_only_fraction) for r in rs),default=0.0))
        passes.append(all(bool(r.policy_pass) for r in rs) if rs else True)
    out['n_release_local_only_components']=counts
    out['release_local_only_fraction']=fracs
    out['release_local_identity_policy_pass']=passes
    return out


def _check_release_local_identity_policy(audit: pd.DataFrame, kind: str, max_fraction: float, min_releases: int=2):
    policy=_release_local_policy_table(audit,kind,max_fraction,min_releases)
    bad=policy[~policy.policy_pass] if not policy.empty else policy
    if not bad.empty:
        raise HistoricalIdentityConflictError(
            f'{kind}: release-local-only identity components exceed preregistered fraction {max_fraction:.4f}.',bad
        )


def release_local_identity_rate_by_schema(
    doc_audit: pd.DataFrame, assay_audit: pd.DataFrame, max_fraction: float=0.01, min_releases: int=2
) -> pd.DataFrame:
    parts=[_release_local_policy_table(doc_audit,'document',max_fraction,min_releases),
           _release_local_policy_table(assay_audit,'assay',max_fraction,min_releases)]
    parts=[x for x in parts if not x.empty]
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(
        columns=['kind','schema_generation','n_components','n_release_local_only_components',
                 'release_local_only_fraction','n_processed_releases_in_family','min_releases_for_policy','policy_evaluable',
                 'max_fraction','policy_pass'])



def _fail_external_conflicts(
    audit: pd.DataFrame,
    kind: str,
):
    """Fail closed on unresolved external-identity conflicts."""

    if audit.empty:
        return

    if kind == 'document':

        conflict = (
            audit['n_doi_aliases'].gt(1)
            | audit['n_pmid_aliases'].gt(1)
        )

        if (
            'temporal_recuration_pass'
            in audit.columns
        ):
            temporal_ok = (
                audit[
                    'temporal_recuration_pass'
                ]
                .fillna(False)
                .astype(bool)
            )
        else:
            temporal_ok = pd.Series(
                False,
                index=audit.index,
                dtype=bool,
            )

        bad = audit[
            conflict
            & ~temporal_ok
        ]

        msg = (
            'Conflicting external document aliases joined one '
            'historical identity component without a validated '
            'temporal-recuration exception.'
        )

    else:

        bad = audit[
            audit[
                'n_source_aliases'
            ].gt(1)
        ]

        msg = (
            'Conflicting external source-assay aliases joined '
            'one historical identity component.'
        )


    if not bad.empty:

        raise HistoricalIdentityConflictError(
            msg,
            bad,
        )

def _internal_multiplicity_policy_table(
    audit: pd.DataFrame,
    kind: str,
    max_fraction: float,
) -> pd.DataFrame:
    """Return the preregistered weak-only internal-ID multiplicity audit table."""

    columns = [
        'kind',
        'schema_generation',
        'n_components_total',
        'n_overmerge_denominator_components',
        'n_weak_only_internal_multiplicity',
        'fraction',
        'max_fraction',
    ]

    if audit.empty:
        return pd.DataFrame(columns=columns)

    gens = _schema_generations(audit)

    rows = []

    for gen in gens:

        if gen != 'UNKNOWN':
            sub = audit[
                audit['schema_generations']
                .fillna('')
                .map(
                    lambda s:
                        gen in str(s).split('|')
                )
            ]
        else:
            sub = audit

        # Release-local-only components cannot exhibit meaningful
        # cross-release internal-ID multiplicity and would mechanically
        # dilute the over-merge denominator.
        release_local = (
            sub['release_local_only_component']
            .eq(True)
        )

        denom = sub[
            ~release_local
        ]

        n = len(denom)

        suspicious = int(
            denom[
                'internal_id_multiplicity_weak_only'
            ]
            .eq(True)
            .sum()
        )

        frac = (
            suspicious / n
            if n
            else 0.0
        )

        rows.append({
            'kind':
                kind,

            'schema_generation':
                gen,

            'n_components_total':
                len(sub),

            'n_overmerge_denominator_components':
                n,

            'n_weak_only_internal_multiplicity':
                suspicious,

            'fraction':
                frac,

            'max_fraction':
                float(max_fraction),
        })

    return pd.DataFrame(
        rows,
        columns=columns,
    )


def _check_internal_multiplicity_policy(
    audit: pd.DataFrame,
    kind: str,
    max_fraction: float,
):
    """Fail closed when preregistered weak-only multiplicity exceeds threshold."""

    policy = _internal_multiplicity_policy_table(
        audit,
        kind,
        max_fraction,
    )

    if policy.empty:
        return

    bad = policy[
        policy['fraction']
        > float(max_fraction)
    ]

    if not bad.empty:

        raise HistoricalIdentityConflictError(
            f'{kind}: weak-only internal ChEMBL-ID multiplicity '
            f'exceeds preregistered fraction {max_fraction:.4f}.',
            bad,
        )


def identity_reconciliation_policy_summary(
    doc_audit: pd.DataFrame, assay_audit: pd.DataFrame,
    max_fraction: float=0.01, release_local_max_fraction: float=0.01,
    release_local_min_releases: int=2
) -> pd.DataFrame:
    rows=[]
    for kind,audit in [('document',doc_audit),('assay',assay_audit)]:
        if audit.empty: continue
        for gen in _schema_generations(audit):
            sub=audit[audit['schema_generations'].fillna('').map(lambda s: gen in str(s).split('|'))] if gen!='UNKNOWN' else audit
            n=len(sub)
            nonlocal_sub=sub[~sub['release_local_only_component'].fillna(False)]
            over_denom=len(nonlocal_sub)
            over_num=int(nonlocal_sub.internal_id_multiplicity_weak_only.sum()) if over_denom else 0
            over_frac=over_num/over_denom if over_denom else 0.0
            local_num=int(sub.release_local_only_component.fillna(False).sum())
            local_frac=local_num/n if n else 0.0
            rels=sorted({r for x in sub.get('release_labels',pd.Series(dtype=object)).fillna('')
                         for r in str(x).split('|') if r})
            local_evaluable=len(rels)>=int(release_local_min_releases)
            rows.append({
                'kind':kind,'schema_generation':gen,'n_components':n,
                'n_internal_id_multiplicity':int(sub.internal_id_multiplicity.sum()),
                'n_external_supported_internal_multiplicity':int(sub.internal_id_multiplicity_external_supported.sum()),
                'n_weak_only_internal_multiplicity':int(sub.internal_id_multiplicity_weak_only.sum()),
                'n_overmerge_denominator_components':over_denom,
                'weak_only_internal_multiplicity_fraction':over_frac,
                'internal_multiplicity_max_fraction':float(max_fraction),
                'internal_multiplicity_policy_pass':bool(over_frac<=float(max_fraction)),
                'n_weak_only_components':int(sub.weak_only_component.sum()),
                'weak_only_component_fraction':float(sub.weak_only_component.mean()) if n else 0.0,
                'n_components_with_quarantined_weak_aliases':int(sub.get('n_quarantined_weak_aliases',pd.Series(0,index=sub.index)).fillna(0).gt(0).sum()),
                'n_quarantined_weak_aliases':int(sub.get('n_quarantined_weak_aliases',pd.Series(0,index=sub.index)).fillna(0).sum()),
                'n_release_local_only_components':local_num,
                'release_local_only_fraction':local_frac,
                'release_local_max_fraction':float(release_local_max_fraction),
                'n_processed_releases_in_family':len(rels),
                'release_local_min_releases':int(release_local_min_releases),
                'release_local_policy_evaluable':bool(local_evaluable),
                'release_local_policy_pass':bool((not local_evaluable) or local_frac<=float(release_local_max_fraction)),
                'policy_pass':bool(over_frac<=float(max_fraction) and ((not local_evaluable) or local_frac<=float(release_local_max_fraction))),
            })
    return pd.DataFrame(rows)

def identity_component_size_distribution(doc_audit: pd.DataFrame, assay_audit: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for kind,audit in [('document',doc_audit),('assay',assay_audit)]:
        if audit.empty: continue
        gens=sorted({g for s in audit.schema_generations.fillna('') for g in str(s).split('|') if g}) or ['UNKNOWN']
        for gen in gens:
            sub=audit[audit.schema_generations.fillna('').map(lambda s:gen in str(s).split('|'))] if gen!='UNKNOWN' else audit
            x=sub.groupby('n_occurrences').size().rename('n_components').reset_index()
            x['kind']=kind; x['schema_generation']=gen; rows.append(x)
    return pd.concat(rows,ignore_index=True)[['kind','schema_generation','n_occurrences','n_components']] if rows else pd.DataFrame(columns=['kind','schema_generation','n_occurrences','n_components'])


def weak_alias_document_rate_by_schema(doc_audit: pd.DataFrame) -> pd.DataFrame:
    if doc_audit.empty:
        return pd.DataFrame(columns=['schema_generation','n_components','n_weak_only_components','weak_only_fraction'])
    gens=sorted({g for s in doc_audit.schema_generations.fillna('') for g in str(s).split('|') if g}) or ['UNKNOWN']
    rows=[]
    for gen in gens:
        sub=doc_audit[doc_audit.schema_generations.fillna('').map(lambda s:gen in str(s).split('|'))] if gen!='UNKNOWN' else doc_audit
        rows.append({'schema_generation':gen,'n_components':len(sub),'n_weak_only_components':int(sub.weak_only_component.sum()),'weak_only_fraction':float(sub.weak_only_component.mean()) if len(sub) else 0.0})
    return pd.DataFrame(rows)


def canonicalize_cross_release_identities(
    release_frames: list[pd.DataFrame],
    internal_id_weak_only_max_fraction: float=0.01,
    release_local_only_max_fraction: float=0.01,
    release_local_policy_min_releases: int=2,
    copy_frames: bool=True,
):
    """Reconcile document/assay identities using *unique metadata occurrences*, never activity rows.

    Native ChEMBL document/assay IDs are internal aliases: renumbering/merging is logged, not fatal.
    Conflicting external DOI/PMID/source-assay IDs remain fail-closed. The implementation avoids a
    full historical activity-frame concat and never includes ``target_sequence`` in reconciliation.
    """
    if not release_frames:
        return [],pd.DataFrame(),pd.DataFrame()
    frames=[df.copy() for df in release_frames] if copy_frames else release_frames
    if all(df.empty for df in frames):
        for df in frames:
            df['historical_identity_canonicalized']=True
        return frames,pd.DataFrame(),pd.DataFrame()

    # ---- document pass: compact metadata occurrences only ----
    occ_parts=[]
    for fi,df in enumerate(frames):
        if df.empty: continue
        occ,_=_document_occurrences(df,fi)
        if not occ.empty:
            occ['_frame_index']=fi; occ_parts.append(occ)
    doc_occ=pd.concat(occ_parts,ignore_index=True) if occ_parts else pd.DataFrame()
    if doc_occ.empty:
        raise HistoricalIdentityConflictError('No auditable historical document metadata were available.',pd.DataFrame())
    doc_temporal_context=_document_temporal_recuration_context(frames)
    doc_keys,doc_base=_protected_alias_components(
        doc_occ['aliases'].tolist(),
        'DOC',
        'document',
        _identity_fallback_aliases(doc_occ,'document'),
        document_temporal_context=doc_temporal_context,
    )
    doc_audit=_augment_component_audit(doc_base,doc_occ,doc_keys,'document')
    doc_audit=_annotate_release_local_family_metrics(doc_audit,'document',release_local_only_max_fraction,release_local_policy_min_releases)
    _fail_external_conflicts(doc_audit,'document')
    _check_release_local_identity_policy(doc_audit,'document',release_local_only_max_fraction,release_local_policy_min_releases)
    _check_internal_multiplicity_policy(doc_audit,'document',internal_id_weak_only_max_fraction)
    doc_occ['document_identity_key']=doc_keys
    doc_map=dict(zip(doc_occ.occurrence_key,doc_occ.document_identity_key))
    for fi,df in enumerate(frames):
        if df.empty:
            df['document_identity_key']=pd.Series(index=df.index,dtype='object'); continue
        quarantined_dois=_document_local_doi_collision_quarantine(df)
        local=_document_local_keys(
            df,
            fi,
            quarantined_dois=quarantined_dois,
        )
        df['document_identity_key']=local.map(doc_map)
        if df['document_identity_key'].isna().any():
            raise HistoricalIdentityConflictError('Document occurrence could not be mapped back to a canonical identity.',df.loc[df.document_identity_key.isna()].head(50))

    # ---- assay pass: compact metadata occurrences only ----
    #
    # Historical source-assay policy is derived BEFORE graph construction.
    # Strict 1->1 temporal renames are canonicalized to a synthetic source
    # identity; strict 1->N partitions quarantine only the non-injective
    # plain source alias. All other conflicts remain fail-closed.
    assay_source_policy=_assay_source_identity_policy(frames)

    occ_parts=[]
    for fi,df in enumerate(frames):
        if df.empty: continue
        occ,_=_assay_occurrences(
            df,
            fi,
            source_policy=assay_source_policy,
        )
        if not occ.empty:
            occ['_frame_index']=fi; occ_parts.append(occ)

    assay_occ=pd.concat(occ_parts,ignore_index=True) if occ_parts else pd.DataFrame()

    if assay_occ.empty:
        raise HistoricalIdentityConflictError(
            'No auditable historical assay metadata were available.',
            pd.DataFrame(),
        )

    # --------------------------------------------------------
    # Weak-only internal ChEMBL-ID multiplicity remediation.
    #
    # External source-assay policy has already been frozen above.
    # Here we address only weak semantic bridges when their
    # schema-family rate exceeds the preregistered safeguard.
    # --------------------------------------------------------

    forced_weak_quarantine: set[str] = set()

    while True:

        assay_keys,assay_base=_protected_alias_components(
            assay_occ['aliases'].tolist(),
            'ASSAY',
            'assay',
            _identity_fallback_aliases(
                assay_occ,
                'assay',
            ),
            initial_quarantined_weak_aliases=
                forced_weak_quarantine,
        )

        assay_audit=_augment_component_audit(
            assay_base,
            assay_occ,
            assay_keys,
            'assay',
        )

        assay_audit=_augment_assay_source_policy_audit(
            assay_audit,
            assay_occ,
            assay_keys,
        )

        assay_audit=_annotate_release_local_family_metrics(
            assay_audit,
            'assay',
            release_local_only_max_fraction,
            release_local_policy_min_releases,
        )

        # Source-assay hard constraints remain mandatory.
        _fail_external_conflicts(
            assay_audit,
            'assay',
        )

        multiplicity_policy=(
            _internal_multiplicity_policy_table(
                assay_audit,
                'assay',
                internal_id_weak_only_max_fraction,
            )
        )

        offending_generations=set(
            multiplicity_policy.loc[
                multiplicity_policy[
                    'fraction'
                ].gt(
                    float(
                        internal_id_weak_only_max_fraction
                    )
                ),
                'schema_generation',
            ].astype(str)
        )

        if not offending_generations:
            break

        # Identify only suspicious components participating
        # in schema generations that currently exceed the
        # preregistered family-level threshold.
        def _touches_offending_generation(value):

            gs={
                x
                for x in str(
                    value
                    if value is not None
                    else ''
                ).split('|')
                if x
            }

            return bool(
                gs
                & offending_generations
            )


        suspicious_mask=(
            assay_audit[
                'internal_id_multiplicity_weak_only'
            ].eq(True)
            &
            ~assay_audit[
                'release_local_only_component'
            ].eq(True)
            &
            assay_audit[
                'schema_generations'
            ].map(
                _touches_offending_generation
            )
        )

        suspicious_keys=set(
            assay_audit.loc[
                suspicious_mask,
                'canonical_key',
            ].astype(str)
        )

        if not suspicious_keys:
            # Existing final safeguard will report the
            # preregistered-policy failure.
            break

        occ_key_series=pd.Series(
            assay_keys,
            index=assay_occ.index,
            dtype='object',
        )

        weak_candidates=set()

        for aliases in assay_occ.loc[
            occ_key_series.isin(
                suspicious_keys
            ),
            'aliases',
        ]:

            weak_candidates.update(
                a
                for a in aliases
                if (
                    a
                    and _is_weak_alias(
                        a,
                        'assay',
                    )
                )
            )

        new_weak=(
            weak_candidates
            - forced_weak_quarantine
        )

        if not new_weak:
            # Nothing else can be removed safely.
            # Do not relax the threshold; final safeguard
            # remains fail-closed.
            break

        forced_weak_quarantine.update(
            new_weak
        )


    _check_release_local_identity_policy(
        assay_audit,
        'assay',
        release_local_only_max_fraction,
        release_local_policy_min_releases,
    )

    _check_internal_multiplicity_policy(
        assay_audit,
        'assay',
        internal_id_weak_only_max_fraction,
    )

    assay_occ['assay_identity_key']=assay_keys

    assay_map=dict(
        zip(
            assay_occ.occurrence_key,
            assay_occ.assay_identity_key,
        )
    )

    for fi,df in enumerate(frames):

        if df.empty:
            df['assay_identity_key']=pd.Series(index=df.index,dtype='object')
            df['historical_identity_canonicalized']=True
            continue

        local=_assay_local_keys(
            df,
            fi,
            source_policy=assay_source_policy,
        )

        df['assay_identity_key']=local.map(
            assay_map
        )

        if df['assay_identity_key'].isna().any():

            raise HistoricalIdentityConflictError(
                'Assay occurrence could not be mapped back to a canonical identity.',
                df.loc[
                    df.assay_identity_key.isna()
                ].head(50),
            )

        df['historical_identity_canonicalized']=True
    return frames,doc_audit,assay_audit


def _canon_float(x, sigfigs=12):
    if pd.isna(x):
        return ''
    return format(float(x), f'.{sigfigs}g')


def add_activity_key(df: pd.DataFrame, sigfigs=12) -> pd.DataFrame:
    out = df.copy()
    # Cross-release assay identity must already be the reconciled canonical key produced from
    # identity-only aliases. Synthetic/isolated compatibility callers may omit it, in which case
    # a native assay ChEMBL ID can be promoted, but longitudinal F0 reconstruction canonicalizes
    # all release frames before this function is used.
    if 'assay_identity_key' not in out.columns:
        if 'assay_chembl_id' not in out.columns:
            raise ValueError('assay_identity_key is required when assay_chembl_id is unavailable')
        out['assay_identity_key'] = out['assay_chembl_id'].map(lambda x: f'CHEMBL_ASSAY:{x}')
    vals = []
    for _, r in out.iterrows():
        parts = [
            str(r['assay_identity_key']), str(r['full_inchikey']), str(r['standard_type']),
            str(r['standard_relation']), _canon_float(r['standard_value_molar'], sigfigs),
            str(r['target_accession'])
        ]
        vals.append(hashlib.sha256('|'.join(parts).encode()).hexdigest())
    out['activity_key'] = vals
    # Endpoint-specific pair is deliberate: IC50 and Ki are separate pre-registered tracks.
    out['pair_key'] = (
        out['full_inchikey'].astype(str) + '|' + out['target_accession'].astype(str)
        + '|' + out['standard_type'].astype(str)
    )
    out['cross_endpoint_pair_key'] = (
        out['full_inchikey'].astype(str) + '|' + out['target_accession'].astype(str)
    )
    return out


def _stable_sort_columns(df: pd.DataFrame) -> list[str]:
    cols=[]
    for c in ['release_date','release_label','document_identity_key','assay_identity_key','document_chembl_id','assay_chembl_id','molecule_chembl_id','activity_key']:
        if c in df.columns:
            cols.append(c)
    return cols


def reconstruct_first_availability(
    release_frames: list[pd.DataFrame],
    manifest: pd.DataFrame,
    sigfigs=12,
    eligibility_col='primary_record_eligible',
) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Reconstruct first eligible archive availability without relying on activity_id.

    The earliest processed archive is a left-censoring boundary. Any activity first observed in
    that archive is marked as ``first_availability_left_censored=True`` because its true first
    ChEMBL availability may predate the archive series available to this study.

    Duplicate-resolution rule: within a release, rows sharing the exact composite activity key
    represent one activity identity for first-availability/counting purposes. Multiplicity is
    retained and a deterministic provenance row is carried forward.
    """
    if not all('historical_identity_canonicalized' in df.columns for df in release_frames):
        release_frames, _, _ = canonicalize_cross_release_identities(release_frames)
    date_map = dict(zip(manifest['release_label'], pd.to_datetime(manifest['exact_release_date'])))
    if not date_map:
        raise ValueError('Manifest is empty')
    if any(pd.isna(v) for v in date_map.values()):
        bad=[k for k,v in date_map.items() if pd.isna(v)]
        raise ValueError(f'Exact release date missing in manifest for: {bad}')
    # The censor boundary is the earliest archive in the verified processed manifest, not the
    # earliest release that happens to contain an eligible row. Empty early releases are evidence
    # of absence and must not move the censor boundary forward. Notebook 04 requires one extracted
    # frame for every manifest row before calling this function.
    archive_left_censor_date=min(pd.Timestamp(v) for v in date_map.values())
    archive_left_censor_labels=sorted(k for k,v in date_map.items() if pd.Timestamp(v)==archive_left_censor_date)
    all_rows=[]
    for df in release_frames:
        if eligibility_col not in df.columns:
            raise ValueError(f'{eligibility_col} missing from extracted frame')
        elig = df[eligibility_col].fillna(False)
        x=add_activity_key(df.loc[elig].copy(), sigfigs)
        if 'release_label' not in x:
            raise ValueError('release_label missing from extracted frame')
        x['release_date']=x['release_label'].map(date_map)
        if x['release_date'].isna().any():
            bad=x.loc[x.release_date.isna(),'release_label'].unique().tolist()
            raise ValueError(f'Exact release date missing for: {bad}')
        if x.empty:
            continue
        mult=(x.groupby(['release_label','activity_key'], dropna=False).size()
              .rename('within_release_duplicate_multiplicity').reset_index())
        x=x.merge(mult,on=['release_label','activity_key'],how='left',validate='many_to_one')
        x=x.sort_values(_stable_sort_columns(x), kind='mergesort').drop_duplicates(
            ['release_label','activity_key'], keep='first'
        )
        all_rows.append(x)
    if not all_rows:
        return pd.DataFrame(), pd.DataFrame()

    combined=pd.concat(all_rows, ignore_index=True)
    combined=combined.sort_values(_stable_sort_columns(combined), kind='mergesort')
    earliest_processed_date = archive_left_censor_date
    earliest_labels = archive_left_censor_labels

    first=combined.drop_duplicates('activity_key', keep='first').copy()
    first=first.rename(columns={'release_label':'first_available_release','release_date':'first_available_date'})
    first['archive_left_censor_release_date'] = earliest_processed_date
    first['archive_left_censor_release_label'] = '|'.join(map(str, earliest_labels))
    first['first_availability_left_censored'] = pd.to_datetime(first['first_available_date']).eq(earliest_processed_date)

    pair_first=(first.groupby('pair_key',as_index=False)['first_available_date'].min()
                .rename(columns={'first_available_date':'pair_first_available_date'}))
    first=first.merge(pair_first,on='pair_key',how='left',validate='many_to_one')
    first['pair_first_left_censored'] = pd.to_datetime(first['pair_first_available_date']).eq(earliest_processed_date)
    first['is_pair_first_release'] = first['first_available_date'].eq(first['pair_first_available_date'])

    # Same-release ties are retained for audit, but exactly one deterministic representative is marked.
    ties=first[first['is_pair_first_release']].copy()
    tie_counts=ties.groupby('pair_key').size().rename('pair_first_tie_count')
    first=first.merge(tie_counts,on='pair_key',how='left')
    first['pair_first_tie_count']=first['pair_first_tie_count'].fillna(0).astype(int)
    first['is_pair_first_representative']=False
    if not ties.empty:
        tie_sort=[c for c in ['pair_key','document_identity_key','assay_identity_key','document_chembl_id','assay_chembl_id','activity_key'] if c in ties.columns]
        ties=ties.sort_values(tie_sort, kind='mergesort')
        reps=ties.drop_duplicates('pair_key',keep='first')['activity_key']
        first.loc[first['activity_key'].isin(set(reps)),'is_pair_first_representative']=True

    counts=combined.groupby('activity_key').size().rename('n_release_occurrences').reset_index()
    audit_cols=[
        'activity_key','first_available_release','first_available_date',
        'first_availability_left_censored','archive_left_censor_release_label',
        'archive_left_censor_release_date','within_release_duplicate_multiplicity'
    ]
    audit=first[audit_cols].merge(counts,on='activity_key',how='left')
    return first, audit


def annotate_broad_prior_other_endpoint(
    primary_first: pd.DataFrame,
    broad_first: pd.DataFrame,
) -> pd.DataFrame:
    """Annotate primary pairs with prior evidence from another endpoint in a broader knowledge track.

    ``broad_first`` is reconstructed from confidence>=8, all assay types, IC50+Ki eligible records.
    This diagnostic does not alter primary eligibility or temporal gates.
    """
    out=primary_first.copy()
    if out.empty:
        out['prior_other_endpoint_exposure']=False
        out['prior_other_endpoint_date']=pd.NaT
        return out
    if broad_first.empty:
        out['prior_other_endpoint_exposure']=False
        out['prior_other_endpoint_date']=pd.NaT
        return out

    b=broad_first[['cross_endpoint_pair_key','standard_type','first_available_date']].copy()
    b['first_available_date']=pd.to_datetime(b['first_available_date'])
    b=(b.groupby(['cross_endpoint_pair_key','standard_type'],as_index=False)['first_available_date'].min())

    by_cross={}
    for key,g in b.groupby('cross_endpoint_pair_key'):
        by_cross[key]=[(str(r.standard_type), pd.Timestamp(r.first_available_date)) for r in g.itertuples(index=False)]

    dates=[]
    flags=[]
    for r in out.itertuples(index=False):
        candidates=[dt for typ,dt in by_cross.get(r.cross_endpoint_pair_key,[]) if typ != str(r.standard_type)]
        prior=min(candidates) if candidates else pd.NaT
        pair_date=pd.Timestamp(r.pair_first_available_date)
        flag=(not pd.isna(prior)) and prior < pair_date
        dates.append(prior)
        flags.append(bool(flag))
    out['prior_other_endpoint_date']=dates
    out['prior_other_endpoint_exposure']=flags
    return out


def add_curation_lag(df: pd.DataFrame) -> pd.DataFrame:
    out=df.copy()
    out['document_publication_date']=pd.to_datetime(
        out['document_year'].astype('Int64').astype(str)+'-07-01', errors='coerce'
    )
    # Year midpoint is descriptive only. It is never used for E/R/C/P assignment.
    out['curation_lag_days_approx']=(
        pd.to_datetime(out['first_available_date'])-out['document_publication_date']
    ).dt.days
    left=out.get('first_availability_left_censored', pd.Series(False,index=out.index)).fillna(False)
    out['curation_lag_main_eligible']=(~left) & out['document_publication_date'].notna()
    out['curation_lag_left_censored']=left
    return out
