from __future__ import annotations
import pandas as pd


def unit_conversion_audit(df: pd.DataFrame, release_label: str | None=None) -> pd.DataFrame:
    """Snapshot audit of molar-conversion success before unit-ineligible rows are excluded."""
    x=df.copy()
    if release_label is not None:
        x['release_label']=release_label
    keys=[c for c in ['release_label','target_accession','standard_type','assay_type','confidence_score'] if c in x.columns]
    x=x[x['relation_eligible'].fillna(False)].copy()
    if x.empty:
        return pd.DataFrame(columns=keys+['relation_eligible_rows','unit_convertible_rows','unit_unconvertible_rows','unit_convertible_rate'])
    out=(x.groupby(keys,dropna=False)
         .agg(relation_eligible_rows=('unit_convertible','size'),
              unit_convertible_rows=('unit_convertible','sum'))
         .reset_index())
    out['unit_convertible_rows']=out['unit_convertible_rows'].astype(int)
    out['unit_unconvertible_rows']=out['relation_eligible_rows']-out['unit_convertible_rows']
    out['unit_convertible_rate']=out['unit_convertible_rows']/out['relation_eligible_rows'].clip(lower=1)
    return out


def assay_type_composition(layered: pd.DataFrame, panel: dict, endpoint='IC50') -> pd.DataFrame:
    x=layered[(layered.standard_type==endpoint)&layered.target_accession.isin(set(panel.values()))].copy()
    if x.empty:
        return pd.DataFrame()
    return (x.groupby(['backtest','layer','target_accession','assay_type'],dropna=False)
            .agg(activity_identities=('activity_key','nunique'),
                 unique_molecules=('full_inchikey','nunique'),
                 unique_documents=('document_identity_key','nunique'))
            .reset_index())


def confidence_composition(layered: pd.DataFrame, panel: dict, endpoint='IC50') -> pd.DataFrame:
    x=layered[(layered.standard_type==endpoint)&layered.target_accession.isin(set(panel.values()))].copy()
    if x.empty:
        return pd.DataFrame()
    x['confidence_score']=pd.to_numeric(x['confidence_score'],errors='coerce')
    return (x.groupby(['backtest','layer','target_accession','confidence_score'],dropna=False)
            .agg(activity_identities=('activity_key','nunique'),
                 unique_molecules=('full_inchikey','nunique'),
                 unique_documents=('document_identity_key','nunique'))
            .reset_index())


def cross_endpoint_prior_exposure(layered_primary: pd.DataFrame, panel: dict, endpoint='IC50') -> pd.DataFrame:
    x=layered_primary[(layered_primary.standard_type==endpoint)&
                      layered_primary.target_accession.isin(set(panel.values()))&
                      layered_primary.layer.eq('P')&layered_primary.eligible_primary_P].copy()
    if x.empty:
        return pd.DataFrame()
    rows=[]
    for (bt,acc),g in x.groupby(['backtest','target_accession']):
        pairs=g[['pair_key','prior_other_endpoint_exposure']].drop_duplicates('pair_key')
        n=len(pairs); prior=int(pairs.prior_other_endpoint_exposure.fillna(False).sum())
        rows.append({'backtest':bt,'target_accession':acc,'endpoint':endpoint,
                     'unique_primary_P_pairs':n,'pairs_with_prior_other_endpoint':prior,
                     'prior_other_endpoint_fraction':prior/n if n else float('nan')})
    return pd.DataFrame(rows)
