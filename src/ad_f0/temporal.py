from __future__ import annotations
import pandas as pd


def layer_for_date(d, spec):
    d=pd.Timestamp(d)
    if d <= pd.Timestamp(spec['E_end']): return 'E'
    if pd.Timestamp(spec['R_start']) < d <= pd.Timestamp(spec['R_end']): return 'R'
    if pd.Timestamp(spec['C_start']) < d <= pd.Timestamp(spec['C_end']): return 'C'
    if pd.Timestamp(spec['P_start']) < d <= pd.Timestamp(spec['P_end']): return 'P'
    return None


def assign_backtest_layers(df: pd.DataFrame, backtests: dict) -> pd.DataFrame:
    rows=[]
    for name,spec in backtests.items():
        x=df.copy()
        x['backtest']=name
        x['layer']=[layer_for_date(d,spec) for d in x['first_available_date']]
        x=x[x['layer'].notna()].copy()
        # A primary P pair is endpoint-specific and temporally new in P. Multiple tied assays
        # may exist, but gates use unique pair_key, not row count.
        x['eligible_primary_P'] = x['layer'].eq('P') & x['is_pair_first_release'].fillna(False)
        x['eligible_primary_P_representative'] = x['layer'].eq('P') & x['is_pair_first_representative'].fillna(False)
        rows.append(x)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
