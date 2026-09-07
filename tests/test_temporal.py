import pandas as pd
from ad_f0.temporal import layer_for_date, assign_backtest_layers

SPEC={'E_end':'2010-01-01','R_start':'2010-01-01','R_end':'2012-01-01',
      'C_start':'2012-01-01','C_end':'2014-01-01','P_start':'2014-01-01','P_end':'2017-01-01'}

def test_boundaries():
    assert layer_for_date('2010-01-01',SPEC)=='E'
    assert layer_for_date('2010-01-02',SPEC)=='R'
    assert layer_for_date('2012-01-01',SPEC)=='R'
    assert layer_for_date('2012-01-02',SPEC)=='C'
    assert layer_for_date('2014-01-02',SPEC)=='P'


def test_later_repeat_not_new_p():
    df=pd.DataFrame({'first_available_date':pd.to_datetime(['2014-06-01','2014-07-01']),
                     'is_pair_first_release':[True,False],
                     'is_pair_first_representative':[True,False]})
    out=assign_backtest_layers(df,{'A':SPEC})
    assert out['eligible_primary_P'].tolist()==[True,False]
