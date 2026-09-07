from ad_f0.units import to_molar,pactivity_from_molar

def test_units():
    assert abs(to_molar(1000,'nM')-1e-6)<1e-15
    assert abs(pactivity_from_molar(1e-6)-6)<1e-12
