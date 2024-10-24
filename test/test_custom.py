import pandas as pd
import geneo as gen

def test_custom_pedigree():
    inds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fathers = [0, 0, 0, 1, 1, 0, 3, 3, 6, 6]
    mothers = [0, 0, 0, 2, 2, 0, 4, 4, 5, 5]
    sexes = [1, 2, 1, 2, 2, 1, 2, 1, 1, 2]
    df = pd.DataFrame({'ind': inds, 'father': fathers,
                       'mother': mothers, 'sex': sexes})
    ped = gen.genealogy(df)
    phi = gen.phi(ped)
    mean = gen.phiMean(phi)
    assert mean == 0.125