import pandas as pd
import geneo as gen
from geneo import geneaJi

ped = gen.genealogy(geneaJi)

def test_probands():
    assert gen.pro(ped) == [1, 2, 29]

def test_founders():
    assert gen.founder(ped) == [17, 19, 20, 23, 25, 26]
    
def test_mrca():
    data = [[4, 4], [4, 4], [3, 3]]
    df = pd.DataFrame(data, index=[1, 2, 29], columns=[14, 20])
    assert all(df == gen.findMRCA(ped, [1, 2, 29]))
    
def test_inbreeding():
    assert gen.f(ped, pro=[1]).iloc[0, 0] == 0.18359375

def test_pairwise_kinship():
    assert gen.phi(ped, pro=[1, 2]).iloc[0, 1] == 0.37109375
    
def test_probands_kinship():
    phi1 = gen.phi(ped)
    phi2 = pd.DataFrame([[0.591796875, 0.37109375, 0.072265625],
                         [0.37109375, 0.591796875, 0.072265625],
                         [0.072265625, 0.072265625, 0.53515625]],
                         index=[1, 2, 29], columns=[1, 2, 29])
    assert all(phi1 == phi2)
    

def test_mean_kinship():
    phi = gen.phi(ped)
    assert gen.phiMean(phi) == 0.171875
    
    
def test_common_founders():
    assert gen.findFounders(ped, [1, 2, 29]) == [17, 19, 20, 25, 26]

def test_coverage():
    data = [3, 3, 3, 1, 3, 3]
    df = pd.DataFrame(data, index=[17, 19, 20, 23, 25, 26], columns=['coverage'])
    assert all(df == gen.rec(ped))

def test_distance():
    assert gen.findDistance(ped, [1, 2], 25) == 12

def test_individual_occurrence():
    data = [[6, 6, 2], [8, 8, 2], [1, 1, 2], [0, 0, 1], [8, 8, 3], [8, 8, 3]]
    df = pd.DataFrame(data, index=[17, 19, 20, 23, 25, 26], columns=[1, 2, 29])
    assert all(df == gen.occ(ped))

def test_total_occurrence():
    data = [14, 18, 4, 1, 19, 19]
    df = pd.DataFrame(data, index=[17, 19, 20, 23, 25, 26], columns=['total'])
    assert all(df == (gen.occ(ped, typeOcc="TOTAL")))

def test_individual_completeness():
    assert gen.completeness(ped, type="IND").iloc[7, 0] == 3.125

def test_generations_completeness():
    data = [100.0, 62.5, 18.75]
    df = pd.DataFrame(data, index=[0, 4, 6], columns=['mean'])
    assert all(df == gen.completeness(ped, genNo=[0, 4, 6]))
    
def test_branching1():
    iso_ped = gen.branching(ped, pro=[1])
    assert gen.founder(iso_ped) == [17, 19, 20, 25, 26]
    
def test_branching2():
    iso_ped = gen.branching(ped, ancestors=[13])
    assert gen.pro(iso_ped) == [1, 2]
    
def test_branching3():
    iso_ped = gen.branching(ped, pro=[1], ancestors=[13])
    assert gen.noind(iso_ped) == 4
