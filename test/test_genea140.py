import geneakit as gen
from geneakit import genea140

ped = gen.genealogy(genea140)

def test_printing():
    assert repr(ped) == 'A pedigree with:\n - 41523 individuals;\n' + \
        ' - 68248 parent-child relations;\n - 20773 men;\n - 20750 women;\n' + \
        ' - 140 probands;\n - 18 generations.'
    
def test_number_of_individuals():
    assert gen.noind(ped) == 41523

def test_number_of_men():
    assert gen.nomen(ped) == 20773

def test_number_of_women():
    assert gen.nowomen(ped) == 20750

def test_children():
    assert gen.children(ped, [33724]) == [10033, 113470]

def test_genetic_contribution():
    assert gen.gc(ped).sum().sum() == 140

def test_descendants():
    assert gen.descendant(ped, [10086]) == [10009, 10018, 10033, 33724, 105379,
        113470, 408065, 408069, 408075, 408375, 408937, 409808, 623919, 712249,
        712256, 860834, 860838, 868738, 868740, 868743]

def test_depth():
    assert gen.depth(ped) == 18

def test_siblings():
    assert gen.sibship(ped, [11520]) == [15397, 39369, 49658]

def test_half_siblings():
    assert gen.sibship(ped, [11520], halfSibling=False) == []

def test_kinships():
    phi = gen.phi(ped)
    assert gen.phiMean(phi) == 0.0011437357709631094

def test_sparse_kinships():
    phi = gen.phi(ped, sparse=True)
    assert gen.phiMean(phi) == 0.0011437357709631094