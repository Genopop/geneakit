import numpy as np
import pandas as pd
import cgeneo

# Identify individuals in a pedigree

def pro(gen):
    proband_ids = cgeneo.get_proband_ids(gen)
    return proband_ids

def founder(gen):
    founder_ids = cgeneo.get_founder_ids(gen)
    return founder_ids

def half_founder(gen):
    half_founder_ids = cgeneo.get_half_founder_ids(gen)
    return half_founder_ids

def parent(gen, individuals, **kwargs):
    output = kwargs.get('output', 'FaMo')
    parents = dict()
    if output == 'FaMo':
        parents['Fathers'] = cgeneo.get_father_ids(gen, individuals)
        parents['Mothers'] = cgeneo.get_mother_ids(gen, individuals)
    if output == 'Fa':
        parents['Fathers'] = cgeneo.get_father_ids(gen, individuals)
    if output == 'Mo':
        parents['Mothers'] = cgeneo.get_mother_ids(gen, individuals)
    return parents

def sibship(gen, individuals, **kwargs):
    halfSibling = kwargs.get('halfSibling', True)
    sibling_ids = cgeneo.get_sibling_ids(gen, individuals, halfSibling)
    return sibling_ids

def children(gen, individuals):
    children_ids = cgeneo.get_children_ids(gen, individuals)
    return children_ids

def ancestor(gen, individuals, **kwargs):
    type = kwargs.get('type', 'UNIQUE')
    if type == 'UNIQUE':
        ancestor_ids = cgeneo.get_ancestor_ids(gen, individuals)
    elif type == 'TOTAL':
        ancestor_ids = cgeneo.get_all_ancestor_ids(gen, individuals)
    return ancestor_ids

def descendant(gen, individuals, **kwargs):
    type = kwargs.get('type', 'UNIQUE')
    if type == 'UNIQUE':
        descendant_ids = cgeneo.get_descendant_ids(gen, individuals)
    elif type == 'TOTAL':
        descendant_ids = cgeneo.get_all_descendant_ids(gen, individuals)
    return descendant_ids

def commonAncestor(gen, individuals):
    common_ancestor_ids = cgeneo.get_common_ancestor_ids(gen, individuals)
    return common_ancestor_ids

def findFounders(gen, individuals):
    common_founder_ids = cgeneo.get_common_founder_ids(gen, individuals)
    return common_founder_ids

def findMRCA(gen, individuals):
    mrca_ids = cgeneo.get_mrca_ids(gen, individuals)
    cmatrix = cgeneo.get_mrca_meioses(gen, individuals, mrca_ids)
    meioses_matrix = pd.DataFrame(
        cmatrix, index=individuals, columns=mrca_ids, copy = False)
    return meioses_matrix

def find_Min_Distance_MRCA(genMatrix, **kwargs):
    individuals = kwargs.get('individuals', None)
    if individuals is None:
        individuals = genMatrix.index
    ancestors = kwargs.get('ancestors', None)
    if ancestors is None:
        ancestors = genMatrix.columns
    meioses_matrix = genMatrix.loc[individuals, ancestors]
    founder_vector = []
    proband1_vector = []
    proband2_vector = []
    distances_vector = []
    for ancestor in ancestors:
        min_distance = np.inf
        min_pairs = []
        for i, proband1 in enumerate(individuals):
            for proband2 in individuals[i+1:]:
                distance = meioses_matrix.at[proband1, ancestor]
                + meioses_matrix.at[proband2, ancestor]
                if distance < min_distance:
                    min_distance = distance
                    min_pairs = [(proband1, proband2)]
                elif distance == min_distance:
                    min_pairs.append((proband1, proband2))
        for proband1, proband2 in min_pairs:
            founder_vector.append(ancestor)
            proband1_vector.append(proband1)
            proband2_vector.append(proband2)
            distances_vector.append(min_distance)
    df = pd.DataFrame(
        {'founder': founder_vector,
         'proband1': proband1_vector,
         'proband2': proband2_vector,
         'distance': distances_vector
        })
    return df