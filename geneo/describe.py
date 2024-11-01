import pandas as pd
import numpy as np
import cgeneo

# Describe a pedigree

def noind(gen):
    number_of_individuals = cgeneo.get_number_of_individuals(gen)
    return number_of_individuals

def nomen(gen):
    number_of_men = cgeneo.get_number_of_men(gen)
    return number_of_men

def nowomen(gen):
    number_of_women = cgeneo.get_number_of_women(gen)
    return number_of_women

def depth(gen):
    pedigree_depth = cgeneo.get_pedigree_depth(gen)
    return pedigree_depth

def min(gen, individuals):
    minima = cgeneo.get_min_ancestor_path_lengths(gen, individuals)
    df = pd.DataFrame([minima], index=['min'], columns=individuals, copy=False)
    return df

def mean(gen, individuals):
    means = cgeneo.get_mean_ancestor_path_lengths(gen, individuals)
    df = pd.DataFrame([means], index=['mean'], columns=individuals, copy=False)
    return df

def max(gen, individuals):
    maxima = cgeneo.get_max_ancestor_path_lengths(gen, individuals)
    df = pd.DataFrame([maxima], index=['max'], columns=individuals, copy=False)
    return df

def meangendepth(gen, **kwargs):
    pro = kwargs.get('pro', None)
    if pro is None:
        pro = cgeneo.get_proband_ids(gen)
    type = kwargs.get('type', 'MEAN')
    data = cgeneo.get_mean_pedigree_depths(gen, pro)
    if type == 'MEAN':
        return np.mean(data)
    elif type == 'IND':
        df = pd.DataFrame(data, index=pro, columns=['Exp.Gen.Depth'],
                          copy=False)
        return df

def nochildren(gen, individuals):
    number_of_children = cgeneo.get_number_of_children(gen, individuals)
    return number_of_children

def completeness(gen, **kwargs):
    pro = kwargs.get('pro', None)
    genNo = kwargs.get('genNo', None)
    type = kwargs.get('type', 'MEAN')
    if pro is None:
        pro = cgeneo.get_proband_ids(gen)
    if genNo is None:
        depth = cgeneo.get_pedigree_depth(gen)
        genNo = list(range(0, depth))
    if type == 'MEAN':
        data = cgeneo.compute_mean_completeness(gen, pro)
        pedigree_completeness = pd.DataFrame(
            data, columns=['mean'], copy=False)
    elif type == 'IND':
        data = cgeneo.compute_individual_completeness(gen, pro)
        pedigree_completeness = pd.DataFrame(
            data, columns=pro, copy=False)
    return pedigree_completeness.iloc[genNo, :]

def implex(gen, **kwargs):
    pro = kwargs.get('pro', None)
    if pro is None:
        pro = cgeneo.get_proband_ids(gen)
    genNo = kwargs.get('genNo', None)
    if genNo is None:
        depth = cgeneo.get_pedigree_depth(gen)
        genNo = list(range(0, depth))
    type = kwargs.get('type', 'MEAN')
    onlyNewAnc = kwargs.get('onlyNewAnc', False)
    if type == 'MEAN':
        data = cgeneo.compute_mean_implex(gen, pro, onlyNewAnc)
        pedigree_implex = pd.DataFrame(
            data, columns=['mean'], copy=False)
    elif type == 'IND':
        data = cgeneo.compute_individual_implex(gen, pro, onlyNewAnc)
        pedigree_implex = pd.DataFrame(
            data, columns=pro, copy=False)
    return pedigree_implex.iloc[genNo, :]

def occ(gen, **kwargs):
    pro = kwargs.get('pro', None)
    ancestors = kwargs.get('ancestors', None)
    if pro is None:
        pro = cgeneo.get_proband_ids(gen)
    if ancestors is None:
        ancestors = cgeneo.get_founder_ids(gen)
    typeOcc = kwargs.get('typeOcc', 'IND')
    if typeOcc == 'TOTAL':
        data = cgeneo.count_total_occurrences(gen, ancestors, pro)
        pedigree_occurrence = pd.DataFrame(
            data, index=ancestors, columns=['total'], copy=False)
    elif typeOcc == 'IND':
        data = cgeneo.count_individual_occurrences(gen, ancestors, pro)
        pedigree_occurrence = pd.DataFrame(
            data, index=ancestors, columns=pro, copy=False)
    return pedigree_occurrence

def rec(gen, **kwargs):
    pro = kwargs.get('pro', None)
    ancestors = kwargs.get('ancestors', None)
    if pro is None:
        pro = cgeneo.get_proband_ids(gen)
    if ancestors is None:
        ancestors = cgeneo.get_founder_ids(gen)
    data = cgeneo.count_coverage(gen, pro, ancestors)
    coverage = pd.DataFrame(
        data, index=ancestors, columns=['coverage'], copy=False)
    return coverage

def findDistance(gen, individuals, ancestor):
    distance = cgeneo.get_min_common_ancestor_path_length(
        gen, individuals[0], individuals[1], ancestor)
    return distance