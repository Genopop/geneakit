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
    
def get_ancestor_depths(pedigree, proband_id):
    """Returns array of depths for all ancestors of a proband"""
    ancestors = cgeneo.get_ancestor_ids(pedigree, [proband_id])
    return [cgeneo.get_individual_depth(pedigree[a]) for a in ancestors]
    
def meangendepthVar(gen, **kwargs):
    """
    Computes variance of proband mean depths (matches R's GLPriv.entropie3V logic)
    """
    pro = kwargs.get('pro', None)
    type_ = kwargs.get('type', 'MEAN')
    
    if pro is None:
        pro = cgeneo.get_proband_ids(gen)
    
    if type_ == 'MEAN':
        # Get mean depth for each proband
        mean_depths = cgeneo.get_mean_pedigree_depths(gen, pro)
        
        # Compute variance of these means using R's logic
        variance = np.var(mean_depths, ddof=1)
        return pd.DataFrame({'Exp.Gen.Depth.Var': [variance]}, index=['Mean'])

    elif type_ == 'IND':
        # Variance of depths within each proband's ancestry
        var_per_proband = []
        for p in pro:
            depths = cgeneo.get_ancestor_depths(gen, p)
            var = np.var(depths, ddof=1) if len(depths) > 1 else 0.0
            var_per_proband.append(var)
        
        return pd.DataFrame(
            {'Exp.Gen.Depth.Var': var_per_proband},
            index=[f"Ind {p}" for p in pro]
        )

    else:
        raise ValueError("type must be 'MEAN' or 'IND'")

def nochildren(gen, individuals):
    number_of_children = cgeneo.get_number_of_children(gen, individuals)
    return number_of_children

def completeness(gen, **kwargs):
    pro = kwargs.get('pro', None)
    genNo = kwargs.get('genNo', None)
    type = kwargs.get('type', 'MEAN')
    if pro is None:
        pro = cgeneo.get_proband_ids(gen)
    if type == 'MEAN':
        data = cgeneo.compute_mean_completeness(gen, pro)
        pedigree_completeness = pd.DataFrame(
            data, columns=['mean'], copy=False)
    elif type == 'IND':
        data = cgeneo.compute_individual_completeness(gen, pro)
        pedigree_completeness = pd.DataFrame(
            data, columns=pro, copy=False)
    if genNo is None:
        return pedigree_completeness
    else:
        return pedigree_completeness.iloc[genNo, :]
    
def completenessVar(gen, **kwargs):
    """
    Computes the variance of the completeness index across probands for each generation.
    Matches R's var() calculation (sample variance) using ddof=1.
    """
    pro = kwargs.get('pro', None)
    genNo = kwargs.get('genNo', None)
    
    if pro is None:
        pro = cgeneo.get_proband_ids(gen)
    
    # Get individual completeness matrix (generations x probands)
    data = cgeneo.compute_individual_completeness(gen, pro)
    
    # Compute sample variance (matching R's var()) using ddof=1
    variances = np.var(data, axis=1, ddof=1)
    
    # Create DataFrame with generations as index
    generations = np.arange(data.shape[0])
    df = pd.DataFrame({'completeness.var': variances}, index=generations)
    
    # Filter specific generations if genNo is provided
    if genNo is not None:
        df = df.loc[genNo, :]
    
    return df

def implex(gen, **kwargs):
    pro = kwargs.get('pro', None)
    if pro is None:
        pro = cgeneo.get_proband_ids(gen)
    genNo = kwargs.get('genNo', None)
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
    if genNo is None:
        return pedigree_implex
    else:
        return pedigree_implex.iloc[genNo, :]
    
def implexVar(gen, **kwargs):
    """
    Computes the variance of the implex index across probands for each generation.
    Matches R's var() calculation (sample variance) using ddof=1.
    
    Parameters:
    - gen: The pedigree object
    - pro (list): List of proband IDs (default: all probands)
    - genNo (list): Specific generations to include (default: all)
    - onlyNewAnc (bool): Whether to count only new ancestors (default: False)
    
    Returns:
    - pd.DataFrame: Variance of implex per generation
    """
    pro = kwargs.get('pro', None)
    genNo = kwargs.get('genNo', None)
    only_new_anc = kwargs.get('onlyNewAnc', False)
    
    if pro is None:
        pro = cgeneo.get_proband_ids(gen)
    
    # Get individual implex matrix (generations x probands)
    data = cgeneo.compute_individual_implex(gen, pro, only_new_anc)
    
    # Compute sample variance (matching R's var()) using ddof=1
    variances = np.var(data, axis=0, ddof=1)
    
    # Create DataFrame with generations as index
    generations = np.arange(data.shape[1])
    df = pd.DataFrame({'implex.var': variances}, index=generations)
    
    # Filter specific generations if genNo is provided
    if genNo is not None:
        df = df.loc[genNo, :]
    
    return df

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