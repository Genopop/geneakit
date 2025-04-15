import pandas as pd
import numpy as np
import cgeneo
from .extract import branching

def noind(gen):
    """Get total number of individuals in the genealogy
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        
    Returns:
        int: Count of all individuals in the pedigree
        
    Examples:
        >>> import geneo as gen
        >>> ped = gen.genealogy(gen.genea140)
        >>> total = gen.noind(ped)
        >>> print(f"Pedigree size: {total}")
        Pedigree size: 41523
    """
    number_of_individuals = cgeneo.get_number_of_individuals(gen)
    return number_of_individuals

def nomen(gen):
    """Count male individuals in the genealogy
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        
    Returns:
        int: Number of individuals marked as male (sex=1)
        
    Examples:
        >>> import geneo as gen
        >>> ped = gen.genealogy(gen.genea140)
        >>> males = gen.nomen(ped)
        >>> print(f"Male count: {males}")
        Male count: 20773
    """
    number_of_men = cgeneo.get_number_of_men(gen)
    return number_of_men

def nowomen(gen):
    """Count female individuals in the genealogy
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        
    Returns:
        int: Number of individuals marked as female (sex=2)
        
    Examples:
        >>> import geneo as gen
        >>> ped = gen.genealogy(gen.genea140)
        >>> females = gen.nowomen(ped)
        >>> print(f"Female count: {females}")
        Female count: 20750
    """
    number_of_women = cgeneo.get_number_of_women(gen)
    return number_of_women

def depth(gen):
    """Calculate maximum generational depth
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        
    Returns:
        int: Maximum number of generations between probands
             and their deepest ancestors
            
    Examples:
        >>> import geneo as gen
        >>> ped = gen.genealogy(gen.genea140)
        >>> gen_depth = gen.depth(ped)
        >>> print(f"Generational depth: {gen_depth}")
        Generational depth: 18
    """
    pedigree_depth = cgeneo.get_pedigree_depth(gen)
    return pedigree_depth

def min(gen, individuals):
    """Calculate minimum ancestral path lengths
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        individuals (list): Target individual IDs
        
    Returns:
        pd.DataFrame: Minimum generational distances with:
            - Columns: Individual IDs
            - Single row: 'min' values
            
    Examples:
        >>> import geneo as gen
        >>> ped = gen.genealogy(gen.geneaJi)
        >>> min_depths = gen.min(ped, [17, 26])
        >>> print(min_depths)
              17  26
        min    4   3
    """
    minima = cgeneo.get_min_ancestor_path_lengths(gen, individuals)
    df = pd.DataFrame([minima], index=['min'], columns=individuals, copy=False)
    return df

def mean(gen, individuals):
    """Calculate average ancestral path lengths
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        individuals (list): Target individual IDs
        
    Returns:
        pd.DataFrame: Mean generational distances with:
            - Columns: Individual IDs
            - Single row: 'mean' values
            
    Examples:
        >>> import geneo as gen
        >>> ped = gen.genealogy(gen.geneaJi)
        >>> avg_depths = gen.mean(ped, [17, 26])
        >>> print(avg_depths)
                    17        26
        mean  4.285714  6.052632
    """
    means = cgeneo.get_mean_ancestor_path_lengths(gen, individuals)
    df = pd.DataFrame([means], index=['mean'], columns=individuals, copy=False)
    return df

max_ = max

def max(gen, individuals):
    """Calculate maximum ancestral path lengths
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        individuals (list): Target individual IDs
        
    Returns:
        pd.DataFrame: Maximum generational distances with:
            - Columns: Individual IDs
            - Single row: 'max' values
            
    Examples:
        >>> import geneo as gen
        >>> ped = gen.genealogy(gen.geneaJi)
        >>> max_depths = gen.max(ped, [17, 26])
        >>> print(max_depths)
              17  26
        max    5   7
    """
    maxima = cgeneo.get_max_ancestor_path_lengths(gen, individuals)
    df = pd.DataFrame([maxima], index=['max'], columns=individuals, copy=False)
    return df

def meangendepth(gen, **kwargs):
    """Calculate expected genealogical depth
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        pro (list, optional): Proband IDs to include (default: all)
        type (str): Output format:
            'MEAN' - Average depth (default)
            'IND' - Individual depths
            
    Returns:
        float | pd.DataFrame: Either:
            - Single mean value for 'MEAN' type
            - DataFrame with individual depths for 'IND' type
            
    Examples:
        >>> import geneo as gen
        >>> ped = gen.genealogy(gen.geneaJi)
        >>> avg_depth = gen.meangendepth(ped)
        >>> print(f"Expected depth: {avg_depth:.2f}")
        Expected depth: 4.12
        
        >>> ind_depths = gen.meangendepth(ped, type='IND')
        >>> print(ind_depths.head())
            Exp.Gen.Depth
        1         4.59375
        2         4.59375
        29        3.18750
    """
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

def get_generational_counts(gen, pro):
    current_gen = pro
    vctF = []
    vctDF = []

    while current_gen:
        founders = 0
        semi_founders = 0
        next_gen = []
        
        for ind in current_gen:
            father = gen[ind].father.ind
            mother = gen[ind].mother.ind
            
            if father == 0 and mother == 0:
                founders += 1
            elif (father == 0) != (mother == 0):  # Logical XOR for semi-founders
                semi_founders += 1
            
            # Collect parents for next generation
            if father != 0:
                next_gen.append(father)
            if mother != 0:
                next_gen.append(mother)
        
        vctF.append(founders)
        vctDF.append(semi_founders)
        current_gen = list(next_gen)

    return vctF, vctDF

def variance3V(gen, pro=None):
    if pro is None:
        pro = cgeneo.get_proband_ids(gen)
    N = len(pro)
    if N == 0:
        return 0.0

    vctF, vctDF = get_generational_counts(gen, pro)
    
    P = 0.0
    sum_sq = 0.0
    
    for genNo in range(len(vctF)):
        weight = N * (2 ** genNo)
        if weight == 0:
            continue
        
        # Founder contribution
        termF = (genNo * vctF[genNo]) / weight
        P += termF
        sum_sq += (genNo ** 2 * vctF[genNo]) / weight
        
        # Semi-founder contribution (0.5 weight)
        termDF = (genNo * vctDF[genNo] * 0.5) / weight
        P += termDF
        sum_sq += (genNo ** 2 * vctDF[genNo] * 0.5) / weight
    
    variance = sum_sq - (P ** 2)
    return variance

def meangendepthVar(gen, **kwargs):
    """Calculate variance of genealogical depth
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        pro (list, optional): Proband IDs to include (default: all)
        type (str): Calculation type:
            'MEAN' - Population variance (default)
            'IND' - Individual variances
            
    Returns:
        float | pd.DataFrame: Either:
            - Single variance value for 'MEAN' type
            - DataFrame with individual variances for 'IND' type
            
    Examples:
        >>> import geneo as gen
        >>> ped = gen.genealogy(gen.geneaJi)
        >>> pop_var = gen.meangendepthVar(ped)
        >>> print(f"Population variance: {pop_var:.2f}")
        Population variance: 1.65
        
        >>> ind_vars = gen.meangendepthVar(ped, type='IND')
        >>> print(ind_vars.head())
            Mean.Gen.Depth
        1         1.241211
        2         1.241211
        29        1.152344

    """
    pro = kwargs.get('pro', None)
    if pro is None:
        pro = cgeneo.get_proband_ids(gen)
    type = kwargs.get('type', 'MEAN')
    if type == 'MEAN':
        variance = variance3V(gen, pro=pro)
        return variance
    elif type == 'IND':
        variances = [variance3V(branching(gen, pro=[ind])) for ind in pro]
        return pd.DataFrame(
            variances,
            columns=["Mean.Gen.Depth"],
            index=pro
        )

def nochildren(gen, individuals):
    """Count number of children per individual
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        individuals (list): Target individual IDs
        
    Returns:
        list: Child counts in same order as the target individuals
        
    Examples:
        >>> import geneo as gen
        >>> ped = gen.genealogy(gen.geneaJi)
        >>> child_counts = gen.nochildren(ped, [14, 20])
        >>> print(child_counts)
        [4, 3]
    """
    number_of_children = cgeneo.get_number_of_children(gen, individuals)
    return number_of_children

def completeness(gen, **kwargs):
    """Calculate pedigree completeness index
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        pro (list, optional): Proband IDs (default: all)
        genNo (int, optional): Specific generation to return
        type (str): 'MEAN' for average or 'IND' for per-proband
        
    Returns:
        pd.DataFrame: Completeness scores with:
            - Rows: Generations
            - Columns: Probands (IND) or 'mean'
            
    Examples:
        >>> import geneo as gen
        >>> ped = gen.genealogy(gen.geneaJi)
        >>> comp = gen.completeness(ped)
        >>> print(comp.head(3))
            mean
        0  100.0
        1  100.0
        2  100.0
    """
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
    """Calculate variance of completeness index
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        pro (list, optional): Proband IDs (default: all)
        genNo (list, optional): Specific generation(s) to return
        
    Returns:
        pd.DataFrame: Variance values with:
            - Rows: Generations
            - Column: 'completeness.var'
            
    Notes:
        Uses sample variance (ddof=1) matching R's var()
        
    Examples:
        >>> import geneo as gen
        >>> ped = gen.genealogy(gen.geneaJi)
        >>> comp_var = gen.completenessVar(ped)
        >>> print(comp_var)
           completeness.var
        0          0.000000
        1          0.000000
        2          0.000000
        3        208.333333
        4       1054.687500
        5        468.750000
        6        117.187500
        7          3.255208
    """
    pro = kwargs.get('pro', None)
    genNo = kwargs.get('genNo', None)
    
    if pro is None:
        pro = cgeneo.get_proband_ids(gen)
    
    data = cgeneo.compute_individual_completeness(gen, pro)
    variances = np.var(data, axis=1, ddof=1)
    generations = np.arange(data.shape[0])
    df = pd.DataFrame({'completeness.var': variances}, index=generations)
    
    if genNo is not None:
        df = df.loc[genNo, :]
    
    return df

def implex(gen, **kwargs):
    """Calculate genealogical implex (pedigree collapse)
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        pro (list, optional): Proband IDs (default: all)
        genNo (list, optional): Specific generation(s) to return
        type (str): 'MEAN' for average or 'IND' for per-proband
        onlyNewAnc (bool): Count only new ancestors
        
    Returns:
        pd.DataFrame: Implex values with:
            - Rows: Generations
            - Columns: Probands (IND) or 'mean'
            
    Examples:
        >>> import geneo as gen
        >>> ped = gen.genealogy(geneaJi)
        >>> imp = gen.implex(ped)
        >>> print(imp)
                 mean
        0  100.000000
        1  100.000000
        2  100.000000
        3   58.333333
        4   25.000000
        5   10.416667
        6    5.208333
        7    1.041667
    """
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
    """Calculate variance of implex index
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        pro (list, optional): Proband IDs (default: all)
        genNo (list, optional): Specific generation(s) to return
        onlyNewAnc (bool): Count only new ancestors
        
    Returns:
        pd.DataFrame: Variance values with:
            - Rows: Generations
            - Column: 'implex.var'
            
    Examples:
        >>> import geneo as gen
        >>> ped = gen.genealogy(gen.geneaJi)
        >>> imp_var = gen.implexVar(ped)
        >>> print(imp_var)
           implex.var
        0    0.000000
        1    0.000000
        2    0.000000
        3   52.083333
        4  117.187500
        5   13.020833
        6    3.255208
        7    0.813802
    """
    pro = kwargs.get('pro', None)
    genNo = kwargs.get('genNo', None)
    only_new_anc = kwargs.get('onlyNewAnc', False)
    
    if pro is None:
        pro = cgeneo.get_proband_ids(gen)
    
    data = cgeneo.compute_individual_implex(gen, pro, only_new_anc)
    variances = np.var(data, axis=0, ddof=1)
    generations = np.arange(data.shape[1])
    df = pd.DataFrame({'implex.var': variances}, index=generations)
    
    if genNo is not None:
        df = df.loc[genNo, :]
    
    return df

def occ(gen, **kwargs):
    """Count ancestor occurrences in proband genealogies
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        pro (list, optional): Proband IDs (default: all)
        ancestors (list, optional): Target ancestors (default: all founders)
        typeOcc (str): 'IND' for per-proband or 'TOTAL' for sum
        
    Returns:
        pd.DataFrame: Occurrence counts with:
            - Rows: Ancestor IDs
            - Columns: Probands (IND) or 'total'
            
    Examples:
        >>> import geneo as gen
        >>> ped = gen.genealogy(gen.geneaJi)
        >>> occurrences = gen.occ(ped, ancestors=[17, 25])
        >>> print(occurrences)
            1   2   29
        17   6   6   2
        25   8   8   3
    """
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
    """Calculate ancestor coverage across probands
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        pro (list, optional): Proband IDs (default: all)
        ancestors (list, optional): Target ancestors (default: all founders)
        
    Returns:
        pd.DataFrame: Coverage counts with:
            - Rows: Ancestor IDs
            - Column: 'coverage' (number of probands descending from ancestor)
            
    Examples:
        >>> import geneo as gen
        >>> ped = gen.genealogy(gen.geneaJi)
        >>> coverage = gen.rec(ped)
        >>> print(coverage)
            coverage
        17         3
        19         3
        20         3
        23         1
        25         3
        26         3
    """
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
    """Calculate minimal genetic distance through common ancestor
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        individuals (list): Pair of individual IDs [ID1, ID2]
        ancestor (int): Common ancestor ID
        
    Returns:
        int: Total generational steps (ID1→ancestor + ID2→ancestor)
        
    Examples:
        >>> import geneo as gen
        >>> ped = gen.genealogy(gen.geneaJi)
        >>> dist = gen.findDistance(ped, [1, 29], 17)
        >>> print(f"Genetic distance: {dist}")
        Genetic distance: 8
    """
    distance = cgeneo.get_min_common_ancestor_path_length(
        gen, individuals[0], individuals[1], ancestor)
    return distance