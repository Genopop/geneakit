import numpy as np
import pandas as pd
import cgeneo

def pro(gen):
    """Get proband IDs (individuals without children in the genealogy)
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        
    Returns:
        list: Proband IDs sorted in ascending order
        
    Examples:
        >>> import geneo as gen
        >>> from geneo import geneaJi
        >>> ped = gen.genealogy(geneaJi)
        >>> pro = gen.pro(pedigree)
        >>> print(pro)
        [1, 2, 29]
        
    See Also:
        gen.founder: Get individuals without parents
        gen.children: Find children of specified individuals
    """
    proband_ids = cgeneo.get_proband_ids(gen)
    return proband_ids

def founder(gen):
    """Get founder IDs (individuals without known parents)
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
    
    Returns:
        list: Founder IDs sorted in ascending order
        
    Notes:
        Founders are defined as individuals with both parents = 0
        Use gen.half_founder() for individuals with one unknown parent
        
    Examples:
        >>> import geneo as gen
        >>> from geneo import genea140
        >>> ped = gen.genealogy(genea140)
        >>> founders = gen.founder(pedigree)
        >>> print(f"Founders count: {len(founders)}")
        Founders count: 7399
    """
    founder_ids = cgeneo.get_founder_ids(gen)
    return founder_ids

def half_founder(gen):
    """Get half-founder IDs (individuals with one unknown parent)
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        
    Returns:
        list: Half-founder IDs sorted in ascending order
        
    Notes:
        Half-founders have either father=0 or mother=0, but not both
        
    Examples:
        >>> import geneo as gen
        >>> from geneo import geneaJi
        >>> ped = gen.genealogy(geneaJi)
        >>> hf = gen.half_founder(pedigree)
        >>> print(hf)
        [9, 11]
    """
    half_founder_ids = cgeneo.get_half_founder_ids(gen)
    return half_founder_ids

def parent(gen, individuals, **kwargs):
    """Get parental IDs for specified individuals
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        individuals (list): Target individual IDs
        output (str): Parent type to return:
            'FaMo' - Both parents (default)
            'Fa' - Fathers only
            'Mo' - Mothers only
            
    Returns:
        dict: Dictionary with 'Fathers' and/or 'Mothers' keys containing
              lists of parental IDs (0 indicates unknown parent)
              
    Examples:
        >>> import geneo as gen
        >>> from geneo import geneaJi
        >>> ped = gen.genealogy(geneaJi)
        >>> pro = gen.pro(ped)
        >>> parents = gen.parent(ped, pro)
        >>> print(parents['Fathers'])
        [4, 28]
        
    See Also:
        gen.children: Get inverse relationship
        gen.sibship: Find siblings through shared parents
    """
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
    """Get sibling IDs for specified individuals
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        individuals (list): Target individual IDs
        halfSibling (bool): Include half-siblings if True (default)
        
    Returns:
        dict: Dictionary mapping each individual to their siblings' IDs
        
    Notes:
        Full siblings share both parents
        Half-siblings share one parent
        
    Examples:
        >>> import geneo as gen
        >>> from geneo import genea140
        >>> ped = gen.genealogy(genea140)
        >>> siblings = gen.sibship(ped, [113470])
        >>> print(siblings)
        [10033]
    """
    halfSibling = kwargs.get('halfSibling', True)
    sibling_ids = cgeneo.get_sibling_ids(gen, individuals, halfSibling)
    return sibling_ids

def children(gen, individuals):
    """Get children IDs for specified individuals
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        individuals (list): Target individual IDs
        
    Returns:
        list: Children's IDs sorted in alphabetical order
        
    Examples:
        >>> import geneo as gen
        >>> from geneo import genea140
        >>> ped = gen.genealogy(genea140)
        >>> children = gen.children(ped, [10086])
        >>> print(children)
        [33724]
        
    See Also:
        gen.parent: Inverse relationship lookup
    """
    children_ids = cgeneo.get_children_ids(gen, individuals)
    return children_ids

def ancestor(gen, individuals, **kwargs):
    """Get ancestor IDs for specified individuals
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        individuals (list): Target individual IDs
        type (str): Ancestor retrieval mode:
            'UNIQUE' - One occurrence per ancestor (default)
            'TOTAL' - All occurrences of ancestors
            
    Returns:
        list: Ancestor IDs
        
    Examples:
        >>> import geneo as gen
        >>> from geneo import genea140
        >>> ped = gen.genealogy(genea140)
        >>> pro = gen.pro(ped)
        >>> all_ancestors = gen.ancestor(ped, pro, type='TOTAL')
        >>> print(f"Ancestors count: {len(all_ancestors)}")
        Ancestors count: 575418
    """
    type = kwargs.get('type', 'UNIQUE')
    if type == 'UNIQUE':
        ancestor_ids = cgeneo.get_ancestor_ids(gen, individuals)
    elif type == 'TOTAL':
        ancestor_ids = cgeneo.get_all_ancestor_ids(gen, individuals)
    return ancestor_ids

def descendant(gen, individuals, **kwargs):
    """Get descendant IDs for specified individuals
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        individuals (list): Target individual IDs
        type (str): Descendant retrieval mode:
            'UNIQUE' - One occurrence per descendant (default)
            'TOTAL' - All occurrences of descendants
            
    Returns:
        list: Descendant IDs
        
    Examples:
        >>> import geneo as gen
        >>> from geneo import geneaa140
        >>> ped = gen.genealogy(genea140)
        >>> founders = gen.founder(ped)
        >>> all_descendants = gen.descendant(ped, founders, type='TOTAL')
        >>> print(f"Descendants count: {len(all_descendants)}")
        Descendants count: 2123427
    """
    type = kwargs.get('type', 'UNIQUE')
    if type == 'UNIQUE':
        descendant_ids = cgeneo.get_descendant_ids(gen, individuals)
    elif type == 'TOTAL':
        descendant_ids = cgeneo.get_all_descendant_ids(gen, individuals)
    return descendant_ids

def commonAncestor(gen, individuals):
    """Find common ancestors shared by multiple individuals
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        individuals (list): Target individual IDs
        
    Returns:
        list: Shared ancestor IDs sorted in ascending order
        
    Examples:
        >>> import geneo as gen
        >>> from geneo import genea140
        >>> ped = gen.genealogy(genea140)
        >>> common_anc = gen.commonAncestor(ped, [113470, 10033])
        >>> print(f"Common ancestors count: {len(common_anc)}")
        Common ancestors count: 38
    """
    common_ancestor_ids = cgeneo.get_common_ancestor_ids(gen, individuals)
    return common_ancestor_ids

def findFounders(gen, individuals):
    """Find common founders shared by multiple individuals
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        individuals (list): Target individual IDs
        
    Returns:
        list: Shared founder IDs sorted in ascending order
        
    Examples:
        >>> import geneo as gen
        >>> from geneo import genea140
        >>> ped = gen.genealogy(geneaJi)
        >>> shared_founders = gen.findFounders(ped, [1, 29])
        >>> print(len(shared_founders))
        5
    """
    common_founder_ids = cgeneo.get_common_founder_ids(gen, individuals)
    return common_founder_ids

def findMRCA(gen, individuals):
    """Identify Most Recent Common Ancestors (MRCAs)
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        individuals (list): Target individual IDs
        
    Returns:
        pd.DataFrame: Meioses matrix with:
            - Rows: Target individuals
            - Columns: MRCA IDs
            - Values: Number of meioses (generational steps)
            
    Examples:
        >>> import geneo as gen
        >>> from geneo import geneaJi
        >>> ped = gen.genealogy(geneaJi)
        >>> mrca_matrix = gen.findMRCA(ped, [1, 29])
        >>> print(mrca_matrix)
            14  20
        1    4   4
        29   3   3
    """
    mrca_ids = cgeneo.get_mrca_ids(gen, individuals)
    cmatrix = cgeneo.get_mrca_meioses(gen, individuals, mrca_ids)
    meioses_matrix = pd.DataFrame(
        cmatrix, index=individuals, columns=mrca_ids, copy = False)
    return meioses_matrix

def find_Min_Distance_MRCA(genMatrix, **kwargs):
    """Calculate minimum genetic distances through MRCAs
    
    Args:
        genMatrix (pd.DataFrame): Meioses matrix from findMRCA
        individuals (list): Subset of individuals to analyze (default: all)
        ancestors (list): Subset of ancestors to consider (default: all)
        
    Returns:
        pd.DataFrame: Minimum distances with columns:
            - founder: MRCA ID
            - proband1: First individual ID
            - proband2: Second individual ID  
            - distance: Total meioses count
            
    Examples:
        >>> import geneo as gen
        >>> from geneo import geneaJi
        >>> ped = gen.genealogy(geneaJi)
        >>> mrca_matrix = gen.findMRCA(ped, [1, 29])
        >>> min_dist = gen.find_Min_Distance_MRCA(mrca_matrix)
        >>> print(min_dist)
           founder  proband1  proband2  distance
        0       14         1        29         4
        1       20         1        29         4
    """
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
            
    df = pd.DataFrame({
        'founder': founder_vector,
        'proband1': proband1_vector,
        'proband2': proband2_vector,
        'distance': distances_vector
    })
    return df