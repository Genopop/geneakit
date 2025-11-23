import time
import cgeneakit
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from scipy.stats import bootstrap

def phi(gen, **kwargs):
    """Compute kinship coefficients between probands.
    
    Args:
        gen (cgeneakit.Pedigree): Initialized genealogy object.
        pro (list, optional): Proband IDs. Defaults to all probands.
        verbose (bool, default False): Print details.
        compute (bool, default True): Estimate memory if False.
        sparse (bool, default False): Use sparse computation algorithm (experimental!).
        raw (bool, default False): If True, returns (matrix, ids). If False, returns pd.DataFrame.
    
    Returns:
        pd.DataFrame: Default.
        tuple: (matrix, list) If raw=True.
    """
    pro = kwargs.get('pro', None)
    if pro is None:
        pro = cgeneakit.get_proband_ids(gen)
    verbose = kwargs.get('verbose', False)
    compute = kwargs.get('compute', True)
    sparse = kwargs.get('sparse', False)
    raw = kwargs.get('raw', False)
    
    if not compute:
        required_memory = cgeneakit.get_required_memory_for_kinships(gen, pro)
        print(f'You will require at least {round(required_memory, 2)} GB of RAM.')
        return
    
    if verbose:
        begin = time.time()
        
    if sparse:
        data, indices, indptr = cgeneakit.compute_kinships_sparse(gen, pro, verbose)
        
        kinship_matrix = csr_matrix(
            (data, indices, indptr), 
            shape=(len(pro), len(pro)),
            copy=False
        )
        
        if raw:
            if verbose:
                print(f'Elapsed time: {round(time.time() - begin, 2)} seconds')
            return kinship_matrix, pro
        else:
            if verbose:
                print(f'Elapsed time: {round(time.time() - begin, 2)} seconds')
            return kinship_matrix

    else:
        cmatrix = cgeneakit.compute_kinships(gen, pro, verbose)
        kinship_matrix = pd.DataFrame(cmatrix, index=pro, columns=pro, copy=False)
    
    if verbose:
        end = time.time()
        print(f'Elapsed time: {round(end - begin, 2)} seconds')
    return kinship_matrix

def phiMean(kinship_matrix):
    """Calculate mean kinship coefficient excluding self-pairs
    
    Args:
        kinship_matrix (pd.DataFrame | csc_matrix): Kinship matrix from gen.phi()
        
    Returns:
        float: Mean kinship coefficient across all unique proband pairs
        
    Examples:
        >>> import geneakit as gen
        >>> from geneakit import geneaJi
        >>> ped = gen.genealogy(geneaJi)
        >>> kin_mat = gen.phi(ped)
        >>> mean_phi = gen.phiMean(kin_mat)
        >>> print(f"Average kinship: {mean_phi:.4f}")
        Average kinship: 0.1719
    """
    if issparse(kinship_matrix):
        # Handle Generic Sparse Matrix (CSR or CSC)
        total = kinship_matrix.sum()
        diag_sum = kinship_matrix.diagonal().sum()
    else:
        # Handle Dense Matrix or DataFrame
        total = kinship_matrix.sum().sum()
        # np.diag works on dense arrays; if DataFrame, need to ensure access
        if isinstance(kinship_matrix, pd.DataFrame):
            diag_sum = np.diag(kinship_matrix.values).sum()
        else:
            diag_sum = np.diag(kinship_matrix).sum()
        
    n = kinship_matrix.shape[0]
    return (total - diag_sum) / (n**2 - n)

def phiOver(phiMatrix, threshold):
    """Identify proband pairs exceeding kinship threshold
    
    Args:
        phiMatrix (pd.DataFrame): Kinship matrix from gen.phi()
        threshold (float): Minimum kinship value (0-0.5+)
        
    Returns:
        pd.DataFrame: Pairs exceeding threshold with columns:
            - pro1, pro2: Individual IDs
            - kinship: Coefficient value
            
    Examples:
        >>> import geneakit as gen
        >>> from geneakit import geneaJi
        >>> ped = gen.genealogy(geneaJi)
        >>> kin_mat = gen.phi(ped)
        >>> high_kinship = gen.phiOver(kin_mat, 0.1)
        >>> print(high_kinship)
           pro1  pro2   kinship
        0     2     1  0.371094
    """
    pairs = []
    if issparse(phiMatrix):
        # Optimize for sparse
        cx = phiMatrix.tocoo()
        for i, j, v in zip(cx.row, cx.col, cx.data):
            if i > j and v > threshold: # Lower triangle
                pairs.append({
                    'pro1': i, 'pro2': j, 'kinship': v
                })
        return pd.DataFrame(pairs)

    rows = phiMatrix.shape[0]
    for i in range(rows):
        for j in range(i):
            val = phiMatrix.iloc[i, j] if isinstance(phiMatrix, pd.DataFrame) else phiMatrix[i, j]
            if val > threshold:
                p1 = phiMatrix.index[i] if isinstance(phiMatrix, pd.DataFrame) else i
                p2 = phiMatrix.columns[j] if isinstance(phiMatrix, pd.DataFrame) else j
                pairs.append({
                    'pro1': p1,
                    'pro2': p2,
                    'kinship': val
                })
    return pd.DataFrame(pairs)

def phiCI(phiMatrix, prob=[0.025, 0.05, 0.95, 0.975], b=5000):
    """Calculate bootstrap confidence intervals for mean kinship
    
    Args:
        phiMatrix (pd.DataFrame | np.ndarray): Kinship matrix
        prob (list): Confidence probabilities (default: 95% CI)
        b (int): Bootstrap resamples (default: 5000)
        
    Returns:
        pd.DataFrame: Confidence bounds for each probability pair
        
    Examples:
        >>> import geneakit as gen
        >>> from geneakit import genea140
        >>> ped = gen.genealogy(genea140)
        >>> kin_mat = gen.phi(ped)
        >>> ci = gen.phiCI(kin_mat)
        >>> print(ci)
                  2.5%      5.0%     95.0%     97.5%
        Mean  0.000886  0.000924  0.001388  0.001442
    """
    phi_array = phiMatrix.to_numpy() if isinstance(phiMatrix, pd.DataFrame) else phiMatrix
    if issparse(phiMatrix):
        phi_array = phiMatrix.toarray() 

    n = phi_array.shape[0]
    data = (np.arange(n),)
    
    def statistic(indices):
        submatrix = phi_array[indices][:, indices]
        elements_less_than_half = submatrix[submatrix < 0.5]
        return np.mean(elements_less_than_half) if elements_less_than_half.size > 0 else np.nan
    
    sorted_prob = sorted(prob)
    quantiles = {}
    i, j = 0, len(sorted_prob) - 1
    
    while i < j:
        lower = sorted_prob[i]
        upper = sorted_prob[j]
        confidence_level = upper - lower
        
        res = bootstrap(
            data,
            statistic,
            method='percentile',
            confidence_level=confidence_level,
            n_resamples=b,
            vectorized=False
        )
        quantiles[lower] = res.confidence_interval.low
        quantiles[upper] = res.confidence_interval.high
        i += 1
        j -= 1
    
    results = [quantiles.get(p, np.nan) for p in prob]
    return pd.DataFrame([results], columns=[f"{p*100}%" for p in prob], index=["Mean"])

def f(gen, **kwargs):
    """Calculate inbreeding coefficients (F) for probands
    
    Args:
        gen (cgeneakit.Pedigree): Initialized genealogy object
        pro (list, optional): Proband IDs (default: all)
        
    Returns:
        pd.DataFrame: Inbreeding coefficients with:
            - Index: Proband IDs
            - Column: 'F' values (0-1)
            
    Examples:
        >>> import geneakit as gen
        >>> from geneakit import geneaJi
        >>> ped = gen.genealogy(geneaJi)
        >>> inbreeding = gen.f(ped)
        >>> print(inbreeding)
                   F
        1   0.183594
        2   0.183594
        29  0.070312
    """
    pro = kwargs.get('pro', None)
    if pro is None:
        pro = cgeneakit.get_proband_ids(gen)
    cmatrix = cgeneakit.compute_inbreedings(gen, pro)
    return pd.DataFrame(cmatrix, index=pro, columns=['F'], copy=False)

def fCI(vectF, prob=[0.025, 0.05, 0.95, 0.975], b=5000):
    """Calculate BCa bootstrap confidence intervals for mean inbreeding
    
    Args:
        vectF (pd.DataFrame | np.ndarray): Inbreeding coefficients
        prob (list): Confidence probabilities
        b (int): Bootstrap resamples
        
    Returns:
        pd.DataFrame: Confidence bounds using bias-corrected method
        
    Examples:
        >>> import geneakit as gen
        >>> from geneakit import geneaJi
        >>> ped = gen.genealogy(geneaJi)
        >>> inbreeding = gen.f(ped)
        >>> f_ci = gen.fCI(inbreeding)
        >>> print(f_ci)
                  2.5%      5.0%     95.0%     97.5%
        Mean  0.070312  0.070312  0.183594  0.183594
    """
    f_array = vectF.to_numpy() if isinstance(vectF, pd.DataFrame) else np.array(vectF)
    n = f_array.shape[0]
    data = (np.arange(n),)
    
    def statistic(indices):
        submatrix = f_array[indices]
        return np.mean(submatrix)
    
    sorted_prob = sorted(prob)
    quantiles = {}
    i, j = 0, len(sorted_prob) - 1
    
    while i < j:
        lower = sorted_prob[i]
        upper = sorted_prob[j]
        confidence_level = upper - lower
        
        res = bootstrap(
            data,
            statistic,
            method='BCa',
            confidence_level=confidence_level,
            n_resamples=b,
            vectorized=False
        )
        quantiles[lower] = res.confidence_interval.low
        quantiles[upper] = res.confidence_interval.high
        i += 1
        j -= 1
    
    results = [quantiles.get(p, np.nan) for p in prob]
    return pd.DataFrame([results], columns=[f"{p*100}%" for p in prob], index=["Mean"])

def gc(pedigree, **kwargs):
    """Compute genetic contribution of ancestors to probands
    
    Args:
        pedigree (cgeneakit.Pedigree): Initialized genealogy object
        pro (list, optional): Proband IDs (default: all)
        ancestors (list, optional): Founder IDs (default: all founders)
        
    Returns:
        pd.DataFrame: Contribution matrix with:
            - Rows: Proband IDs
            - Columns: Ancestor IDs
            - Values: Expected genetic contributions
            
    Examples:
        >>> import geneakit as gen
        >>> from geneakit import geneaJi
        >>> ped = gen.genealogy(geneaJi)
        >>> contributions = gen.gc(ped)
        >>> print(contributions)
                17       19      20    23        25        26
        1   0.3125  0.21875  0.0625  0.00  0.109375  0.109375
        2   0.3125  0.21875  0.0625  0.00  0.109375  0.109375
        29  0.1250  0.06250  0.2500  0.25  0.156250  0.156250
    """
    pro = kwargs.get('pro', None)
    ancestors = kwargs.get('ancestors', None)
    if pro is None:
        pro = cgeneakit.get_proband_ids(pedigree)
    if ancestors is None:
        ancestors = cgeneakit.get_founder_ids(pedigree)
        
    cmatrix = cgeneakit.compute_genetic_contributions(pedigree, pro, ancestors)
    return pd.DataFrame(cmatrix, index=pro, columns=ancestors, copy=False)