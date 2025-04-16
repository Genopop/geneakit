import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.stats import bootstrap
from scipy.stats import norm
import cgeneo
import time

def phi(gen, **kwargs):
    """Compute kinship coefficients between probands
    
    Calculates pairwise kinship coefficients (φ) measuring the probability
    that two individuals share alleles identical by descent from common ancestors.

    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        pro (list, optional): Proband IDs to include (default: all)
        verbose (bool): Print computation details if True (default: False)
        compute (bool): Estimate memory if False (default: True)
        sparse (bool): Return sparse matrix if True (default: False)

    Returns:
        pd.DataFrame | csc_matrix: Kinship matrix with:
            - Rows/columns: Proband IDs
            - Values: Kinship coefficients (0-0.5+)

    Examples:
        >>> import geneo as gen
        >>> from geneo import geneaJi
        >>> ped = gen.genealogy(geneaJi)
        >>> kin_mat = gen.phi(ped)
        >>> print(kin_mat)
                  1         2         29
        1   0.591797  0.371094  0.072266
        2   0.371094  0.591797  0.072266
        29  0.072266  0.072266  0.535156
    """
    pro = kwargs.get('pro', None)
    if pro is None:
        pro = cgeneo.get_proband_ids(gen)
    verbose = kwargs.get('verbose', False)
    compute = kwargs.get('compute', True)
    sparse = kwargs.get('sparse', False)
    
    if not compute:
        required_memory = cgeneo.get_required_memory_for_kinships(gen, pro)
        required_memory = round(required_memory, 2)
        print(f'You will require at least {required_memory} GB of RAM.')
        return
    
    if verbose: begin = time.time()
        
    if sparse:
        indices, indptr, data = cgeneo.compute_sparse_kinships(gen, pro, verbose)
        kinship_matrix = csc_matrix((data, indices, indptr),
                                    shape=(len(pro), len(pro)),
                                    dtype=np.float32)
    else:
        cmatrix = cgeneo.compute_kinships(gen, pro, verbose)
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
        >>> import geneo as gen
        >>> from geneo import geneaJi
        >>> ped = gen.genealogy(geneaJi)
        >>> kin_mat = gen.phi(ped)
        >>> mean_phi = gen.phiMean(kin_mat)
        >>> print(f"Average kinship: {mean_phi:.4f}")
        Average kinship: 0.1719
    """
    if isinstance(kinship_matrix, csc_matrix):
        total = kinship_matrix.sum()
        diag_sum = kinship_matrix.diagonal().sum()
    else:
        total = kinship_matrix.sum().sum()
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
        >>> import geneo as gen
        >>> from geneo import geneaJi
        >>> ped = gen.genealogy(geneaJi)
        >>> kin_mat = gen.phi(ped)
        >>> high_kinship = gen.phiOver(kin_mat, 0.1)
        >>> print(high_kinship)
           pro1  pro2   kinship
        0     2     1  0.371094
    """
    pairs = []
    for i in range(phiMatrix.shape[0]):
        for j in range(i):
            if phiMatrix.iloc[i, j] > threshold:
                pairs.append({
                    'pro1': phiMatrix.index[i],
                    'pro2': phiMatrix.columns[j],
                    'kinship': phiMatrix.iloc[i, j]
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
        >>> import geneo as gen
        >>> from geneo import genea140
        >>> ped = gen.genealogy(genea140)
        >>> kin_mat = gen.phi(ped)
        >>> ci = gen.phiCI(kin_mat)
        >>> print(ci)
                  2.5%      5.0%     95.0%     97.5%
        Mean  0.000886  0.000924  0.001388  0.001442
    """
    phi_array = phiMatrix.to_numpy() if isinstance(phiMatrix, pd.DataFrame) else phiMatrix
    n = phi_array.shape[0]
    
    # Prepare indices as input data for bootstrap resampling
    data = (np.arange(n),)
    
    # Define statistic function to compute mean of elements < 0.5 in resampled submatrix
    def statistic(indices):
        submatrix = phi_array[indices][:, indices]
        elements_less_than_half = submatrix[submatrix < 0.5]
        return np.mean(elements_less_than_half) if elements_less_than_half.size > 0 else np.nan
    
    # Pair probabilities into confidence intervals
    sorted_prob = sorted(prob)
    quantiles = {}
    i, j = 0, len(sorted_prob) - 1
    
    while i < j:
        lower = sorted_prob[i]
        upper = sorted_prob[j]
        confidence_level = upper - lower
        
        # Compute confidence interval using bootstrap
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
    
    # Compile results in original probability order
    results = [quantiles.get(p, np.nan) for p in prob]
    
    return pd.DataFrame(
        [results],
        columns=[f"{p*100}%" for p in prob],
        index=["Mean"]
    )

def f(gen, **kwargs):
    """Calculate inbreeding coefficients (F) for probands
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        pro (list, optional): Proband IDs (default: all)
        
    Returns:
        pd.DataFrame: Inbreeding coefficients with:
            - Index: Proband IDs
            - Column: 'F' values (0-1)
            
    Examples:
        >>> import geneo as gen
        >>> from geneo import geneaJi
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
        pro = cgeneo.get_proband_ids(gen)
    cmatrix = cgeneo.compute_inbreedings(gen, pro)
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
        >>> import geneo as gen
        >>> from geneo import geneaJi
        >>> ped = gen.genealogy(geneaJi)
        >>> inbreeding = gen.f(ped)
        >>> f_ci = gen.fCI(inbreeding)
        >>> print(f_ci)
                  2.5%      5.0%     95.0%     97.5%
        Mean  0.070312  0.070312  0.183594  0.183594
    """
    f_array = vectF.to_numpy() if isinstance(vectF, pd.DataFrame) else np.array(vectF)
    n = f_array.shape[0]
    
    # Prepare indices as input data for bootstrap resampling
    data = (np.arange(n),)
    
    # Define statistic function to compute mean of elements < 0.5 in resampled submatrix
    def statistic(indices):
        submatrix = f_array[indices]
        return np.mean(submatrix)
    
    # Pair probabilities into confidence intervals
    sorted_prob = sorted(prob)
    quantiles = {}
    i, j = 0, len(sorted_prob) - 1
    
    while i < j:
        lower = sorted_prob[i]
        upper = sorted_prob[j]
        confidence_level = upper - lower
        
        # Compute confidence interval using bootstrap
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
    
    # Compile results in original probability order
    results = [quantiles.get(p, np.nan) for p in prob]
    
    return pd.DataFrame(
        [results],
        columns=[f"{p*100}%" for p in prob],
        index=["Mean"]
    )

def meioses(gen, **kwargs):
    """Compute meiotic distances between probands
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        pro (list, optional): Proband IDs (default: all)
        verbose (bool): Print computing info if True
        
    Returns:
        pd.DataFrame: Distance matrix with:
            - Rows/columns: Proband IDs
            - Values: Minimum meioses between pairs
            
    Examples:
        >>> import geneo as gen
        >>> from geneo import geneaJi
        >>> ped = gen.genealogy(geneaJi)
        >>> dist_mat = gen.meioses(ped)
        >>> print(dist_mat)
            1   2   29
        1    0   2   7
        2    2   0   7
        29   7   7   0
    """
    pro = kwargs.get('pro', None)
    if pro is None:
        pro = cgeneo.get_proband_ids(gen)
    verbose = kwargs.get('verbose', False)
    
    if verbose: begin = time.time()
    cmatrix = cgeneo.compute_meiotic_distances(gen, pro, verbose)
    if verbose: 
        print(f"Time: {time.time()-begin:.2f}s")
        
    return pd.DataFrame(cmatrix, index=pro, columns=pro, copy=False)

def corr(gen, **kwargs):
    """Calculate genetic correlation matrix
    
    Args:
        gen (cgeneo.Pedigree): Initialized genealogy object
        pro (list, optional): Proband IDs (default: all)
        verbose (bool): Print computing info if True
        
    Returns:
        pd.DataFrame: Correlation matrix with:
            - Rows/columns: Proband IDs
            - Values: Genetic correlation coefficients
            
    Notes:
        Correlations represent shared ancestry proportion
    """
    pro = kwargs.get('pro', None)
    if pro is None:
        pro = cgeneo.get_proband_ids(gen)
    verbose = kwargs.get('verbose', False)
    
    if verbose: begin = time.time()
    cmatrix = cgeneo.compute_correlations(gen, pro, verbose)
    if verbose: print(f"Time: {time.time()-begin:.2f}s")
        
    return pd.DataFrame(cmatrix, index=pro, columns=pro, copy=False)

def gc(pedigree, **kwargs):
    """Compute genetic contribution of ancestors to probands
    
    Args:
        pedigree (cgeneo.Pedigree): Initialized genealogy object
        pro (list, optional): Proband IDs (default: all)
        ancestors (list, optional): Founder IDs (default: all founders)
        
    Returns:
        pd.DataFrame: Contribution matrix with:
            - Rows: Proband IDs
            - Columns: Ancestor IDs
            - Values: Expected genetic contributions
            
    Examples:
        >>> import geneo as gen
        >>> from geneo import geneaJi
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
        pro = cgeneo.get_proband_ids(pedigree)
    if ancestors is None:
        ancestors = cgeneo.get_founder_ids(pedigree)
        
    cmatrix = cgeneo.compute_genetic_contributions(pedigree, pro, ancestors)
    return pd.DataFrame(cmatrix, index=pro, columns=ancestors, copy=False)