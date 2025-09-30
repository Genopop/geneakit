import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.stats import bootstrap
from numba import njit, prange
import cgeneakit
import time

# Maximally JIT-compiled helper functions
@njit(parallel=False, cache=True, fastmath=True)
def compute_D_vector(n, sire, dam, F):
    """Fully JIT-compiled D vector computation"""
    D = np.ones(n, dtype=np.float64)
    
    for i in prange(n):
        if sire[i] >= 0 and dam[i] >= 0:
            # Both parents known
            D[i] = 0.5 - 0.25 * (F[sire[i]] + F[dam[i]])
        elif sire[i] >= 0 and dam[i] < 0:
            # Only sire known
            D[i] = 0.75 - 0.25 * F[sire[i]]
        elif dam[i] >= 0 and sire[i] < 0:
            # Only dam known
            D[i] = 0.75 - 0.25 * F[dam[i]]
        # else: D[i] remains 1.0 (no parents)
    
    return D

@njit(cache=True, fastmath=True)
def backward_solve_optimized(z, sire, dam, n):
    """Optimized backward substitution with better memory access pattern"""
    # Process in reverse order for (I - T')z = e_j
    for i in range(n-1, -1, -1):
        if z[i] != 0.0:
            z_half = 0.5 * z[i]
            if sire[i] >= 0:
                z[sire[i]] += z_half
            if dam[i] >= 0:
                z[dam[i]] += z_half
    return z

@njit(cache=True, fastmath=True)
def forward_solve_optimized(y, sire, dam, n):
    """Optimized forward substitution with better memory access pattern"""
    # Process in forward order for (I - T)y = w
    for i in range(n):
        parent_contrib = 0.0
        if sire[i] >= 0:
            parent_contrib += y[sire[i]]
        if dam[i] >= 0:
            parent_contrib += y[dam[i]]
        if parent_contrib != 0.0:
            y[i] += 0.5 * parent_contrib
    return y

@njit(cache=True, fastmath=True)
def compute_column_dense(j, n, sire, dam, D, pro_idx):
    """Compute a single column of A matrix - fully JIT compiled"""
    # Initialize work arrays
    z = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    
    # Set unit vector
    z[j] = 1.0
    
    # Step 1: Backward substitution
    z = backward_solve_optimized(z, sire, dam, n)
    
    # Step 2: Apply D
    for i in range(n):
        y[i] = D[i] * z[i]
    
    # Step 3: Forward substitution
    y = forward_solve_optimized(y, sire, dam, n)
    
    # Extract values for probands only
    n_pro = len(pro_idx)
    result = np.zeros(n_pro, dtype=np.float64)
    for i in range(n_pro):
        result[i] = y[pro_idx[i]]
    
    return result

@njit(parallel=True, cache=True, fastmath=True)
def compute_A_matrix_parallel(n, n_pro, sire, dam, D, pro_idx):
    """Fully JIT-compiled parallel computation of entire A matrix"""
    A_matrix = np.zeros((n_pro, n_pro), dtype=np.float64)
    
    # Process columns in parallel
    for j_idx in prange(n_pro):
        j = pro_idx[j_idx]
        
        # Compute column j
        col_values = compute_column_dense(j, n, sire, dam, D, pro_idx)
        
        # Store in matrix
        for i_idx in range(n_pro):
            A_matrix[i_idx, j_idx] = col_values[i_idx]
    
    return A_matrix

@njit(cache=True)
def compute_A_matrix_sequential(n, n_pro, sire, dam, D, pro_idx):
    """Sequential version for comparison or when parallel isn't beneficial"""
    A_matrix = np.zeros((n_pro, n_pro), dtype=np.float64)
    
    for j_idx in range(n_pro):
        j = pro_idx[j_idx]
        col_values = compute_column_dense(j, n, sire, dam, D, pro_idx)
        A_matrix[:, j_idx] = col_values
    
    return A_matrix

@njit(cache=True)
def compute_sparse_entries(n, n_pro, sire, dam, D, pro_idx):
    """Compute sparse matrix entries - fully JIT compiled"""
    nnzero = 0
    # First pass to count non-zero entries
    for j_idx in range(n_pro):
        j = pro_idx[j_idx]
        col_values = compute_column_dense(j, n, sire, dam, D, pro_idx)
        for i_idx in range(n_pro):
            if col_values[i_idx] > 0.0:
                nnzero += 1
    
    # Pre-allocate lists for CSC format
    data = np.zeros(nnzero, dtype=np.float64)
    indices = np.zeros(nnzero, dtype=np.int32)
    indptr = np.zeros(n_pro+1, dtype=np.int32)
    entry_count = 0
    
    for j_idx in range(n_pro):
        j = pro_idx[j_idx]
        col_values = compute_column_dense(j, n, sire, dam, D, pro_idx)
        
        for i_idx in range(n_pro):
            val = col_values[i_idx]
            if val > 0.0:
                indices[entry_count] = i_idx
                data[entry_count] = val
                entry_count += 1

        indptr[j_idx + 1] = entry_count
    
    # Return only used portion
    return (data, indices, indptr)

# Additional optimization: Batch processing for very large pedigrees
@njit(parallel=True, cache=True)
def compute_A_matrix_blocked(n, n_pro, sire, dam, D, pro_idx, block_size=64):
    """Block-wise computation for better cache utilization"""
    A_matrix = np.zeros((n_pro, n_pro), dtype=np.float64)
    
    # Process in blocks for better cache locality
    n_blocks = (n_pro + block_size - 1) // block_size
    
    for block_j in prange(n_blocks):
        j_start = block_j * block_size
        j_end = min(j_start + block_size, n_pro)
        
        for j_idx in range(j_start, j_end):
            j = pro_idx[j_idx]
            col_values = compute_column_dense(j, n, sire, dam, D, pro_idx)
            
            # Store results
            for i_idx in range(n_pro):
                A_matrix[i_idx, j_idx] = col_values[i_idx]
    
    return A_matrix

def A(gen, **kwargs):
    """Compute additive genetic relationship matrix using maximally JIT-optimized Colleau's method
    
    This version maximizes JIT compilation for optimal performance.
    All core computations are JIT-compiled with Numba.
    
    Args:
        gen (cgeneakit.Pedigree): Initialized genealogy object
        pro (list, optional): Proband IDs to include (default: all)
        sparse (bool, optional): Return sparse matrix if True (default: False)
        verbose (bool, optional): Print timing info if True (default: False)
        parallel (bool, optional): Use parallel computation if True (default: True)
        block_size (int, optional): Block size for cache optimization (default: 64)
        
    Returns:
        pd.DataFrame or scipy.sparse.csc_matrix: Additive relationship matrix
    """
    
    # Get parameters
    pro = kwargs.get('pro', None)
    if pro is None:
        pro = cgeneakit.get_proband_ids(gen)
    sparse = kwargs.get('sparse', False)
    verbose = kwargs.get('verbose', False)
    parallel = kwargs.get('parallel', True)
    block_size = kwargs.get('block_size', 64)
    
    if verbose: 
        begin = time.time()
        print(f"Computing A matrix for {len(pro)} probands...")
    
    # Get all individuals and create mapping
    all_ids = list(gen.keys())
    n = len(all_ids)
    id_to_idx = {id_val: idx for idx, id_val in enumerate(all_ids)}
    
    # Get proband indices
    pro_idx = np.array([id_to_idx[p] for p in pro if p in id_to_idx], dtype=np.int32)
    n_pro = len(pro_idx)
    
    if verbose:
        print(f"Total individuals: {n}, Probands: {n_pro}")
    
    # Build parent indices - optimized with NumPy
    ped = cgeneakit.output_pedigree(gen)
    sire = np.full(n, -1, dtype=np.int32)
    dam = np.full(n, -1, dtype=np.int32)
    
    # Vectorized parent index assignment
    for i in range(n):
        if ped[i,1] > 0 and ped[i,1] in id_to_idx:
            sire[i] = id_to_idx[ped[i,1]]
        if ped[i,2] > 0 and ped[i,2] in id_to_idx:
            dam[i] = id_to_idx[ped[i,2]]
    
    # Get inbreeding coefficients
    F = np.array(cgeneakit.compute_inbreedings(gen, all_ids), dtype=np.float64)
    
    # Compute D vector using JIT-compiled function
    if verbose:
        d_start = time.time()
    D = compute_D_vector(n, sire, dam, F)
    if verbose:
        print(f"D vector computed in {time.time()-d_start:.3f}s")
    
    # Compute A matrix
    if verbose:
        matrix_start = time.time()
    
    if sparse:
        # Compute sparse entries
        data, indices, indptr = compute_sparse_entries(
            n, n_pro, sire, dam, D, pro_idx
        )
        
        # Create sparse matrix
        A_sparse = csc_matrix((data, indices, indptr), copy=False)
        result = A_sparse.tocsc()
        
    else:
        # Choose computation strategy based on size and settings
        if n_pro < 100:
            # Small matrices: sequential is often faster
            A_matrix = compute_A_matrix_sequential(n, n_pro, sire, dam, D, pro_idx)
        elif parallel and n_pro > 500:
            # Large matrices with parallel enabled: use blocked parallel
            A_matrix = compute_A_matrix_blocked(
                n, n_pro, sire, dam, D, pro_idx, block_size
            )
        elif parallel:
            # Medium matrices with parallel: use simple parallel
            A_matrix = compute_A_matrix_parallel(n, n_pro, sire, dam, D, pro_idx)
        else:
            # Parallel disabled: use sequential
            A_matrix = compute_A_matrix_sequential(n, n_pro, sire, dam, D, pro_idx)
        
        result = pd.DataFrame(A_matrix, index=pro, columns=pro)
    
    if verbose:
        matrix_time = time.time() - matrix_start
        total_time = time.time() - begin
        print(f"Matrix computation: {matrix_time:.3f}s")
        print(f"Total time: {total_time:.3f}s")
    
    return result
        
def phi(gen, **kwargs):
    """Compute kinship coefficients between probands
    
    Calculates pairwise kinship coefficients (φ) measuring the probability
    that two individuals share alleles identical by descent from common ancestors.

    Args:
        gen (cgeneakit.Pedigree): Initialized genealogy object
        pro (list, optional): Proband IDs to include (default: all)
        verbose (bool): Print computation details if True (default: False)
        compute (bool): Estimate memory if False (default: True)
        sparse (bool): Return sparse matrix if True (default: False)

    Returns:
        pd.DataFrame | csc_matrix: Kinship matrix with:
            - Rows/columns: Proband IDs
            - Values: Kinship coefficients (0-0.5+)

    Examples:
        >>> import geneakit as gen
        >>> from geneakit import geneaJi
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
        pro = cgeneakit.get_proband_ids(gen)
    verbose = kwargs.get('verbose', False)
    compute = kwargs.get('compute', True)
    sparse = kwargs.get('sparse', False)
    
    if not compute:
        required_memory = cgeneakit.get_required_memory_for_kinships(gen, pro)
        required_memory = round(required_memory, 2)
        print(f'You will require at least {required_memory} GB of RAM.')
        return
    
    if verbose: begin = time.time()
        
    if sparse:
        kinship_matrix = A(gen, pro=pro, sparse=True, verbose=verbose) / 2
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
        gen (cgeneakit.Pedigree): Initialized genealogy object
        pro (list, optional): Proband IDs (default: all)
        verbose (bool): Print computing info if True
        
    Returns:
        pd.DataFrame: Distance matrix with:
            - Rows/columns: Proband IDs
            - Values: Minimum meioses between pairs
            
    Examples:
        >>> import geneakit as gen
        >>> from geneakit import geneaJi
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
        pro = cgeneakit.get_proband_ids(gen)
    verbose = kwargs.get('verbose', False)
    
    if verbose: begin = time.time()
    cmatrix = cgeneakit.compute_meiotic_distances(gen, pro, verbose)
    if verbose: 
        print(f"Time: {time.time()-begin:.2f}s")
        
    return pd.DataFrame(cmatrix, index=pro, columns=pro, copy=False)

def corr(gen, **kwargs):
    """Calculate genetic correlation matrix
    
    Args:
        gen (cgeneakit.Pedigree): Initialized genealogy object
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
        pro = cgeneakit.get_proband_ids(gen)
    verbose = kwargs.get('verbose', False)
    
    if verbose: begin = time.time()
    cmatrix = cgeneakit.compute_correlations(gen, pro, verbose)
    if verbose: print(f"Time: {time.time()-begin:.2f}s")
        
    return pd.DataFrame(cmatrix, index=pro, columns=pro, copy=False)

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
