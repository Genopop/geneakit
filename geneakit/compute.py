import time
import cgeneakit
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix, issparse
from scipy.stats import bootstrap

def phi(gen, **kwargs):
    """Compute kinship coefficients between probands.
    
    Args:
        gen (cgeneakit.Pedigree): Initialized genealogy object.
        pro (list, optional): Proband IDs. Defaults to all probands.
        verbose (bool, default False): Print details.
        compute (bool, default True): Estimate memory if False.
        sparse (bool, default False): Use sparse computation algorithm (experimental!).
        raw (bool, default False): If True:
            - Sparse mode: returns (csr_matrix, ids). Matrix is LOWER TRIANGULAR.
            - Dense mode: returns (ndarray, ids).
            If False (default): returns pd.DataFrame (SparseDataFrame if sparse=True).
    
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
        if raw:
            # Mode 1: Raw output (Lower Triangular CSR)
            # Call C++ with symmetric_coo=False
            data, indices, indptr = cgeneakit.compute_kinships_sparse(
                gen, pro, verbose, False
            )
            
            lt_matrix = csr_matrix(
                (data, indices, indptr), 
                shape=(len(pro), len(pro)),
                copy=False
            )
            
            if verbose:
                print(f'Elapsed time: {round(time.time() - begin, 2)} seconds')
            return lt_matrix, pro
            
        else:
            # Mode 2: DataFrame output (Symmetrical COO)
            # Call C++ with symmetric_coo=True
            # Returns (data, rows, cols) directly
            data, rows, cols = cgeneakit.compute_kinships_sparse(
                gen, pro, verbose, True
            )
            
            # Create Symmetric COO Matrix directly from C++ vectors
            sym_matrix = coo_matrix(
                (data, (rows, cols)), 
                shape=(len(pro), len(pro))
            )
            
            # Create Sparse DataFrame directly
            kinship_matrix = pd.DataFrame.sparse.from_spmatrix(
                sym_matrix, index=pro, columns=pro
            )
            
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
    """Calculate mean kinship coefficient excluding self-pairs"""
    if issparse(kinship_matrix) or hasattr(kinship_matrix, "sparse"):
        if isinstance(kinship_matrix, pd.DataFrame):
             total = kinship_matrix.sum().sum()
             # Approximate access for DataFrame to avoid densifying
             # Assuming standard extraction:
             diag_sum = np.diag(kinship_matrix).sum() 
        else:
            total = kinship_matrix.sum()
            diag_sum = kinship_matrix.diagonal().sum()
    else:
        total = kinship_matrix.sum().sum()
        if isinstance(kinship_matrix, pd.DataFrame):
            diag_sum = np.diag(kinship_matrix.values).sum()
        else:
            diag_sum = np.diag(kinship_matrix).sum()
        
    n = kinship_matrix.shape[0]
    return (total - diag_sum) / (n**2 - n)

def phiOver(phiMatrix, threshold):
    """Identify proband pairs exceeding kinship threshold"""
    pairs = []
    
    is_sp = issparse(phiMatrix) or hasattr(phiMatrix, "sparse")
    
    if is_sp:
        if isinstance(phiMatrix, pd.DataFrame):
            cx = phiMatrix.sparse.to_coo()
            idx_map = phiMatrix.index
        else:
            cx = phiMatrix.tocoo()
            idx_map = None

        for i, j, v in zip(cx.row, cx.col, cx.data):
            if i > j and v > threshold: 
                pairs.append({
                    'pro1': idx_map[i] if idx_map is not None else i, 
                    'pro2': idx_map[j] if idx_map is not None else j, 
                    'kinship': v
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
    """Calculate bootstrap confidence intervals for mean kinship"""
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
    """Calculate inbreeding coefficients (F) for probands"""
    pro = kwargs.get('pro', None)
    if pro is None:
        pro = cgeneakit.get_proband_ids(gen)
    cmatrix = cgeneakit.compute_inbreedings(gen, pro)
    return pd.DataFrame(cmatrix, index=pro, columns=['F'], copy=False)

def fCI(vectF, prob=[0.025, 0.05, 0.95, 0.975], b=5000):
    """Calculate BCa bootstrap confidence intervals for mean inbreeding"""
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
    """Compute genetic contribution of ancestors to probands"""
    pro = kwargs.get('pro', None)
    ancestors = kwargs.get('ancestors', None)
    if pro is None:
        pro = cgeneakit.get_proband_ids(pedigree)
    if ancestors is None:
        ancestors = cgeneakit.get_founder_ids(pedigree)
        
    cmatrix = cgeneakit.compute_genetic_contributions(pedigree, pro, ancestors)
    return pd.DataFrame(cmatrix, index=pro, columns=ancestors, copy=False)