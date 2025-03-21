import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.stats import bootstrap
from scipy.stats import norm
import cgeneo
import time

# Compute

def phi(gen, **kwargs):
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
    if verbose:
        begin = time.time()
    if sparse:
        indices, indptr, data = cgeneo.compute_sparse_kinships(
            gen, pro, verbose)
        kinship_matrix = csc_matrix((data, indices, indptr),
                                    shape=(len(pro), len(pro)),
                                    dtype=np.float32)
    else:
        cmatrix = cgeneo.compute_kinships(gen, pro, verbose)
        kinship_matrix = pd.DataFrame(
            cmatrix, index=pro, columns=pro, copy=False)
    if verbose:
        end = time.time()
        elapsed_time = round(end - begin, 2)
        print(f'Elapsed time: {elapsed_time} seconds')
    return kinship_matrix

def phiMean(kinship_matrix):
    mean = 0
    if type(kinship_matrix) == csc_matrix:
        for i in range(kinship_matrix.shape[0]):
            for j in range(i):
                mean += kinship_matrix[j, i]
    elif type(kinship_matrix) == pd.DataFrame:
        for i in range(kinship_matrix.shape[0]):
            for j in range(i):
                mean += kinship_matrix.iloc[i, j]
    else:
        raise TypeError('Input must be a DataFrame or a CSC matrix.')
    mean /= kinship_matrix.shape[0] * (kinship_matrix.shape[0] - 1) / 2
    return mean

def phiOver(phiMatrix, threshold):
    pairs = []
    for i in range(phiMatrix.shape[0]):
        for j in range(i):
            if phiMatrix.iloc[i, j] > threshold:
                pairs.append([i, j, phiMatrix.index[i], phiMatrix.columns[j],
                              phiMatrix.iloc[i, j]])
    return pd.DataFrame(pairs, columns=['line', 'column', 'pro1',
                                        'pro2', 'kinship'])

def phiCI(phiMatrix, prob=[0.025, 0.05, 0.95, 0.975], b=5000):
    phi_array = phiMatrix.to_numpy() if isinstance(phiMatrix, pd.DataFrame) else np.array(phiMatrix)
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
    pro = kwargs.get('pro', None)
    if pro is None:
        pro = cgeneo.get_proband_ids(gen)
    cmatrix = cgeneo.compute_inbreedings(gen, pro)
    inbreeding_matrix = pd.DataFrame(
        cmatrix, index=pro, columns=['F'], copy=False)
    return inbreeding_matrix

def fCI(vectF, prob=[0.025, 0.05, 0.95, 0.975], b=5000):
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
    pro = kwargs.get('pro', None)
    if pro is None:
        pro = cgeneo.get_proband_ids(gen)
    verbose = kwargs.get('verbose', False)
    if verbose:
        begin = time.time()
    cmatrix = cgeneo.compute_meiotic_distances(gen, pro, verbose)
    meioses_matrix = pd.DataFrame(
        cmatrix, index=pro, columns=pro, copy=False)
    if verbose:
        end = time.time()
        elapsed_time = round(end - begin, 2)
        print(f'Elapsed time: {elapsed_time} seconds')
    return meioses_matrix

def corr(gen, **kwargs):
    pro = kwargs.get('pro', None)
    if pro is None:
        pro = cgeneo.get_proband_ids(gen)
    verbose = kwargs.get('verbose', False)
    if verbose:
        begin = time.time()
    cmatrix = cgeneo.compute_correlations(gen, pro, verbose)
    correlation_matrix = pd.DataFrame(
        cmatrix, index=pro, columns=pro, copy=False)
    if verbose:
        end = time.time()
        elapsed_time = round(end - begin, 2)
        print(f'Elapsed time: {elapsed_time} seconds')
    return correlation_matrix

def gc(pedigree, **kwargs):
    pro = kwargs.get('pro', None)
    ancestors = kwargs.get('ancestors', None)
    if pro is None:
        pro = cgeneo.get_proband_ids(pedigree)
    if ancestors is None:
        ancestors = cgeneo.get_founder_ids(pedigree)
    cmatrix = cgeneo.compute_genetic_contributions(pedigree, pro, ancestors)
    contribution_matrix = pd.DataFrame(
        cmatrix, index=pro, columns=ancestors, copy=False)
    return contribution_matrix
