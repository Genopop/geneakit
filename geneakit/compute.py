import gc as gc_
import time

import cgeneakit
import numpy as np
import pandas as pd
from numba import njit
from scipy.sparse import csr_matrix, csc_matrix
from scipy.stats import bootstrap

def get_previous_generation(pedigree, ids):
    """Returns the previous generation of a set of individuals."""
    parent_set = set()
    for id in ids:
        individual = pedigree[id]
        if individual.father.ind:
            parent_set.add(individual.father.ind)
        if individual.mother.ind:
            parent_set.add(individual.mother.ind)
    return sorted(parent_set)

def get_generations(pedigree, proband_ids):
    """Go from the bottom to the top of the pedigree."""
    generations = []
    generation = proband_ids
    while generation:
        generations.append(generation)
        generation = get_previous_generation(pedigree, generation)
    return generations

def copy_bottom_up(generations):
    """Drag the individuals up (bottom-up closure)."""
    bottom_up = []
    ids = sorted(generations[0])
    bottom_up.append(ids)
    for i in range(len(generations) - 1):
        next_generation = sorted(set(bottom_up[i]) | set(generations[i + 1]))
        bottom_up.append(next_generation)
    bottom_up.reverse()
    return bottom_up

def copy_top_down(generations):
    """Drag the individuals down (top-down closure)."""
    gens_reversed = list(reversed(generations))
    top_down = []
    ids = sorted(gens_reversed[0])
    top_down.append(ids)
    for i in range(len(gens_reversed) - 1):
        next_generation = sorted(set(top_down[i]) | set(gens_reversed[i + 1]))
        top_down.append(next_generation)
    return top_down

def intersect_both_directions(bottom_up, top_down):
    """Find the intersection of the two sets (the vertex cuts)."""
    vertex_cuts = []
    for i in range(len(bottom_up)):
        vertex_cut = sorted(np.intersect1d(bottom_up[i], top_down[i]))
        vertex_cuts.append(vertex_cut)
    return vertex_cuts

def cut_vertices(pedigree, proband_ids):
    """Separate individuals into generations using recursive-cut algorithm."""
    generations = get_generations(pedigree, proband_ids)
    if not generations:
        return [proband_ids]
    bottom_up = copy_bottom_up(generations)
    top_down = copy_top_down(generations)
    vertex_cuts = intersect_both_directions(bottom_up, top_down)
    vertex_cuts[-1] = sorted(proband_ids)
    return vertex_cuts

@njit(cache=True)
def build_transfer_maps(n_next, next_cut, prev_map, father_indices, mother_indices):
    """
    Builds the P matrix (rows=next, cols=prev) and P.T matrix (rows=prev, cols=next)
    in CSR-friendly arrays.
    """
    n_prev = len(prev_map)
    
    est_size_p = n_next * 2
    p_rows = np.empty(est_size_p, dtype=np.int32)
    p_cols = np.empty(est_size_p, dtype=np.int32)
    p_data = np.empty(est_size_p, dtype=np.float32)
    
    parent_pointers = np.full((n_next, 2), -1, dtype=np.int32)
    count_p = 0

    for i in range(n_next):
        ind = next_cut[i]
        
        if ind < len(prev_map) and prev_map[ind] >= 0:
            prev_idx = prev_map[ind]
            p_rows[count_p] = i
            p_cols[count_p] = prev_idx
            p_data[count_p] = 1.0
            count_p += 1
            parent_pointers[i, 0] = -2
            parent_pointers[i, 1] = -2
        else:
            f = father_indices[ind]
            m = mother_indices[ind]
            
            if f >= 0 and f < len(prev_map):
                f_idx = prev_map[f]
                if f_idx >= 0:
                    p_rows[count_p] = i
                    p_cols[count_p] = f_idx
                    p_data[count_p] = 0.5
                    count_p += 1
                    parent_pointers[i, 0] = f_idx
            
            if m >= 0 and m < len(prev_map):
                m_idx = prev_map[m]
                if m_idx >= 0:
                    p_rows[count_p] = i
                    p_cols[count_p] = m_idx
                    p_data[count_p] = 0.5
                    count_p += 1
                    parent_pointers[i, 1] = m_idx

    p_indptr = np.zeros(n_next + 1, dtype=np.int32)
    for k in range(count_p):
        row = p_rows[k]
        p_indptr[row + 1] += 1
    
    for k in range(n_next):
        p_indptr[k+1] += p_indptr[k]
        
    final_p_indices = np.empty(count_p, dtype=np.int32)
    final_p_data = np.empty(count_p, dtype=np.float32)
    
    final_p_indices[:] = p_cols[:count_p]
    final_p_data[:] = p_data[:count_p]

    pt_counts = np.zeros(n_prev, dtype=np.int32)
    for k in range(count_p):
        col = p_cols[k]
        pt_counts[col] += 1
        
    pt_indptr = np.zeros(n_prev + 1, dtype=np.int32)
    for k in range(n_prev):
        pt_indptr[k+1] = pt_indptr[k] + pt_counts[k]
        
    pt_indices = np.empty(count_p, dtype=np.int32)
    pt_data = np.empty(count_p, dtype=np.float32)
    
    current_pt_head = pt_indptr.copy()
    
    for k in range(count_p):
        row = p_rows[k] 
        col = p_cols[k] 
        val = p_data[k]
        
        dest = current_pt_head[col]
        pt_indices[dest] = row
        pt_data[dest] = val
        current_pt_head[col] += 1
        
    return (final_p_indices, p_indptr, final_p_data, 
            pt_indices, pt_indptr, pt_data, 
            parent_pointers)

@njit(cache=True)
def compute_row_spa(
    row_idx,
    P_indices, P_indptr, P_data,
    PT_indices, PT_indptr, PT_data,
    K_prev_indices, K_prev_indptr, K_prev_data,
    spa_values, spa_markers, spa_indices, marker_key
):
    """
    Computes a single row of K_next using a Sparse Accumulator.
    Returns the number of non-zero elements found.
    """
    nnz_count = 0
    
    p_start = P_indptr[row_idx]
    p_end = P_indptr[row_idx+1]
    
    for p_k in range(p_start, p_end):
        parent_idx = P_indices[p_k]
        parent_weight = P_data[p_k]
        
        k_start = K_prev_indptr[parent_idx]
        k_end = K_prev_indptr[parent_idx+1]
        
        for k_k in range(k_start, k_end):
            ancestor_idx = K_prev_indices[k_k]
            kinship_val = K_prev_data[k_k]
            
            val = parent_weight * kinship_val
            
            pt_start = PT_indptr[ancestor_idx]
            pt_end = PT_indptr[ancestor_idx+1]
            
            for pt_k in range(pt_start, pt_end):
                child_idx = PT_indices[pt_k]
                child_weight = PT_data[pt_k]
                
                contribution = val * child_weight
                
                if spa_markers[child_idx] != marker_key:
                    spa_markers[child_idx] = marker_key
                    spa_values[child_idx] = contribution
                    spa_indices[nnz_count] = child_idx
                    nnz_count += 1
                else:
                    spa_values[child_idx] += contribution

    return nnz_count

@njit(cache=True)
def compute_generation_loop(
    n_next,
    P_indices, P_indptr, P_data,
    PT_indices, PT_indptr, PT_data,
    K_prev_indices, K_prev_indptr, K_prev_data,
    parent_pointers, current_self_kinships
):
    # Heuristic for output size
    est_nnz = n_next * 100 
    out_indices = np.empty(est_nnz, dtype=np.int32)
    out_data = np.empty(est_nnz, dtype=np.float32)
    out_indptr = np.empty(n_next + 1, dtype=np.int32)
    out_indptr[0] = 0
    
    current_ptr = 0
    
    # Sparse Accumulator Structures
    spa_values = np.zeros(n_next, dtype=np.float32)
    spa_markers = np.full(n_next, -1, dtype=np.int32)
    spa_indices = np.empty(n_next, dtype=np.int32)
    
    output_diagonals = np.empty(n_next, dtype=np.float32)
    
    for i in range(n_next):
        # Compute row using SPA
        nnz_in_row = compute_row_spa(
            i, P_indices, P_indptr, P_data, 
            PT_indices, PT_indptr, PT_data,
            K_prev_indices, K_prev_indptr, K_prev_data,
            spa_values, spa_markers, spa_indices, i
        )
        
        # Check diagonal correction
        raw_self = 0.0
        if spa_markers[i] == i:
            raw_self = spa_values[i]
        
        # Calculate Diagonal Correction
        f = parent_pointers[i, 0]
        m = parent_pointers[i, 1]
        corr_diag = raw_self
        
        if f != -2: # If not copied
            phi_pp = 0.0
            phi_mm = 0.0
            if f >= 0:
                phi_pp = current_self_kinships[f]
            if m >= 0:
                phi_mm = current_self_kinships[m]
            corr_diag = 0.5 + raw_self - 0.25 * (phi_pp + phi_mm)
        
        output_diagonals[i] = corr_diag
        
        # Update/Force diagonal in SPA
        if spa_markers[i] != i:
            spa_markers[i] = i
            spa_indices[nnz_in_row] = i
            nnz_in_row += 1
        spa_values[i] = corr_diag
        
        # Sort indices for canonical CSR format (improves locality for next step)
        # Only sort the valid portion
        current_indices_view = spa_indices[:nnz_in_row]
        current_indices_view.sort()
        
        # Harvest results to CSR buffers
        row_count = 0
        for k in range(nnz_in_row):
            col = spa_indices[k]
            val = spa_values[col]
            
            if current_ptr >= len(out_indices):
                new_size = len(out_indices) * 2
                new_ind = np.empty(new_size, dtype=np.int32)
                new_dat = np.empty(new_size, dtype=np.float32)
                new_ind[:current_ptr] = out_indices[:current_ptr]
                new_dat[:current_ptr] = out_data[:current_ptr]
                out_indices = new_ind
                out_data = new_dat
            
            out_indices[current_ptr] = col
            out_data[current_ptr] = val
            current_ptr += 1
            row_count += 1
        
        out_indptr[i+1] = out_indptr[i] + row_count

    return out_data[:current_ptr], out_indices[:current_ptr], out_indptr, output_diagonals


def compute_kinships_sparse(gen, pro=None, verbose=False):
    if pro is None:
        pro = cgeneakit.get_proband_ids(gen)

    n_total = len(gen)
    father_indices = np.full(n_total, -1, dtype=np.int32)
    mother_indices = np.full(n_total, -1, dtype=np.int32)
    for individual in gen.values():
        if individual.father.ind:
            father_indices[individual.rank] = individual.father.rank
        if individual.mother.ind:
            mother_indices[individual.rank] = individual.mother.rank

    raw_vertex_cuts = cut_vertices(gen, pro)
    cuts_mapped = []
    for cut in raw_vertex_cuts:
        mapped = np.array([gen[id].rank for id in cut], dtype=np.int32)
        cuts_mapped.append(mapped)

    if not cuts_mapped:
        return csc_matrix((0, 0), dtype=np.float32)

    n_founders = len(cuts_mapped[0])
    current_data = np.full(n_founders, 0.5, dtype=np.float32)
    current_indices = np.arange(n_founders, dtype=np.int32)
    current_indptr = np.arange(n_founders + 1, dtype=np.int32)
    current_self_kinships = np.full(n_founders, 0.5, dtype=np.float32)

    for t in range(len(cuts_mapped) - 1):
        current_cut = cuts_mapped[t]
        next_cut = cuts_mapped[t + 1]
        n_prev = len(current_cut)
        n_next = len(next_cut)

        if verbose:
            print(f"Processing cut {t+1}/{len(cuts_mapped)-1}: {n_prev} -> {n_next} individuals")

        prev_map = np.full(n_total, -1, dtype=np.int32)
        prev_map[current_cut] = np.arange(n_prev, dtype=np.int32)

        (P_ind, P_ptr, P_dat, 
         PT_ind, PT_ptr, PT_dat, 
         parent_pointers) = build_transfer_maps(
            n_next, next_cut, prev_map, father_indices, mother_indices
        )
        
        next_data, next_indices, next_indptr, next_self = compute_generation_loop(
            n_next,
            P_ind, P_ptr, P_dat,
            PT_ind, PT_ptr, PT_dat,
            current_indices, current_indptr, current_data,
            parent_pointers, current_self_kinships
        )
        
        current_data = next_data
        current_indices = next_indices
        current_indptr = next_indptr
        current_self_kinships = next_self
        
        del P_ind, P_ptr, P_dat, PT_ind, PT_ptr, PT_dat
        gc_.collect()

    return csr_matrix((current_data, current_indices, current_indptr), 
                      shape=(n_next, n_next))

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
    
    if verbose:
        begin = time.time()
        
    if sparse:
        kinship_matrix = compute_kinships_sparse(gen, pro=pro, verbose=verbose)
        pro_sorted = sorted(list(pro))
        kinship_matrix = pd.DataFrame.sparse.from_spmatrix(
            kinship_matrix, index=pro_sorted, columns=pro_sorted
        )
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