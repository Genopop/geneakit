import gc as gc_
import time
import cgeneakit
import numpy as np
import pandas as pd
from numba import njit, prange, get_num_threads, get_thread_id
from scipy.sparse import csr_matrix, csc_matrix
from scipy.stats import bootstrap

# ---------------------------------------------------------
# Helper Functions (Graph Traversal)
# ---------------------------------------------------------

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
    vertex_cuts[-1] = proband_ids
    return vertex_cuts

# ---------------------------------------------------------
# Numba Optimized Kernels
# ---------------------------------------------------------

@njit(cache=True)
def build_maps(n_next, next_cut, prev_map, father_indices, mother_indices):
    """
    Constructs two mapping structures:
    1. up_map: Stores (parent1_idx, parent2_idx) for each individual in next_cut.
               Indices point to rows in the previous K matrix.
    2. down_map: A CSR-like structure mapping an index in prev_cut to a list of 
                 children indices in next_cut.
    """
    # UP MAP: n_next x 2
    # -2 means "is self/copy", -1 means "missing parent", >=0 is index in prev
    up_map = np.full((n_next, 2), -1, dtype=np.int32)
    
    # Prepare for DOWN MAP (CSR construction)
    # We need counts first
    n_prev = len(prev_map)
    down_counts = np.zeros(n_prev, dtype=np.int32)
    
    for i in range(n_next):
        ind = next_cut[i]
        
        # Check if individual exists in previous cut (pass-through/copy)
        if ind < len(prev_map) and prev_map[ind] >= 0:
            prev_idx = prev_map[ind]
            up_map[i, 0] = prev_idx
            up_map[i, 1] = -2 # Flag for Copy
            down_counts[prev_idx] += 1
        else:
            f = father_indices[ind]
            m = mother_indices[ind]
            
            f_idx = -1
            if f >= 0 and f < len(prev_map):
                f_idx = prev_map[f]
                if f_idx >= 0:
                    up_map[i, 0] = f_idx
                    down_counts[f_idx] += 1
            
            m_idx = -1
            if m >= 0 and m < len(prev_map):
                m_idx = prev_map[m]
                if m_idx >= 0:
                    up_map[i, 1] = m_idx
                    down_counts[m_idx] += 1
                    
    # Build Down Map CSR
    down_indptr = np.zeros(n_prev + 1, dtype=np.int32)
    for k in range(n_prev):
        down_indptr[k+1] = down_indptr[k] + down_counts[k]
        
    down_indices = np.empty(down_indptr[-1], dtype=np.int32)
    # Reuse down_counts as head pointers
    current_heads = down_indptr[:-1].copy() 
    
    for i in range(n_next):
        p1 = up_map[i, 0]
        p2 = up_map[i, 1]
        
        if p2 == -2: # Copy
            dest = current_heads[p1]
            down_indices[dest] = i
            current_heads[p1] += 1
        else:
            if p1 >= 0:
                dest = current_heads[p1]
                down_indices[dest] = i
                current_heads[p1] += 1
            if p2 >= 0:
                dest = current_heads[p2]
                down_indices[dest] = i
                current_heads[p2] += 1
                    
    return up_map, down_indices, down_indptr

@njit(parallel=True, cache=True)
def compute_generation_fused(
    n_next,
    up_map,
    down_indices, down_indptr,
    K_prev_indices, K_prev_indptr, K_prev_data,
    current_self_kinships,
    scratch_maps, scratch_vals, scratch_cols
):
    """
    Computes K_next = P * K_prev * P.T + D entirely in one pass using parallelism.
    
    Strategy:
    For each row 'i' in the next generation:
      1. Logic Row Merge: Identify parents (p1, p2). Logically merge K_prev[p1] and K_prev[p2].
         This creates a stream of (ancestor_idx, weight).
      2. Logic Column Scatter: For each (ancestor_idx, weight), look up the children 
         of that ancestor in the next generation (using down_map).
      3. Accumulate contributions into a thread-local dense row (SPA).
    """
        
    out_indptr = np.empty(n_next + 1, dtype=np.int64)
    out_indptr[0] = 0
        
    row_nnzs = np.zeros(n_next, dtype=np.int32)
    
    # --- PASS 1: COUNT NNZ ---
    for i in prange(n_next):
        tid = get_thread_id()
        spa_map = scratch_maps[tid]
        
        nnz_count = 0
        
        # Identify parents in prev gen
        p1 = up_map[i, 0]
        p2 = up_map[i, 1]
        is_copy = (p2 == -2)
        
        # We need to iterate over the UNION of parents' ancestors.
        # To do this without sorting/merging full arrays, we just iterate both
        # and use the SPA to deduplicate.
        
        # List of parents to process
        parents_to_scan = np.empty(2, dtype=np.int32)
        pt_count = 0
        if p1 >= 0: 
            parents_to_scan[pt_count] = p1
            pt_count += 1
        if not is_copy and p2 >= 0:
            parents_to_scan[pt_count] = p2
            pt_count += 1
            
        # Logic: Row_i(M) = 0.5*K[p1] + 0.5*K[p2]
        # Then K_next[i, :] = M[i] * P.T
        # Meaning: For every ancestor 'k' in parents' ancestry,
        # add contributions to all children of 'k'.
        
        for p_idx_i in range(pt_count):
            p_row = parents_to_scan[p_idx_i]
            start = K_prev_indptr[p_row]
            end = K_prev_indptr[p_row+1]
            
            for k_idx in range(start, end):
                ancestor = K_prev_indices[k_idx]
                # Ancestor 'ancestor' contributes.
                # Who are the children of 'ancestor' in the current generation?
                
                d_start = down_indptr[ancestor]
                d_end = down_indptr[ancestor+1]
                
                for d_k in range(d_start, d_end):
                    child = down_indices[d_k]
                    
                    # Check against 'i' to see if we visited this child for this row
                    if spa_map[child] != i:
                        spa_map[child] = i
                        nnz_count += 1
        
        # Ensure diagonal is present
        if spa_map[i] != i:
            nnz_count += 1
            
        row_nnzs[i] = nnz_count

    # Cumulative Sum for Indptr
    total_nnz = 0
    for i in range(n_next):
        out_indptr[i] = total_nnz
        total_nnz += row_nnzs[i]
    out_indptr[n_next] = total_nnz
    
    # RESET SCRATCH MAPS
    num_threads = len(scratch_maps)
    for t in prange(num_threads):
        scratch_maps[t, :].fill(-1)

    out_indices = np.empty(total_nnz, dtype=np.int64)
    out_data = np.empty(total_nnz, dtype=np.float32)
    output_diagonals = np.empty(n_next, dtype=np.float32)
    
    # --- PASS 2: FILL DATA ---
    for i in prange(n_next):
        tid = get_thread_id()
        spa_map = scratch_maps[tid]
        spa_values = scratch_vals[tid]
        local_indices = scratch_cols[tid] # Pre-allocated buffer
        
        local_ptr = 0
        
        p1 = up_map[i, 0]
        p2 = up_map[i, 1]
        is_copy = (p2 == -2)
        
        # Re-run the logic
        parents_to_scan = np.empty(2, dtype=np.int32)
        pt_count = 0
        if p1 >= 0: 
            parents_to_scan[pt_count] = p1
            pt_count += 1
        if not is_copy and p2 >= 0:
            parents_to_scan[pt_count] = p2
            pt_count += 1
            
        for p_idx_i in range(pt_count):
            p_row = parents_to_scan[p_idx_i]
            
            # Weight Logic
            # If i is copy of p1: weight is 1.0 * K[p1]
            # If i is child: weight is 0.5 * K[p1]
            scaler = 1.0 if is_copy else 0.5
            
            start = K_prev_indptr[p_row]
            end = K_prev_indptr[p_row+1]
            
            for k_idx in range(start, end):
                ancestor = K_prev_indices[k_idx]
                kinship_val = K_prev_data[k_idx]
                
                val_to_scatter = scaler * kinship_val
                
                d_start = down_indptr[ancestor]
                d_end = down_indptr[ancestor+1]
                
                for d_k in range(d_start, d_end):
                    child = down_indices[d_k]
                    
                    # Determine incoming weight to child
                    # If child is copy of ancestor: weight 1.0
                    # If child is offspring of ancestor: weight 0.5
                    # We check child's up_map
                    c_p1 = up_map[child, 0]
                    c_p2 = up_map[child, 1]
                    
                    child_scaler = 0.0
                    if c_p2 == -2: # Child is copy
                         # If child is copy, p1 must be ancestor
                         if c_p1 == ancestor: child_scaler = 1.0
                    else:
                        # Child is offspring
                        if c_p1 == ancestor: child_scaler += 0.5
                        if c_p2 == ancestor: child_scaler += 0.5
                    
                    contribution = val_to_scatter * child_scaler
                    
                    if spa_map[child] != i:
                        spa_map[child] = i
                        spa_values[child] = contribution
                        local_indices[local_ptr] = child
                        local_ptr += 1
                    else:
                        spa_values[child] += contribution
                        
        # Diagonal Correction
        # phi_ii = 0.5 + 0.5 * phi_fm (if not founder/copy)
        diag_val = 0.0
        
        if is_copy:
            # Just the calculated value (which should match prev gen self)
            if spa_map[i] == i:
                diag_val = spa_values[i]
            else:
                # Should not happen for valid copy, but safeguard
                diag_val = 0.5 
        else:
            phi_pp = 0.0
            phi_mm = 0.0
            if p1 >= 0: phi_pp = current_self_kinships[p1]
            if p2 >= 0: phi_mm = current_self_kinships[p2]
            
            raw_val = 0.0
            if spa_map[i] == i:
                raw_val = spa_values[i]
                
            # The raw_val accumulated is 0.25*phi_ff + 0.25*phi_mm + 0.5*phi_fm
            # We want 0.5 + 0.5*phi_fm
            # So: 0.5 + raw_val - 0.25*phi_ff - 0.25*phi_mm
            
            diag_val = 0.5 + raw_val - 0.25 * (phi_pp + phi_mm)

        output_diagonals[i] = diag_val
        
        if spa_map[i] != i:
            spa_map[i] = i
            local_indices[local_ptr] = i
            local_ptr += 1
        
        spa_values[i] = diag_val
        
        # Sort locally
        local_indices[:local_ptr].sort()
        
        # Write global
        global_offset = out_indptr[i]
        for k in range(local_ptr):
            col = local_indices[k]
            out_indices[global_offset + k] = col
            out_data[global_offset + k] = spa_values[col]
            
    return out_indices, out_indptr, out_data, output_diagonals

def compute_kinships_sparse(gen, pro=None, verbose=False):
    if pro is None:
        pro = cgeneakit.get_proband_ids(gen)

    n_total = len(gen)
    father_indices = np.full(n_total, -1, dtype=np.int32)
    mother_indices = np.full(n_total, -1, dtype=np.int32)
    
    # Build lookup arrays once
    for individual in gen.values():
        if individual.father.ind:
            father_indices[individual.rank] = individual.father.rank
        if individual.mother.ind:
            mother_indices[individual.rank] = individual.mother.rank

    # Cut vertices
    if verbose: print("Computing vertex cuts...")
    raw_vertex_cuts = cut_vertices(gen, pro)
    cuts_mapped = []
    for cut in raw_vertex_cuts:
        mapped = np.array([gen[id].rank for id in cut], dtype=np.int32)
        cuts_mapped.append(mapped)

    if not cuts_mapped:
        return csc_matrix((0, 0), dtype=np.float32)

    # Initialize Founders
    n_founders = len(cuts_mapped[0])
    current_data = np.full(n_founders, 0.5, dtype=np.float32)
    current_indices = np.arange(n_founders, dtype=np.int64) 
    current_indptr = np.arange(n_founders + 1, dtype=np.int64)
    current_self_kinships = np.full(n_founders, 0.5, dtype=np.float32)
    
    # Allocate Scratch Space for Numba Threads
    n_threads = get_num_threads()
    for t in range(len(cuts_mapped) - 1):
        current_cut = cuts_mapped[t]
        next_cut = cuts_mapped[t + 1]
        n_next = len(next_cut)
        n_prev = len(current_cut)

        if verbose:
            print(f"Processing cut {t+1}/{len(cuts_mapped)-1}: {n_prev} -> {n_next} individuals")

        prev_map = np.full(n_total, -1, dtype=np.int32)
        prev_map[current_cut] = np.arange(n_prev, dtype=np.int32)

        up_map, down_indices, down_indptr = build_maps(
            n_next, next_cut, prev_map, father_indices, mother_indices
        )
        
        # Allocate scratch buffers for this generation
        scratch_maps = np.full((n_threads, n_next), -1, dtype=np.int32)
        scratch_vals = np.zeros((n_threads, n_next), dtype=np.float32)
        scratch_cols = np.zeros((n_threads, n_next), dtype=np.int32)
        
        # Call the optimized function
        next_indices, next_indptr, next_data, next_self = compute_generation_fused(
            n_next,
            up_map,
            down_indices, down_indptr,
            current_indices, current_indptr, current_data,
            current_self_kinships,
            scratch_maps, scratch_vals, scratch_cols
        )
        
        current_data = next_data
        current_indices = next_indices
        current_indptr = next_indptr
        current_self_kinships = next_self
        
        del up_map, down_indices, down_indptr, next_indices, next_data, next_self
        del scratch_maps, scratch_vals, scratch_cols
        if t % 5 == 0:
            gc_.collect()

    return csr_matrix((current_data, current_indices, current_indptr), 
                      shape=(n_next, n_next))

# ---------------------------------------------------------
# Standard Interfaces (phi, phiMean, etc.)
# ---------------------------------------------------------

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
        kinship_matrix = pd.DataFrame.sparse.from_spmatrix(
            kinship_matrix, index=pro, columns=pro)
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