import time
import cgeneakit
from numba import njit
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix, diags
from scipy.stats import bootstrap

def get_previous_generation(pedigree, ids):
    """Returns the previous generation of a set of individuals."""
    parent_set = set()
    for id in ids:
        individual = pedigree[id]
        if individual.father and individual.father.ind:
            parent_set.add(individual.father.ind)
        if individual.mother and individual.mother.ind:
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
def build_transfer_matrix_jit(n_next, next_cut, prev_map, father_indices, mother_indices):
    """
    Build COO arrays for the transfer matrix P using float64 to save memory.
    """
    est_size = n_next * 2
    rows = np.empty(est_size, dtype=np.int32)
    cols = np.empty(est_size, dtype=np.int32)
    data = np.empty(est_size, dtype=np.float64)
    count = 0

    parent_pointers = np.full((n_next, 2), -1, dtype=np.int32)

    for i in range(n_next):
        ind = next_cut[i]  # local index (0..n_total-1)

        # If this individual is present in previous cut
        if prev_map[ind] >= 0:
            prev_idx = prev_map[ind]
            rows[count] = i
            cols[count] = prev_idx
            data[count] = 1.0
            count += 1
            # Mark as copied individual
            parent_pointers[i, 0] = -2
            parent_pointers[i, 1] = -2
        else:
            # New individual
            father = father_indices[ind]
            mother = mother_indices[ind]

            # Father contribution
            if father >= 0:
                father_idx = prev_map[father]
                rows[count] = i
                cols[count] = father_idx
                data[count] = 0.5
                count += 1
                parent_pointers[i, 0] = father_idx

            # Mother contribution
            if mother >= 0:
                mother_idx = prev_map[mother]
                rows[count] = i
                cols[count] = mother_idx
                data[count] = 0.5
                count += 1
                parent_pointers[i, 1] = mother_idx

    return data[:count], rows[:count], cols[:count], parent_pointers

@njit(cache=True)
def compute_diagonal_correction(n_next, parent_pointers, current_self_kinships, computed_diagonals):
    """
    Compute corrected diagonal values for K_next.

    parent_pointers flags:
      - (-2, -2) => copied individual: keep computed diagonal
      - else => new individual: compute true diagonal using:
           true = 0.5 + (raw - 0.25*(phi_pp + phi_mm))
       where raw = 0.25 phi_pp + 0.25 phi_mm + 0.5 phi_pm
    If a parent is missing from the previous cut, its contribution to raw was 0.
    """
    new_self_kinships = np.empty(n_next, dtype=np.float64)

    for i in range(n_next):
        father_idx = parent_pointers[i, 0]
        mother_idx = parent_pointers[i, 1]

        if father_idx == -2 and mother_idx == -2:
            # copied individual
            new_self_kinships[i] = computed_diagonals[i]
        else:
            phi_pp = 0.0
            phi_mm = 0.0

            if father_idx >= 0:
                phi_pp = current_self_kinships[father_idx]
            if mother_idx >= 0:
                phi_mm = current_self_kinships[mother_idx]

            # raw diagonal produced by P K_prev P^T
            value = computed_diagonals[i]

            # true diagonal (for a child) = 0.5 + 0.5*phi_pm
            # but phi_pm = (value - 0.25*phi_pp - 0.25*phi_mm)/0.5
            # so true = 0.5 + (value - 0.25*(phi_pp + phi_mm))
            true_value = 0.5 + value - 0.25 * (phi_pp + phi_mm)

            new_self_kinships[i] = true_value

    return new_self_kinships

def prune_matrix(mat, threshold):
    """In-place pruning of sparse matrix values below threshold."""
    if threshold > 0 and mat.nnz > 0:
        # Access the data array directly
        mat.data[mat.data < threshold] = 0
        mat.eliminate_zeros()
    return mat

def compute_kinships_sparse(gen, pro=None, verbose=False, threshold=1e-9):
    """Sparse kinship matrix computation using matrix multiplication (P K P')."""

    if pro is None:
        pro = cgeneakit.get_proband_ids(gen)

    # 1) Global compact reindex
    all_ids = list(gen.keys())
    id_to_idx = {id: i for i, id in enumerate(all_ids)}
    n_total = len(all_ids)

    # 2) Build parent index arrays
    father_indices = np.full(n_total, -1, dtype=np.int32)
    mother_indices = np.full(n_total, -1, dtype=np.int32)
    for i, id in enumerate(all_ids):
        ind = gen[id]
        if ind.father is not None and ind.father.ind in id_to_idx:
            father_indices[i] = id_to_idx[ind.father.ind]
        if ind.mother is not None and ind.mother.ind in id_to_idx:
            mother_indices[i] = id_to_idx[ind.mother.ind]

    # 3) Vertex cuts
    raw_vertex_cuts = cut_vertices(gen, pro)
    cuts_mapped = []
    for cut in raw_vertex_cuts:
        mapped = np.array([id_to_idx[idv] for idv in cut if idv in id_to_idx], dtype=np.int32)
        cuts_mapped.append(mapped)

    if not cuts_mapped:
        return csc_matrix((0, 0), dtype=np.float64)

    # 4) Initialize founders
    n_founders = len(cuts_mapped[0])
    current_matrix = diags([0.5] * n_founders, shape=(n_founders, n_founders), 
                          format='csr', dtype=np.float64)
    current_self_kinships = np.full(n_founders, 0.5, dtype=np.float64)

    # 5) Iterate through vertex cuts
    for t in range(len(cuts_mapped) - 1):
        current_cut = cuts_mapped[t]
        next_cut = cuts_mapped[t + 1]
        n_curr = len(current_cut)
        n_next = len(next_cut)

        if verbose:
            print(f"Processing cut {t+1}/{len(cuts_mapped)-1}: {n_curr} -> {n_next} individuals")

        # Build prev_map
        prev_map = np.full(n_total, -1, dtype=np.int32)
        for k in range(n_curr):
            prev_map[current_cut[k]] = k

        # Build transfer matrix P
        p_data, p_rows, p_cols, parent_pointers = build_transfer_matrix_jit(
            n_next, next_cut, prev_map, father_indices, mother_indices
        )

        # Use CSR for P because P.dot(K) (row-based op)
        P = csr_matrix((p_data, (p_rows, p_cols)), shape=(n_next, n_curr))

        # --- MEMORY OPTIMIZED MULTIPLICATION ---
        
        # Step A: Intermediate multiplication P * K_prev
        # This produces an N_next x N_curr matrix
        K_temp = P.dot(current_matrix)
        
        # Prune the intermediate matrix immediately.
        prune_matrix(K_temp, threshold)

        # Step B: Final multiplication K_temp * P.T
        # This produces an N_next x N_next matrix
        # P.T is CSC (efficient column slicing), K_temp is CSR (efficient row slicing).
        K_next_raw = K_temp.dot(P.T)

        # Free heavy memory immediately
        del K_temp
        del P
        
        # --- DIAGONAL CORRECTION (In-place) ---
        
        computed_diagonals = K_next_raw.diagonal()
        corrected_diagonals = compute_diagonal_correction(
            n_next, parent_pointers, current_self_kinships, computed_diagonals
        )

        # Overwrite diagonal in-place (no new allocation)
        K_next_raw.setdiag(corrected_diagonals)
        
        # Update tracking vector
        current_self_kinships = corrected_diagonals

        # Final pruning
        prune_matrix(K_next_raw, threshold)

        # Update current matrix and force garbage collection
        current_matrix = K_next_raw

    return current_matrix

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