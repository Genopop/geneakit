import gc as gc_
import time
import cgeneakit
import numpy as np
import pandas as pd
from numba import njit, prange, get_num_threads, get_thread_id
from scipy.sparse import csr_matrix, issparse
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
    return parent_set

def get_generations(pedigree, proband_ids):
    """Go from the bottom to the top of the pedigree."""
    generations = []
    generation = set(proband_ids)
    while generation:
        generations.append(generation)
        generation = get_previous_generation(pedigree, generation)
    return generations

def copy_bottom_up(generations):
    """Drag the individuals up (bottom-up closure)."""
    bottom_up = []
    ids = generations[0]
    bottom_up.append(ids)
    for i in range(len(generations) - 1):
        next_generation = set(bottom_up[i]) | set(generations[i + 1])
        bottom_up.append(next_generation)
    bottom_up.reverse()
    return bottom_up

def copy_top_down(generations):
    """Drag the individuals down (top-down closure)."""
    gens_reversed = list(reversed(generations))
    top_down = []
    ids = gens_reversed[0]
    top_down.append(ids)
    for i in range(len(gens_reversed) - 1):
        next_generation = set(top_down[i]) | set(gens_reversed[i + 1])
        top_down.append(next_generation)
    return top_down

def intersect_both_directions(bottom_up, top_down):
    """Find the intersection of the two sets (the vertex cuts)."""
    vertex_cuts = []
    for i in range(len(bottom_up)):
        vertex_cut = list(bottom_up[i].intersection(top_down[i]))
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
# Numba Optimized Kernels (int64 support)
# ---------------------------------------------------------

@njit(cache=True)
def build_maps(n_next, next_cut, prev_map, father_indices, mother_indices):
    """
    Constructs mapping structures.
    Uses int64 to support large pedigrees.
    """
    # UP MAP: n_next x 2
    up_map = np.full((n_next, 2), -1, dtype=np.int64)
    
    n_prev = len(prev_map)
    down_counts = np.zeros(n_prev, dtype=np.int64)
    
    for i in range(n_next):
        ind = next_cut[i]
        
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
    down_indptr = np.zeros(n_prev + 1, dtype=np.int64)
    for k in range(n_prev):
        down_indptr[k+1] = down_indptr[k] + down_counts[k]
        
    down_indices = np.empty(down_indptr[-1], dtype=np.int64)
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
    batch_start,
    batch_up_map,   # Sliced map for iterating rows (local index -> parents)
    full_up_map,    # Full map for looking up child info (global index -> parents)
    down_indices, down_starts, down_ends,
    K_prev_indices, K_prev_starts, K_prev_ends, K_prev_data,
    current_self_kinships,
    scratch_maps, scratch_vals, scratch_cols
):
    """
    Computes K_next = P * K_prev * P.T + D.
    Supports batching via batch_start and separated up_maps.
    """
        
    out_indptr = np.empty(n_next + 1, dtype=np.int64)
    out_indptr[0] = 0
        
    row_nnzs = np.zeros(n_next, dtype=np.int64)
    
    # --- PASS 1: COUNT NNZ ---
    for i in prange(n_next):
        tid = get_thread_id()
        spa_map = scratch_maps[tid]
        
        # Row Tag uses Global Index to avoid collision across batches
        row_tag = i + batch_start
        
        nnz_count = 0
        
        p1 = batch_up_map[i, 0]
        p2 = batch_up_map[i, 1]
        is_copy = (p2 == -2)
        
        parents_to_scan = np.empty(2, dtype=np.int64)
        pt_count = 0
        if p1 >= 0: 
            parents_to_scan[pt_count] = p1
            pt_count += 1
        if not is_copy and p2 >= 0:
            parents_to_scan[pt_count] = p2
            pt_count += 1
            
        for p_idx_i in range(pt_count):
            p_row = parents_to_scan[p_idx_i]
            start = K_prev_starts[p_row]
            end = K_prev_ends[p_row]
            
            for k_idx in range(start, end):
                ancestor = K_prev_indices[k_idx]
                d_start = down_starts[ancestor]
                d_end = down_ends[ancestor]
                
                for d_k in range(d_start, d_end):
                    child = down_indices[d_k]
                    if spa_map[child] != row_tag:
                        spa_map[child] = row_tag
                        nnz_count += 1
        
        # Diagonal (Global Index)
        diag_col = i + batch_start
        if spa_map[diag_col] != row_tag:
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
        local_indices = scratch_cols[tid] 
        
        row_tag = i + batch_start
        local_ptr = 0
        
        p1 = batch_up_map[i, 0]
        p2 = batch_up_map[i, 1]
        is_copy = (p2 == -2)
        
        parents_to_scan = np.empty(2, dtype=np.int64)
        pt_count = 0
        if p1 >= 0: 
            parents_to_scan[pt_count] = p1
            pt_count += 1
        if not is_copy and p2 >= 0:
            parents_to_scan[pt_count] = p2
            pt_count += 1
            
        for p_idx_i in range(pt_count):
            p_row = parents_to_scan[p_idx_i]
            
            scaler = 1.0 if is_copy else 0.5
            
            start = K_prev_starts[p_row]
            end = K_prev_ends[p_row]
            
            for k_idx in range(start, end):
                ancestor = K_prev_indices[k_idx]
                kinship_val = K_prev_data[k_idx]
                
                val_to_scatter = scaler * kinship_val
                
                d_start = down_starts[ancestor]
                d_end = down_ends[ancestor]
                
                for d_k in range(d_start, d_end):
                    child = down_indices[d_k]
                    
                    # Child is global index. 
                    # Use FULL map to determine child's parents for scaling.
                    c_p1 = full_up_map[child, 0]
                    c_p2 = full_up_map[child, 1]
                    
                    child_scaler = 0.0
                    if c_p2 == -2: 
                         if c_p1 == ancestor: child_scaler = 1.0
                    else:
                        if c_p1 == ancestor: child_scaler += 0.5
                        if c_p2 == ancestor: child_scaler += 0.5
                    
                    contribution = val_to_scatter * child_scaler
                    
                    if spa_map[child] != row_tag:
                        spa_map[child] = row_tag
                        spa_values[child] = contribution
                        local_indices[local_ptr] = child
                        local_ptr += 1
                    else:
                        spa_values[child] += contribution
        
        diag_col = i + batch_start
        
        # Diagonal Correction
        diag_val = 0.0
        if is_copy:
            if spa_map[diag_col] == row_tag:
                diag_val = spa_values[diag_col]
            else:
                diag_val = 0.5 
        else:
            phi_pp = 0.0
            phi_mm = 0.0
            if p1 >= 0: phi_pp = current_self_kinships[p1]
            if p2 >= 0: phi_mm = current_self_kinships[p2]
            
            raw_val = 0.0
            if spa_map[diag_col] == row_tag:
                raw_val = spa_values[diag_col]
                
            diag_val = 0.5 + raw_val - 0.25 * (phi_pp + phi_mm)

        output_diagonals[i] = diag_val
        
        if spa_map[diag_col] != row_tag:
            spa_map[diag_col] = row_tag
            local_indices[local_ptr] = diag_col
            local_ptr += 1
        
        spa_values[diag_col] = diag_val
        
        # Write global
        global_offset = out_indptr[i]
        for k in range(local_ptr):
            col = local_indices[k]
            out_indices[global_offset + k] = col
            out_data[global_offset + k] = spa_values[col]
            
    return out_indices, out_indptr, out_data, output_diagonals

@njit(parallel=True, cache=True)
def compute_last_generation_dense(
    n_next,
    up_map,
    down_indices, down_starts, down_ends,
    K_prev_indices, K_prev_starts, K_prev_ends, K_prev_data,
    current_self_kinships,
    scratch_vals
):
    """
    Computes K_next directly into a Dense Matrix (float32).
    """
    K_next = np.zeros((n_next, n_next), dtype=np.float32)
    
    for i in prange(n_next):
        tid = get_thread_id()
        spa_values = scratch_vals[tid]
        
        p1 = up_map[i, 0]
        p2 = up_map[i, 1]
        is_copy = (p2 == -2)
        
        parents_to_scan = np.empty(2, dtype=np.int64)
        pt_count = 0
        if p1 >= 0: 
            parents_to_scan[pt_count] = p1
            pt_count += 1
        if not is_copy and p2 >= 0:
            parents_to_scan[pt_count] = p2
            pt_count += 1
            
        for p_idx_i in range(pt_count):
            p_row = parents_to_scan[p_idx_i]
            scaler = 1.0 if is_copy else 0.5
            
            start = K_prev_starts[p_row]
            end = K_prev_ends[p_row]
            
            for k_idx in range(start, end):
                ancestor = K_prev_indices[k_idx]
                kinship_val = K_prev_data[k_idx]
                
                val_to_scatter = scaler * kinship_val
                
                d_start = down_starts[ancestor]
                d_end = down_ends[ancestor]
                
                for d_k in range(d_start, d_end):
                    child = down_indices[d_k]
                    
                    c_p1 = up_map[child, 0]
                    c_p2 = up_map[child, 1]
                    
                    child_scaler = 0.0
                    if c_p2 == -2: 
                         if c_p1 == ancestor: child_scaler = 1.0
                    else:
                        if c_p1 == ancestor: child_scaler += 0.5
                        if c_p2 == ancestor: child_scaler += 0.5
                    
                    contribution = val_to_scatter * child_scaler
                    spa_values[child] += contribution
                        
        diag_val = 0.0
        if is_copy:
            diag_val = spa_values[i]
        else:
            phi_pp = 0.0
            phi_mm = 0.0
            if p1 >= 0: phi_pp = current_self_kinships[p1]
            if p2 >= 0: phi_mm = current_self_kinships[p2]
            
            raw_val = spa_values[i]
            diag_val = 0.5 + raw_val - 0.25 * (phi_pp + phi_mm)

        spa_values[i] = diag_val
        
        for col in range(n_next):
            K_next[i, col] = spa_values[col]
            spa_values[col] = 0.0
            
    return K_next

# ---------------------------------------------------------
# Low Memory Utils (Optimized)
# ---------------------------------------------------------

@njit(cache=True)
def get_usage_counts(n_next, up_map, n_prev):
    """
    Counts how many times each parent in the previous generation is used.
    """
    counts = np.zeros(n_prev, dtype=np.int32)
    for i in range(n_next):
        p1 = up_map[i, 0]
        p2 = up_map[i, 1]
        if p1 >= 0: counts[p1] += 1
        if p2 >= 0 and p2 != -2: counts[p2] += 1
    return counts

@njit(cache=True)
def get_batch_parents_sorted(batch_start, batch_end, up_map, temp_buffer):
    """
    Identifies unique parents for a batch using a pre-allocated buffer.
    Returns a slice of the buffer containing unique sorted parents.
    """
    ptr = 0
    for i in range(batch_start, batch_end):
        p1 = up_map[i, 0]
        p2 = up_map[i, 1]
        if p1 >= 0:
            temp_buffer[ptr] = p1
            ptr += 1
        if p2 >= 0 and p2 != -2:
            temp_buffer[ptr] = p2
            ptr += 1
            
    if ptr == 0:
        return temp_buffer[:0]
        
    # Sort the used portion of the buffer
    view = temp_buffer[:ptr]
    view.sort()
    
    # Unique pass (in-place logic, writing to output array)
    # We count first to allocate exact size for the return array
    unique_count = 1
    for i in range(1, ptr):
        if view[i] != view[i-1]:
            unique_count += 1
            
    out = np.empty(unique_count, dtype=np.int64)
    out[0] = view[0]
    k = 1
    for i in range(1, ptr):
        if view[i] != view[i-1]:
            out[k] = view[i]
            k += 1
            
    return out

@njit(cache=True)
def bulk_decrement_counts(usage_counts, batch_start, batch_end, up_map):
    """
    Efficiently decrements usage counts for the processed batch.
    """
    for i in range(batch_start, batch_end):
        p1 = up_map[i, 0]
        p2 = up_map[i, 1]
        if p1 >= 0: 
            usage_counts[p1] -= 1
        if p2 >= 0 and p2 != -2: 
            usage_counts[p2] -= 1

def compute_kinships_sparse_low_mem(gen, pro=None, dense_output=False, verbose=False):
    if pro is None:
        pro = cgeneakit.get_proband_ids(gen)

    n_total = len(gen)
    father_indices = np.full(n_total, -1, dtype=np.int64)
    mother_indices = np.full(n_total, -1, dtype=np.int64)
    
    for individual in gen.values():
        if individual.father.ind:
            father_indices[individual.rank] = individual.father.rank
        if individual.mother.ind:
            mother_indices[individual.rank] = individual.mother.rank

    if verbose: print("Computing vertex cuts (Low Memory Optimized)...")
    raw_vertex_cuts = cut_vertices(gen, pro)
    cuts_mapped = []
    for cut in raw_vertex_cuts:
        mapped = np.array([gen[id].rank for id in cut], dtype=np.int64)
        cuts_mapped.append(mapped)

    if not cuts_mapped:
        empty_ret = csr_matrix((0, 0), dtype=np.float32)
        return empty_ret.toarray() if dense_output else empty_ret

    n_founders = len(cuts_mapped[0])
    # Store columns and values separately to avoid tuple unpacking overhead later
    prev_cols = []
    prev_vals = []
    for k in range(n_founders):
        prev_cols.append(np.array([k], dtype=np.int64))
        prev_vals.append(np.array([0.5], dtype=np.float32))
    
    current_self_kinships = np.full(n_founders, 0.5, dtype=np.float32)
    
    n_threads = get_num_threads()
    BATCH_SIZE = 1024
    
    max_cut_size = max(len(c) for c in cuts_mapped)
    
    global_K_starts = np.zeros(max_cut_size + 1, dtype=np.int64)
    global_K_ends = np.zeros(max_cut_size + 1, dtype=np.int64)
    
    parent_id_buffer = np.empty(BATCH_SIZE * 2, dtype=np.int64)
    
    # --- PRE-ALLOCATED REUSABLE BUFFERS ---
    # Start with a reasonable size (e.g., 10 MB elements) to minimize resizing
    # but small enough not to hog RAM if pedigree is small.
    current_buf_cap = 1024 * 1024 
    shared_indices_buf = np.empty(current_buf_cap, dtype=np.int64)
    shared_data_buf = np.empty(current_buf_cap, dtype=np.float32)
    
    for t in range(len(cuts_mapped) - 1):
        current_cut = cuts_mapped[t]
        next_cut = cuts_mapped[t + 1]
        n_next = len(next_cut)
        n_prev = len(current_cut)

        if verbose:
            print(f"Processing cut {t+1}/{len(cuts_mapped)-1}: {n_prev} -> {n_next} (Low Mem)")

        prev_map = np.full(n_total, -1, dtype=np.int64)
        prev_map[current_cut] = np.arange(n_prev, dtype=np.int64)
        
        up_map, down_indices, down_indptr = build_maps(
            n_next, next_cut, prev_map, father_indices, mother_indices
        )
        
        down_starts = down_indptr[:-1]
        down_ends = down_indptr[1:]
        
        usage_counts = get_usage_counts(n_next, up_map, n_prev)
        
        next_cols = [None] * n_next
        next_vals = [None] * n_next
        next_self_kinships = np.zeros(n_next, dtype=np.float32)
        
        num_batches = (n_next + BATCH_SIZE - 1) // BATCH_SIZE
        
        scratch_maps = np.full((n_threads, n_next), -1, dtype=np.int64)
        scratch_vals = np.zeros((n_threads, n_next), dtype=np.float32)
        scratch_cols = np.zeros((n_threads, n_next), dtype=np.int64)
        
        for b in range(num_batches):
            start = b * BATCH_SIZE
            end = min(start + BATCH_SIZE, n_next)
            batch_len = end - start
            
            batch_up_map = up_map[start:end] 
            
            unique_parents = get_batch_parents_sorted(start, end, up_map, parent_id_buffer)
            
            # 1. Calculate required size for this batch (No Data Copy yet)
            total_req_size = 0
            for uid in unique_parents:
                # We use len() on the numpy array directly.
                # prev_cols[uid] is the indices array.
                total_req_size += len(prev_cols[uid])
            
            # 2. Resize buffers if necessary (Exponential growth to amortize cost)
            if total_req_size > current_buf_cap:
                while current_buf_cap < total_req_size:
                    current_buf_cap *= 2
                shared_indices_buf = np.empty(current_buf_cap, dtype=np.int64)
                shared_data_buf = np.empty(current_buf_cap, dtype=np.float32)
            
            # 3. Fill Buffer (Replaces np.concatenate)
            # This loop performs slice assignments which are fast in Numpy
            # and avoids allocating new temporary arrays.
            current_ptr = 0
            for uid in unique_parents:
                p_col = prev_cols[uid]
                p_val = prev_vals[uid]
                length = len(p_col)
                
                # Copy data to shared buffer
                shared_indices_buf[current_ptr : current_ptr + length] = p_col
                shared_data_buf[current_ptr : current_ptr + length] = p_val
                
                # Update map for Numba
                global_K_starts[uid] = current_ptr
                current_ptr += length
                global_K_ends[uid] = current_ptr
            
            # 4. Create views (Zero Copy) to pass to Numba
            batch_prev_indices = shared_indices_buf[:total_req_size]
            batch_prev_data = shared_data_buf[:total_req_size]

            b_indices, b_indptr, b_data, b_diags = compute_generation_fused(
                batch_len,
                start, 
                batch_up_map, 
                up_map,       
                down_indices, down_starts, down_ends,
                batch_prev_indices, global_K_starts, global_K_ends, batch_prev_data,
                current_self_kinships,
                scratch_maps, scratch_vals, scratch_cols
            )
            
            for i in range(batch_len):
                row_start = b_indptr[i]
                row_end = b_indptr[i+1]
                next_cols[start + i] = b_indices[row_start:row_end].copy()
                next_vals[start + i] = b_data[row_start:row_end].copy()
                next_self_kinships[start + i] = b_diags[i]
            
            for uid in unique_parents:
                global_K_starts[uid] = 0
                global_K_ends[uid] = 0
            
            bulk_decrement_counts(usage_counts, start, end, up_map)

            # Free memory
            for uid in unique_parents:
                if usage_counts[uid] <= 0:
                    prev_cols[uid] = None
                    prev_vals[uid] = None
            
        prev_cols = next_cols
        prev_vals = next_vals
        current_self_kinships = next_self_kinships
        
        del next_cols, next_vals, up_map, down_indices, down_indptr
    
    if verbose: print("Finalizing matrix construction...")
    
    if dense_output:
        n = len(prev_cols)
        mat = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            mat[i, prev_cols[i]] = prev_vals[i]
        return mat
    else:
        # --- LOW MEMORY FINALIZATION (Two-Pass) ---
        total_nnz = 0
        for cols in prev_cols:
            total_nnz += len(cols)
            
        final_indptr = np.empty(n_next + 1, dtype=np.int64)
        final_indptr[0] = 0

        # Pass 1: Indices (Int64)
        final_indices = np.empty(total_nnz, dtype=np.int64)
        current_ptr = 0
        for i in range(n_next):
            cols = prev_cols[i]
            length = len(cols)
            final_indices[current_ptr : current_ptr + length] = cols
            final_indptr[i+1] = current_ptr + length
            
            prev_cols[i] = None # Release immediately
            current_ptr += length
            
        gc_.collect() # Force reclaim of Int64 arrays
        
        # Pass 2: Data (Float32)
        final_data = np.empty(total_nnz, dtype=np.float32)
        current_ptr = 0
        for i in range(n_next):
            vals = prev_vals[i]
            length = len(vals)
            final_data[current_ptr : current_ptr + length] = vals
            
            prev_vals[i] = None # Release immediately
            current_ptr += length
            
        del prev_cols, prev_vals
        gc_.collect()
        
        return csr_matrix((final_data, final_indices, final_indptr), 
                          shape=(n_next, n_next))
                
def compute_kinships_sparse(gen, pro=None, dense_output=False, verbose=False):
    if pro is None:
        pro = cgeneakit.get_proband_ids(gen)

    n_total = len(gen)
    father_indices = np.full(n_total, -1, dtype=np.int64)
    mother_indices = np.full(n_total, -1, dtype=np.int64)
    
    for individual in gen.values():
        if individual.father.ind:
            father_indices[individual.rank] = individual.father.rank
        if individual.mother.ind:
            mother_indices[individual.rank] = individual.mother.rank

    if verbose: print("Computing vertex cuts...")
    raw_vertex_cuts = cut_vertices(gen, pro)
    cuts_mapped = []
    for cut in raw_vertex_cuts:
        mapped = np.array([gen[id].rank for id in cut], dtype=np.int64)
        cuts_mapped.append(mapped)

    if not cuts_mapped:
        empty_ret = csr_matrix((0, 0), dtype=np.float32)
        if dense_output:
            return empty_ret.toarray()
        return empty_ret

    n_founders = len(cuts_mapped[0])
    current_data = np.full(n_founders, 0.5, dtype=np.float32)
    current_indices = np.arange(n_founders, dtype=np.int64) 
    current_indptr = np.arange(n_founders + 1, dtype=np.int64)
    current_self_kinships = np.full(n_founders, 0.5, dtype=np.float32)
    
    n_threads = get_num_threads()
    
    if len(cuts_mapped) == 1 and dense_output:
         mat = np.zeros((n_founders, n_founders), dtype=np.float32)
         np.fill_diagonal(mat, 0.5)
         return mat

    for t in range(len(cuts_mapped) - 1):
        current_cut = cuts_mapped[t]
        next_cut = cuts_mapped[t + 1]
        n_next = len(next_cut)
        n_prev = len(current_cut)

        if verbose:
            print(f"Processing cut {t+1}/{len(cuts_mapped)-1}: {n_prev} -> {n_next} individuals")

        prev_map = np.full(n_total, -1, dtype=np.int64)
        prev_map[current_cut] = np.arange(n_prev, dtype=np.int64)

        up_map, down_indices, down_indptr = build_maps(
            n_next, next_cut, prev_map, father_indices, mother_indices
        )
        
        is_last_step = (t == len(cuts_mapped) - 2)
        
        if is_last_step and dense_output:
            if verbose: print("Computing final generation directly to dense matrix...")
            scratch_vals = np.zeros((n_threads, n_next), dtype=np.float32)
            
            dense_result = compute_last_generation_dense(
                n_next,
                up_map,
                down_indices, down_indptr[:-1], down_indptr[1:],
                current_indices, current_indptr[:-1], current_indptr[1:], current_data,
                current_self_kinships,
                scratch_vals
            )
            del up_map, down_indices, down_indptr, scratch_vals
            gc_.collect()
            return dense_result
            
        else:
            scratch_maps = np.full((n_threads, n_next), -1, dtype=np.int64)
            scratch_vals = np.zeros((n_threads, n_next), dtype=np.float32)
            scratch_cols = np.zeros((n_threads, n_next), dtype=np.int64)
            
            next_indices, next_indptr, next_data, next_self = compute_generation_fused(
                n_next,
                0, 
                up_map, # batch_up_map (full)
                up_map, # full_up_map
                down_indices, down_indptr[:-1], down_indptr[1:],
                current_indices, current_indptr[:-1], current_indptr[1:], current_data,
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
# Standard Interfaces
# ---------------------------------------------------------

def phi(gen, **kwargs):
    """Compute kinship coefficients between probands.
    
    Args:
        gen (cgeneakit.Pedigree): Initialized genealogy.
        pro (list, optional): Proband IDs.
        verbose (bool, default False): Print details.
        compute (bool, default True): Estimate memory if False.
        sparse (bool, default False): Use sparse computation algorithm (graph cuts).
        dense_output (bool, default False): If True and sparse=True, computes the final matrix
            directly as a dense array. Faster for dense outputs.
        low_memory (bool, default False): If True and sparse=True, optimizes for memory
            usage by deleting parent rows as soon as children are processed. 
            Slightly slower but significantly reduces peak RAM.
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
    dense_output = kwargs.get('dense_output', False)
    low_memory = kwargs.get('low_memory', False)
    
    if not compute:
        required_memory = cgeneakit.get_required_memory_for_kinships(gen, pro)
        print(f'You will require at least {round(required_memory, 2)} GB of RAM.')
        return
    
    if verbose:
        begin = time.time()
        
    if sparse:
        if low_memory:
            kinship_matrix = compute_kinships_sparse_low_mem(gen, pro=pro, dense_output=dense_output, verbose=verbose)
        else:
            kinship_matrix = compute_kinships_sparse(gen, pro=pro, dense_output=dense_output, verbose=verbose)
        
        if raw:
            if verbose:
                print(f'Elapsed time: {round(time.time() - begin, 2)} seconds')
            # Return raw matrix (dense ndarray or sparse csr) + ids
            return kinship_matrix, pro
            
        if dense_output:
            # Result is numpy array
            kinship_matrix = pd.DataFrame(kinship_matrix, index=pro, columns=pro)
        else:
            # Result is CSR matrix -> Sparse DataFrame
            kinship_matrix = pd.DataFrame.sparse.from_spmatrix(
                kinship_matrix, index=pro, columns=pro
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