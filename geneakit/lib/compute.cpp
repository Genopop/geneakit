/*------------------------------------------------------------------------------
MIT License

Copyright (c) 2024 Gilles-Philippe Morin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
------------------------------------------------------------------------------*/

#include "../include/compute.hpp"

// -----------------------------------------------------------------------------
// DATA STRUCTURE: SparseRow
// -----------------------------------------------------------------------------
// The logic relies on a custom Compressed Sparse Row (CSR) format.
// A standard pedigree kinship matrix is symmetric and dense near the diagonal 
// but becomes very sparse for distant relatives.
//
// Instead of storing a vector of size N with mostly zeros: [0, 0, 0.25, 0, 0.5]
// We store:
//   cols: [2, 4]      (Indices of non-zero kinships)
//   vals: [0.25, 0.5] (The kinship coefficients)
// -----------------------------------------------------------------------------
struct SparseRow {
    std::vector<int> cols;
    std::vector<float> vals;

    void clear() {
        std::vector<int>().swap(cols);
        std::vector<float>().swap(vals);
    }
    
    void reserve(size_t n) {
        cols.reserve(n);
        vals.reserve(n);
    }
};

// Returns the previous generation of a set of individuals.
std::unordered_set<Individual<Indices>*> get_previous_generation(
    std::unordered_set<Individual<Indices>*> &individuals) {
    std::unordered_set<Individual<Indices>*> previous_generation;
    for (Individual<Indices>* individual : individuals) {
        if (individual->father) {
            previous_generation.insert(individual->father);
        }
        if (individual->mother) {
            previous_generation.insert(individual->mother);
        }
    }
    return previous_generation;
}

// Go from the bottom to the top of the pedigree
std::vector<std::unordered_set<Individual<Indices>*>> get_generations(
    std::vector<Individual<Indices>*> &probands) {
    std::vector<std::unordered_set<Individual<Indices>*>> generations;
    std::unordered_set<Individual<Indices>*> generation(probands.begin(), probands.end());
    while (!generation.empty()) {
        generations.push_back(generation);
        generation = get_previous_generation(generation);
    }
    return generations;
}

// Drag the individuals up
std::vector<std::unordered_set<Individual<Indices>*>> copy_bottom_up(
    std::vector<std::unordered_set<Individual<Indices>*>> &generations) {
    std::vector<std::unordered_set<Individual<Indices>*>> bottom_up;
    std::unordered_set<Individual<Indices>*> ids(generations[0].begin(), generations[0].end());
    bottom_up.push_back(ids);
    for (int i = 0; i < (int) generations.size() - 1; i++) {
        std::unordered_set<Individual<Indices>*> next_generation;
        std::set<Individual<Indices>*> set1(bottom_up[i].begin(), bottom_up[i].end());
        std::set<Individual<Indices>*> set2(generations[i + 1].begin(), generations[i + 1].end());
        set_union(
            set1.begin(), set1.end(),
            set2.begin(), set2.end(),
            std::inserter(next_generation, next_generation.end())
        );
        bottom_up.push_back(next_generation);
    }
    reverse(bottom_up.begin(), bottom_up.end());
    return bottom_up;
}

// Drag the individuals down
std::vector<std::unordered_set<Individual<Indices>*>> copy_top_down(
    std::vector<std::unordered_set<Individual<Indices>*>> &generations) {
    reverse(generations.begin(), generations.end());
    std::vector<std::unordered_set<Individual<Indices>*>> top_down;
    std::unordered_set<Individual<Indices>*> ids(generations[0].begin(), generations[0].end());
    top_down.push_back(ids);
    for (int i = 0; i < (int) generations.size() - 1; i++) {
        std::unordered_set<Individual<Indices>*> next_generation;
        std::set<Individual<Indices>*> set1(top_down[i].begin(), top_down[i].end());
        std::set<Individual<Indices>*> set2(generations[i + 1].begin(), generations[i + 1].end());
        set_union(
            set1.begin(), set1.end(),
            set2.begin(), set2.end(),
            std::inserter(next_generation, next_generation.end())
        );
        top_down.push_back(next_generation);
    }
    return top_down;
}

// Find the intersection of the two sets
std::vector<std::vector<Individual<Indices>*>> intersect_both_directions(
    std::vector<std::unordered_set<Individual<Indices>*>> &bottom_up,
    std::vector<std::unordered_set<Individual<Indices>*>> &top_down) {
    std::vector<std::vector<Individual<Indices>*>> vertex_cuts;
    for (int i = 0; i < (int) bottom_up.size(); i++) {
        std::set<Individual<Indices>*> set1(bottom_up[i].begin(), bottom_up[i].end());
        std::set<Individual<Indices>*> set2(top_down[i].begin(), top_down[i].end());
        std::vector<Individual<Indices>*> intersection_result;
        set_intersection(
            set1.begin(), set1.end(),
            set2.begin(), set2.end(),
            std::back_inserter(intersection_result)
        );
        std::vector<Individual<Indices>*> vertex_cut(intersection_result.begin(), intersection_result.end());
        vertex_cuts.push_back(vertex_cut);
    }
    return vertex_cuts;
}

// Separate the individuals into generations where individuals who appear
// in two non-contiguous generations are dragged in the generations in-between.
// Based on the recursive-cut algorithm from Kirkpatrick et al.
std::vector<std::vector<Individual<Indices>*>> cut_vertices(
    std::vector<Individual<Indices>*> &probands) {
    
    std::vector<std::unordered_set<Individual<Indices>*>> generations;
    std::vector<std::vector<Individual<Indices>*>> vertex_cuts;
    
    generations = get_generations(probands);
    std::vector<std::unordered_set<Individual<Indices>*>> bottom_up, top_down;
    bottom_up = copy_bottom_up(generations);
    top_down = copy_top_down(generations);
    vertex_cuts = intersect_both_directions(bottom_up, top_down);
    
    // Set the last vertex cut to the probands explicitly
    vertex_cuts[vertex_cuts.size() - 1] = probands;

    return vertex_cuts;
}

// Returns the sparse kinship matrix
// EXPERIMENTAL! Unlike the rest of GeneaKit, this code was generated by Gemini 3 Pro.
SparseResult compute_kinships_sparse(
    Pedigree<> &pedigree, 
    std::vector<int> proband_ids, 
    bool verbose, 
    bool symmetric_coo
) {
    // -------------------------------------------------------------------------
    // INITIALIZATION & PREPARATION
    // -------------------------------------------------------------------------
    // If no specific probands (individuals of interest) are provided, 
    // we default to assuming all leaf nodes or specific flagged individuals 
    // in the pedigree are the targets.
    if (proband_ids.empty()) {
        proband_ids = get_proband_ids(pedigree);
    }

    // -------------------------------------------------------------------------
    // STEP 1: TOPOLOGICAL SORTING & VERTEX CUTS
    // -------------------------------------------------------------------------
    // The core challenge in large pedigree analysis is memory. A dense kinship 
    // matrix for 100k individuals is too large (100k^2 * 4 bytes).
    //
    // However, kinship is determined recursively: phi(i,j) depends only on the 
    // parents of i and j. We don't need the whole matrix at once. We only need 
    // the "active wavefront" of ancestors required to compute the next generation.
    //
    // The "cut_vertices" function performs a graph analysis. It slices the 
    // pedigree into a sequence of "Cuts" (C_0, C_1, ... C_T).
    // - C_0 contains the founders.
    // - C_T contains the probands.
    // - Any cut C_t acts as a "Vertex Separator". To compute kinships for 
    //   C_{t+1}, we only need the kinship information among individuals in C_t.
    //
    // This allows us to maintain a small square matrix (size |C_t| x |C_t|) 
    // rather than the full |N| x |N| matrix.
    if (verbose) std::cout << "Computing vertex cuts..." << std::endl;
    Pedigree<Indices> kinship_pedigree(pedigree);

    std::vector<Individual<Indices>*> probands;
    probands.reserve(proband_ids.size());
    for (const int id : proband_ids) {
        probands.push_back(kinship_pedigree.individuals.at(id));
    }

    std::vector<std::vector<Individual<Indices>*>> cuts = cut_vertices(probands);

    if (cuts.empty()) {
        return {};
    }

    // -------------------------------------------------------------------------
    // STEP 2: FOUNDER INITIALIZATION (THE 1ST CUT)
    // -------------------------------------------------------------------------
    // The first cut consists entirely of founders (individuals with no parents 
    // in the pedigree).
    // The kinship of a founder with themselves is 0.5 (1/2).
    // The kinship between two distinct founders is 0.0 (unrelated).
    // We initialize the sparse rows for this first generation.
    int n_founders = cuts[0].size();
    std::vector<SparseRow> prev_rows(n_founders);
    std::vector<float> self_kinships(n_founders, 0.5);

    for (int i = 0; i < n_founders; ++i) {
        prev_rows[i].cols.push_back(i);
        prev_rows[i].vals.push_back(0.5f);
    }

    const int BATCH_SIZE = 512;

    // -------------------------------------------------------------------------
    // STEP 3: THE WAVEFRONT ITERATION
    // -------------------------------------------------------------------------
    // We iterate through the cuts from t=0 to t=T-1.
    // In each step, we transition the kinship matrix from the "Current Cut" (t)
    // to the "Next Cut" (t+1).
    for (size_t t = 0; t < cuts.size() - 1; ++t) {
        const auto& current_cut = cuts[t];
        const auto& next_cut = cuts[t + 1];
        int n_prev = current_cut.size();
        int n_next = next_cut.size();

        if (verbose) {
            printf("Processing cut %lu/%lu: %d -> %d individuals\n", 
                t + 1, cuts.size() - 1, n_prev, n_next);
        }

        // --- MAPPING PHASE ---
        // We need to map the Graph Objects (Individual pointers) to Matrix Indices.
        // The matrix `prev_rows` is indexed 0..n_prev-1.
        // The matrix `next_rows` will be indexed 0..n_next-1.
        // We store these indices inside the Individual objects to allow O(1) lookups.

        // Pass 1: Reset flags for the Current Cut.
        // We tell these individuals: "You are at row 'i' in the existing matrix."
        #pragma omp parallel for
        for (int i = 0; i < n_prev; ++i) {
            current_cut[i]->data.prev_idx = i;  
            current_cut[i]->data.next_idx = -1; // Reset destination index
        }

        // Pass 2: Set flags for the Next Cut.
        // We tell these individuals: "You will be at row 'i' in the new matrix."
        // Crucially, some individuals exist in BOTH cuts (they are "dragged" forward).
        // If an individual is in both, they will have valid prev_idx AND next_idx.
        // If they are newborn, they have prev_idx == -1.
        #pragma omp parallel for
        for (int i = 0; i < n_next; ++i) {
            next_cut[i]->data.next_idx = i; 
        }
        
        // --- PRE-PROCESSING: SYMMETRY HANDLING ---
        // We only store the Lower Triangle of the kinship matrix to save memory.
        // However, calculating a child's kinship row requires the *entire* kinship 
        // row of the parent.
        // Row(Parent) = [Lower Triangle of Parent] + [Transposed Columns where Parent is col].
        // Here, we build an adjacency list (ut_adj) to quickly find the "Upper Triangle"
        // values needed to reconstruct full rows.
        std::vector<std::vector<std::pair<int, float>>> ut_adj(n_prev);
        for(int r = 0; r < n_prev; ++r) {
            const auto& row = prev_rows[r];
            for(size_t k = 0; k < row.cols.size(); ++k) {
                int c = row.cols[k];
                if (c != r) {
                    // If M[r][c] exists, record that M[c][r] effectively exists too.
                    ut_adj[c].push_back({r, row.vals[k]});
                }
            }
        }

        std::vector<SparseRow> next_rows(n_next);
        std::vector<float> next_self_kinships(n_next);

        int num_batches = (n_next + BATCH_SIZE - 1) / BATCH_SIZE;

        // --- KINSHIP CALCULATION LOOP ---
        // We compute the next matrix in batches for cache locality.
        for (int b = 0; b < num_batches; ++b) {
            int start = b * BATCH_SIZE;
            int end = std::min(start + BATCH_SIZE, n_next);

            #pragma omp parallel 
            {
                // Thread-local accumulators for dense row construction
                std::vector<float> dense_acc(n_next, 0.0f);
                std::vector<int> col_indices; 
                col_indices.reserve(512);
                
                // 'marker' and 'tag' allow us to check if an index is visited in O(1)
                // without clearing the entire array every iteration.
                std::vector<int> marker(n_next, 0);
                int tag = 0;

                #pragma omp for schedule(dynamic)
                for (int i = start; i < end; ++i) {
                    tag++; 
                    col_indices.clear();
                    
                    Individual<Indices>* individual = next_cut[i];
                    
                    // --- 1. IDENTIFY SOURCES ---
                    // To compute row 'i' of the new matrix, we look at where 'i' comes from.
                    // General Formula: phi(i, x) = 0.5 * phi(Father, x) + 0.5 * phi(Mother, x)
                    struct Src { int idx; float s; };
                    Src sources[2];
                    int s_count = 0;

                    if (individual->data.prev_idx != -1) {
                        // CASE A: COPY.
                        // The individual was already in the previous cut.
                        // Their kinships are simply copied over. Scaler = 1.0.
                        sources[s_count++] = {individual->data.prev_idx, 1.0f};
                    } else {
                        // CASE B: NEWBORN.
                        // The individual is new. We aggregate kinships from parents.
                        // Scaler = 0.5 for each parent.
                        // We must ensure parents are actually present in the Previous Cut (prev_idx >= 0).
                        if (individual->father) {
                            int f_idx = individual->father->data.prev_idx;
                            if (f_idx >= 0 && f_idx < n_prev && current_cut[f_idx] == individual->father) {
                                sources[s_count++] = {f_idx, 0.5f};
                            }
                        }
                        if (individual->mother) {
                            int m_idx = individual->mother->data.prev_idx;
                            if (m_idx >= 0 && m_idx < n_prev && current_cut[m_idx] == individual->mother) {
                                sources[s_count++] = {m_idx, 0.5f};
                            }
                        }
                    }

                    // --- 2. SCATTER / PROJECT CONTRIBUTIONS ---
                    // Instead of iterating all 'j' to find phi(i,j), we iterate the 
                    // *known relationships* of the parents and project them forward.
                    // This is a sparse matrix multiplication: C = A * B, but specialized.
                    for (int k = 0; k < s_count; ++k) {
                        int parent_idx = sources[k].idx;
                        float base_scaler = sources[k].s;
                        
                        // This lambda processes a single kinship value: phi(Parent, Ancestor) = kin_val.
                        // We must figure out where 'Ancestor' ends up in the Next Cut to add the contribution.
                        auto process_ancestor = [&](int ancestor_idx, float kin_val) {
                            Individual<Indices>* ancestor = current_cut[ancestor_idx];
                            
                            // 2a. PROJECT TO SELF (Copy)
                            // If the Ancestor itself survives into the Next Cut at index 'self_child_idx'.
                            int self_child_idx = ancestor->data.next_idx;
                            if (self_child_idx != -1 && 
                                self_child_idx < n_next && 
                                next_cut[self_child_idx] == ancestor) {
                                
                                // Only fill Lower Triangle (col <= row)
                                if (self_child_idx <= i) { 
                                    float contribution = kin_val; 
                                    if (marker[self_child_idx] != tag) {
                                        marker[self_child_idx] = tag;
                                        dense_acc[self_child_idx] = contribution;
                                        col_indices.push_back(self_child_idx);
                                    } else {
                                        dense_acc[self_child_idx] += contribution;
                                    }
                                }
                            }

                            // 2b. PROJECT TO CHILDREN (Inheritance)
                            // If the Ancestor has children in the Next Cut, they inherit relationships.
                            // If phi(Parent, Ancestor) exists, then phi(Child, Ancestor's_Child) 
                            // increases by 0.5 * 0.5 * ...
                            // Specifically: phi(i, Ancestor's_Child) += 0.5 * phi(Parent, Ancestor).
                            for (const auto& child : ancestor->children) {
                                int child_idx = child->data.next_idx;
                                
                                // Validation: Child must be in Next Cut and not be the Ancestor (copy) itself.
                                if (child_idx == -1 || child_idx >= n_next || next_cut[child_idx] != child) continue;
                                
                                // Optimization: If the child was simply copied from the previous cut,
                                // we already handled it in "Project to Self" step above via the Ancestor.
                                int child_prev = child->data.prev_idx;
                                if (child_prev != -1 && child_prev < n_prev && current_cut[child_prev] == child) continue;

                                // Maintain Lower Triangle
                                if (child_idx > i) continue; 

                                // Calculate Genetic Contribution
                                float child_scaler = 0.0f;
                                if (child->father && child->father->data.prev_idx == ancestor_idx) child_scaler += 0.5f;
                                if (child->mother && child->mother->data.prev_idx == ancestor_idx) child_scaler += 0.5f;

                                if (child_scaler > 0.0f) {
                                    float contribution = kin_val * child_scaler;
                                    if (marker[child_idx] != tag) {
                                        marker[child_idx] = tag;
                                        dense_acc[child_idx] = contribution;
                                        col_indices.push_back(child_idx);
                                    } else {
                                        dense_acc[child_idx] += contribution;
                                    }
                                }
                            }
                        };

                        // Apply processing to the Lower Triangle of the Parent's row
                        const SparseRow& p_data = prev_rows[parent_idx];
                        for (size_t idx = 0; idx < p_data.cols.size(); ++idx) {
                            process_ancestor(p_data.cols[idx], p_data.vals[idx] * base_scaler);
                        }

                        // Apply processing to the Upper Triangle of the Parent's row (via UT adjacency)
                        const auto& ut_data = ut_adj[parent_idx];
                        for (const auto& pair : ut_data) {
                            process_ancestor(pair.first, pair.second * base_scaler);
                        }
                    }

                    // --- 3. COMPUTE DIAGONAL (SELF-KINSHIP) ---
                    // The diagonal element is special.
                    // If Copied: It is just carried over (accumulated in loop above).
                    // If New: phi(i,i) = 1/2 + 1/2 * phi(Father, Mother).
                    float diag_val = 0.0f;
                    
                    if (individual->data.prev_idx != -1) {
                        // Existing individual: value already summed in dense_acc[i].
                        if (marker[i] == tag) diag_val = dense_acc[i];
                        else diag_val = 0.5f; // Fallback (shouldn't happen for valid pedigree)
                    } else {
                        // New Birth: 
                        // The accumulator currently contains: 
                        //   0.25 * phi(F, F) [from Father source projecting to Father]
                        // + 0.50 * phi(F, M) [from Father projecting to Mother + Mother to Father]
                        // + 0.25 * phi(M, M) [from Mother projecting to Mother]
                        //
                        // We need: 0.5 + 0.5 * phi(F, M).
                        //
                        // Strategy: Take the raw accumulator, subtract the self-kinship terms of 
                        // the parents (which we know: self_kinships array), and adjust the offset.
                        float term_pp = 0.0f;
                        float term_mm = 0.0f;

                        if (individual->father) {
                            int idx = individual->father->data.prev_idx;
                            if (idx >= 0 && idx < n_prev && current_cut[idx] == individual->father)
                                term_pp = 0.25f * self_kinships[idx];
                        }
                        if (individual->mother) {
                            int idx = individual->mother->data.prev_idx;
                            if (idx >= 0 && idx < n_prev && current_cut[idx] == individual->mother)
                                term_mm = 0.25f * self_kinships[idx];
                        }
                        
                        float raw_val = (marker[i] == tag) ? dense_acc[i] : 0.0f;
                        // Math: 0.5 + (raw_val - 0.25*phi_FF - 0.25*phi_MM)
                        // Note: raw_val represents 0.25*phi_FF + 0.5*phi_FM + 0.25*phi_MM.
                        // So (raw_val - term_pp - term_mm) leaves 0.5*phi_FM.
                        diag_val = 0.5f + raw_val - (term_pp + term_mm);
                    }
                    
                    // Store diagonal back into accumulator
                    if (marker[i] != tag) {
                        marker[i] = tag;
                        dense_acc[i] = diag_val;
                        col_indices.push_back(i);
                    } else {
                        dense_acc[i] = diag_val;
                    }
                    next_self_kinships[i] = diag_val;

                    // --- 4. GATHER ---
                    // Compress the dense accumulator back into SparseRow format.
                    std::sort(col_indices.begin(), col_indices.end());
                    auto last = std::unique(col_indices.begin(), col_indices.end());
                    col_indices.erase(last, col_indices.end());

                    next_rows[i].cols.reserve(col_indices.size());
                    next_rows[i].vals.reserve(col_indices.size());
                    
                    for (int col : col_indices) {
                        next_rows[i].cols.push_back(col);
                        next_rows[i].vals.push_back(dense_acc[col]);
                    }
                }
            }
        }
        
        // Advance the state: Next becomes Previous for the upcoming iteration.
        prev_rows = std::move(next_rows);
        self_kinships = std::move(next_self_kinships);
    }

    // -------------------------------------------------------------------------
    // STEP 4: FINAL OUTPUT FORMATTING
    // -------------------------------------------------------------------------
    // The loop finishes when we reach the final cut (the probands).
    // We now convert the internal vector-of-vectors sparse format into the 
    // standard Coordinate (COO) or Compressed Sparse Row (CSR) format requested.
    SparseResult result;
    int n_final = prev_rows.size();

    if (symmetric_coo) {
        // Output full symmetric matrix (both i,j and j,i)
        size_t total_nnz = 0;
        for (int i = 0; i < n_final; ++i) {
            const auto& row = prev_rows[i];
            for (int col : row.cols) total_nnz += (col == i) ? 1 : 2;
        }

        result.data.reserve(total_nnz);
        result.rows.reserve(total_nnz);
        result.indices.reserve(total_nnz);

        for (int i = 0; i < n_final; ++i) {
            const SparseRow& row = prev_rows[i];
            for (size_t k = 0; k < row.cols.size(); ++k) {
                int j = row.cols[k];
                float val = row.vals[k];
                result.rows.push_back(i);
                result.indices.push_back(j);
                result.data.push_back(val);
                if (i != j) {
                    result.rows.push_back(j);
                    result.indices.push_back(i);
                    result.data.push_back(val);
                }
            }
            // Free memory aggressively
            SparseRow().cols.swap(prev_rows[i].cols);
            SparseRow().vals.swap(prev_rows[i].vals);
        }
    } else {
        // Output standard CSR (Lower Triangle only based on internal logic, 
        // though `prev_rows` strictly stores lower triangle).
        size_t total_nnz = 0;
        for (int i = 0; i < n_final; ++i) total_nnz += prev_rows[i].cols.size();

        result.data.reserve(total_nnz);
        result.indices.reserve(total_nnz);
        result.indptr.reserve(n_final + 1);
        result.indptr.push_back(0);

        for (int i = 0; i < n_final; ++i) {
            const SparseRow& row = prev_rows[i];
            result.indices.insert(result.indices.end(), row.cols.begin(), row.cols.end());
            result.data.insert(result.data.end(), row.vals.begin(), row.vals.end());
            result.indptr.push_back(result.data.size());
            
            SparseRow().cols.swap(prev_rows[i].cols);
            SparseRow().vals.swap(prev_rows[i].vals);
        }
    }

    return result;
}

// Returns the kinship coefficient between two individuals.
// A modified version of the recursive algorithm from Karigl.
double compute_kinship(
    const Individual<Indices> *individual1,
    const Individual<Indices> *individual2,
    Matrix<double> &founder_matrix) {
    double kinship = 0.0;
    const int founder_index1 = individual1->data.prev_idx;
    const int founder_index2 = individual2->data.prev_idx;
    if (founder_index1 != -1 && founder_index2 != -1) {
        // The kinship coefficient is stored in the founder matrix.
        kinship = founder_matrix[founder_index1][founder_index2];
    } else if (founder_index1 != -1) {
        // The kinship coefficient was computed between individual 1
        // and the parents of individual 2.
        if (individual2->father) {
            kinship += 0.5 * compute_kinship(
                individual1, individual2->father, founder_matrix);
        }
        if (individual2->mother) {
            kinship += 0.5 * compute_kinship(
                individual1, individual2->mother, founder_matrix);
        }
    } else if (founder_index2 != -1) {
        // Vice versa.
        if (individual1->father) {
            kinship += 0.5 * compute_kinship(
                individual1->father, individual2, founder_matrix);
        }
        if (individual1->mother) {
            kinship += 0.5 * compute_kinship(
                individual1->mother, individual2, founder_matrix);
        }
    } else if (individual1->rank == individual2->rank) {
        // It's the same individual.
        kinship = 0.5;
        if (individual1->father && individual2->mother) {
            kinship += 0.5 * compute_kinship(
                individual1->father, individual2->mother, founder_matrix);
        }
    } else if (individual1->rank < individual2->rank) {
        // Karigl's recursive algorithm.
        if (individual2->father) {
            kinship += 0.5 * compute_kinship(
                individual1, individual2->father, founder_matrix);
        }
        if (individual2->mother) {
            kinship += 0.5 * compute_kinship(
                individual1, individual2->mother, founder_matrix);
        }
    } else {
        if (individual1->father) {
            kinship += 0.5 * compute_kinship(
                individual1->father, individual2, founder_matrix);
        }
        if (individual1->mother) {
            kinship += 0.5 * compute_kinship(
                individual1->mother, individual2, founder_matrix);
        }
    }
    return kinship;
}

// Compute the kinship coefficients with oneself
void compute_kinship_with_oneself(
    std::vector<Individual<Indices> *> &vertex_cut,
    Matrix<double> &founder_matrix,
    Matrix<double> &proband_matrix) {
    #pragma omp parallel for
    for (int i = 0; i < (int) vertex_cut.size(); i++) {
        const Individual<Indices> *individual = vertex_cut[i];
        double kinship = 0.5;
        if (individual->father && individual->mother) {
            kinship += 0.5 * compute_kinship(
                individual->father, individual->mother, founder_matrix
            );
        }
        proband_matrix[i][i] = kinship;
    }
}

// Compute the kinship coefficients between the individuals
void compute_kinship_between_probands(
    std::vector<Individual<Indices> *> &vertex_cut,
    Matrix<double> &founder_matrix,
    Matrix<double> &proband_matrix) {
    #pragma omp parallel for
    for (int i = 0; i < (int) vertex_cut.size(); i++) {
        Individual<Indices> *individual1 = vertex_cut[i];
        #pragma omp parallel for
        for (int j = 0; j < i; j++) {
            Individual<Indices> *individual2 = vertex_cut[j];
            double kinship = compute_kinship(
                individual1, individual2, founder_matrix
            );
            proband_matrix[i][j] = kinship;
            proband_matrix[j][i] = kinship;
        }
    }
}

// Returns the required memory for kinship calculations.
double get_required_memory_for_kinships(
    Pedigree<> &pedigree, std::vector<int> proband_ids) {
    if (proband_ids.empty()) {
        proband_ids = get_proband_ids(pedigree);
    }
    
    Pedigree<Indices> kinship_pedigree(pedigree);

    // Cut the vertices (a vertex corresponds to an individual)
    std::vector<Individual<Indices>*> probands;
    probands.reserve(proband_ids.size());
    for (const int id : proband_ids) {
        probands.push_back(kinship_pedigree.individuals.at(id));
    }
    
    // cut_vertices now expects a vector of pointers, which we now have
    std::vector<std::vector<Individual<Indices>*>> vertex_cuts = cut_vertices(probands);
    
    if (vertex_cuts.size() < 2) {
        return 0;  // Not enough cuts to compute kinships
    }

    // Calculate the size of each pair of vertex cuts
    std::vector<int> sizes;
    for (int i = 0; i < (int) vertex_cuts.size() - 1; i++) {
        sizes.push_back(vertex_cuts[i].size() + vertex_cuts[i + 1].size());
    }

    // Get the maximum size
    int max_size = *std::max_element(sizes.begin(), sizes.end());

    // Get the two sizes that sum to max_size
    int size1 = 0, size2 = 0;
    for (int i = 0; i < (int) sizes.size(); i++) {
        if ((int) vertex_cuts[i].size() + (int) vertex_cuts[i + 1].size() == max_size) {
            size1 = vertex_cuts[i].size();
            size2 = vertex_cuts[i + 1].size();
            break;
        }
    }

    // Prevent integer overflow by using double for multiplication
    double requirement = (1.0 * size1 * size1 + 1.0 * size2 * size2) * sizeof(double) / 1e9;
    return requirement;
}

// Returns the kinship matrix using the algorithm from Morin et al.
Matrix<double> compute_kinships(Pedigree<> &pedigree,
    std::vector<int> proband_ids, bool verbose) {
    // Get the proband IDs if they are not provided
    if (proband_ids.empty()) {
        proband_ids = get_proband_ids(pedigree);
    }
    // Convert the pedigree to a kinship pedigree
    Pedigree<Indices> kinship_pedigree(pedigree);
    std::vector<Individual<Indices>*> probands;
    probands.reserve(proband_ids.size());
    for (const int id : proband_ids) {
        probands.push_back(kinship_pedigree.individuals.at(id));
    }
    // Cut the vertices (a vertex corresponds to an individual)
    std::vector<std::vector<Individual<Indices>*>> vertex_cuts;
    vertex_cuts = cut_vertices(probands);
    // Initialize the founders' kinship matrix
    Matrix<double> founder_matrix = zeros<double>(
        vertex_cuts[0].size(), vertex_cuts[0].size()
    );
    #pragma omp parallel for
    for (int i = 0; i < (int) vertex_cuts[0].size(); i++) {
        founder_matrix[i][i] = 0.5;
    }
    if (verbose) {
        // Print the vertex cuts
        int count = 0;
        for (std::vector<Individual<Indices>*> vertex_cut : vertex_cuts) {
            printf("Cut size %d/%d: %d\n",
                ++count, (int) vertex_cuts.size(), (int) vertex_cut.size());
        }
        printf("Computing the kinship matrix:\n");
    }
    int cut_count = 0;
    // Go from the top to the bottom of the pedigree
    for (int i = 0; i < (int) vertex_cuts.size() - 1; i++) {
        if (verbose) {
            printf("Cut %d out of %d\n", ++cut_count, (int) vertex_cuts.size());
        }
        // Index the founders
        int founder_index = 0;
        for (const auto& individual : vertex_cuts[i]) {
            individual->data.prev_idx = founder_index++;
        }
        // Initialize the probands' kinship matrix
        Matrix<double> proband_matrix(
            vertex_cuts[i + 1].size(), vertex_cuts[i + 1].size()
        );
        std::vector<Individual<Indices> *> next_generation;
        for (const auto& individual : vertex_cuts[i + 1]) {
            next_generation.push_back(individual);
        }
        compute_kinship_with_oneself(
            next_generation, founder_matrix, proband_matrix
        );
        compute_kinship_between_probands(
            next_generation, founder_matrix, proband_matrix
        );
        // The current generation becomes the previous generation
        founder_matrix = std::move(proband_matrix);
    }
    return founder_matrix;
}

// Sort animals according to ID of their sires into SId.
void MyQuickSort(int **Ped, int *SId, int size) {
    std::function<void(int, int)> quicksort = [&](int left, int right) {
        if (left >= right) return;
        int pivot = SId[(left + right) / 2];
        int i = left, j = right;
        while (i <= j) {
            while (Ped[SId[i]][0] < Ped[pivot][0]) i++;
            while (Ped[SId[j]][0] > Ped[pivot][0]) j--;
            if (i <= j) {
                std::swap(SId[i], SId[j]);
                i++;
                j--;
            }
        }
        quicksort(left, j);
        quicksort(i, right);
    };
    quicksort(1, size - 1); // Correctly pass the size of the array
}

// Returns the inbreeding coefficients of a vector of individuals.
// Copied from the article by M Sargolzaei, H Iwaisaki & J-J Colleau (2005).
std::vector<double> compute_inbreedings(Pedigree<> &pedigree,
    std::vector<int> proband_ids) {
    std::vector<double> inbreeding_coefficients(proband_ids.size());
    Pedigree<> extracted_pedigree = extract_pedigree(pedigree, proband_ids, {});
    // n = number of animals in total; m = number of sires and dams in total
    int n = extracted_pedigree.ids.size(), m = 0;
    for (const int id : extracted_pedigree.ids) {
        Individual<> *individual = extracted_pedigree.individuals.at(id);
        if (!individual->children.empty()) {
            m++;
        }
    }
    int i, j, k, rN, rS, S, D, MIP; // integer variables.
    int **Ped = new int*[n + 1], **rPed = new int*[m + 1]; // main and reduced pedigrees, respectively
    for (int i = 0; i <= n; i++) {
        Ped[i] = new int[2];
    }
    for (int i = 0; i <= m; i++) {
        rPed[i] = new int[2];
    }
    int *SId = new int[n + 1]; // will contain the sorted animals ID based on the ID of their sires
    int *Link = new int[n + 1]; // will contain new ID of ancestors at position of their original ID
    int *MaxIdP = new int[m + 1]; // will contain maximum new ID of parents for each paternal
    // group at position of the new ID of each sire
    double *F = new double[n + 1], *B = new double[m + 1], *x = new double[m + 1]; // inbreeding coefficients, within family
    // segregation variances and x arrays, respectively

    for (i = 1; i <= n; ++i) {
        const int id = extracted_pedigree.ids.at(i - 1);
        Individual<> *father = extracted_pedigree.individuals.at(id)->father;
        Individual<> *mother = extracted_pedigree.individuals.at(id)->mother;
        Ped[i][0] = father ? father->rank + 1 : 0;
        Ped[i][1] = mother ? mother->rank + 1 : 0;
    }

    F[0] = -1.0, x[0] = 0.0, Link[0] = 0; // set values for the unknown parent
    for (rN = i = 1; i <= n; ++i) { // extract and recode ancestors
        SId[i] = i, Link[i] = 0; if (i <= m) x[i] = 0.0; // initialization
        S = Ped[i][0], D = Ped[i][1];
        if (S&& !Link[S]) {
            MaxIdP[rN] = Link[S] = rN;
            rPed[rN][0] = Link[Ped[S][0]];
            rPed[rN++][1] = Link[Ped[S][1]];
        }
        if (D&& !Link[D]) {
            Link[D] = rN;
            rPed[rN][0] = Link[Ped[D][0]];
            rPed[rN++][1] = Link[Ped[D][1]];
        }
        if (MaxIdP[Link[S]] < Link[D]) MaxIdP[Link[S]] = Link[D]; // determine
        // maximum ID of parents for each paternal group
    }
    MyQuickSort(Ped, SId, n); // sort animals according to ID of their sires into SId
    for (k = i = 1; i <= n;) { // do from the oldest sire to the youngest sire
        if (!Ped[SId[i]][0]) F[SId[i++]] = 0.0; // sire is not known
        else {
            S = Ped[SId[i]][0], rS = Link[S], MIP = MaxIdP[rS];
            x[rS] = 1.0;
            for (; k <= S; ++k) // compute within family segregation variances
                if (Link[k]) B[Link[k]] = 0.5 - 0.25 * (F[Ped[k][0]] + F[Ped[k][1]]);
            for (j = rS; j; --j) { // trace back the reduced pedigree
                if (x[j]) { // consider only ancestors of the sire
                    if (rPed[j][0]) x[rPed[j][0]] += x[j] * 0.5;
                    if (rPed[j][1]) x[rPed[j][1]] += x[j] * 0.5;
                    x[j] *= B[j];
                }
            }
            for (j = 1; j <= MIP; ++j) // trace forth the reduced pedigree
                x[j] += (x[rPed[j][0]] + x[rPed[j][1]]) * 0.5;
            for (; i <= n; ++i) // obtain F for progeny of the current sire
                if (S != Ped[SId[i]][0]) break;
                else F[SId[i]] = x[Link[Ped[SId[i]][1]]] * 0.5;
            for (j = 1; j <= MIP; ++j) x[j] = 0.0; // set to 0 for next evaluation of sire
        }
    }
    for (int i = 0; i < (int) proband_ids.size(); i++) {
        const int id = proband_ids[i];
        Individual<> *individual = extracted_pedigree.individuals.at(id);
        inbreeding_coefficients[i] = F[individual->rank + 1];
    }
    delete[] Ped;
    delete[] rPed;
    delete[] SId;
    delete[] Link;
    delete[] MaxIdP;
    delete[] F;
    delete[] B;
    delete[] x;
    return inbreeding_coefficients;
}

// Adds the contribution of an individual.
void add_contribution(const Individual<Contribution> *individual,
    const int depth) {
    if (individual->data.is_proband) {
        individual->data.contribution += pow(0.5, depth);
    } else {
        for (const Individual<Contribution> *child :
            individual->children) {
            add_contribution(child, depth + 1);
        }
    }
}

// Returns the genetic contributions of a pedigree.
// Each row corresponds to a proband.
// Each column corresponds to an ancestor.
Matrix<double> compute_genetic_contributions(Pedigree<> &pedigree,
    std::vector<int> proband_ids, std::vector<int> ancestor_ids) {
    if (proband_ids.empty()) {
        proband_ids = get_proband_ids(pedigree);
    }
    if (ancestor_ids.empty()) {
        ancestor_ids = get_founder_ids(pedigree);
    }
    // Convert the pedigree to a kinship pedigree
    Pedigree<Contribution> contribution_pedigree(pedigree);
    // Initialize the contributions matrix
    Matrix<double> contributions(proband_ids.size(), ancestor_ids.size());
    // Mark the probands
    for (int i = 0; i < (int) proband_ids.size(); i++) {
        contribution_pedigree.individuals.at(proband_ids[i])->
            data.is_proband = true;
    }
    // Compute the contributions
    for (int j = 0; j < (int) ancestor_ids.size(); j++) {
        Individual<Contribution> *ancestor = contribution_pedigree.
            individuals.at(ancestor_ids[j]);
        add_contribution(ancestor, 0);
        for (int i = 0; i < (int) proband_ids.size(); i++) {
            Individual<Contribution> *proband = contribution_pedigree.
                individuals.at(proband_ids[i]);
            contributions[i][j] = proband->data.contribution;
            proband->data.contribution = 0.0;
        }
    }
    return contributions;
}
