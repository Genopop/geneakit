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
// -----------------------------------------------------------------------------
// COLLEAU'S INDIRECT METHOD IMPLEMENTATION
// -----------------------------------------------------------------------------

// Helper struct to manage the remapped, topologically sorted pedigree subset
struct RemappedPedigree {
    int num_individuals;
    std::vector<int> original_ids;      // Map: New Index -> Original ID
    std::vector<int> sires;             // Map: New Index -> New Sire Index (-1 if unknown)
    std::vector<int> dams;              // Map: New Index -> New Dam Index (-1 if unknown)
    std::vector<int> proband_indices;   // Indices of the target probands in the new mapping
    std::vector<double> D;              // Diagonal coefficients for Colleau's method
};

// Extracts ancestors, topologically sorts them, and prepares data structures
RemappedPedigree prepare_colleau_pedigree(Pedigree<>& pedigree, const std::vector<int>& proband_ids) {
    RemappedPedigree data;
    
    // 1. Identify all ancestors using BFS/DFS
    std::unordered_set<int> visited;
    std::vector<int> stack;
    std::vector<int> nodes; // Will store all relevant individuals

    // Initialize stack with existing probands
    for (int pid : proband_ids) {
        if (pedigree.individuals.find(pid) != pedigree.individuals.end()) {
            if (visited.find(pid) == visited.end()) {
                visited.insert(pid);
                stack.push_back(pid);
                nodes.push_back(pid);
            }
        }
    }

    size_t head = 0;
    while(head < stack.size()) {
        int id = stack[head++];
        Individual<>* ind = pedigree.individuals.at(id);
        
        // Check parents
        Individual<>* parents[2] = {ind->father, ind->mother};
        for (auto* p : parents) {
            if (p) {
                // We rely on the pedigree map to find IDs. 
                // Assuming pointers in 'individuals' are consistent with the map keys.
                // In GeneaKit, we often have to trust the pointers match the map.
                // We use the pointer to find the ID via the map or assume ID is retrievable.
                // Since Individual struct might not store ID, we search or assume consistent pointer usage.
                // Optimally, we use the pointers for the sort and only map back to IDs later.
                // However, 'compute_inbreedings' needs IDs. 
                // Let's assume we can retrieve ID. If not, we iterate the map (slow) or assume `id` field exists.
                // The provided 'Individual' struct in previous code didn't show an 'id' field, but 'pedigree.ids' exists.
                // We will assume we can get the ID. For now, let's use the 'extract_pedigree' logic implicitly.
                
                // CRITICAL: We need the ID. Let's assume p->id exists or we can map. 
                // In the provided `extract_pedigree` usage, it uses `p->rank`.
                // Let's use the `pedigree.individuals` map to find the ID is too slow.
                // We will use the `extract_pedigree` helper from the existing library if available.
                // Since we are replacing a function, we'll write a robust topological sort using pointers.
            }
        }
    }

    // Re-implementation of extraction + topological sort
    // We use a set of pointers to avoid ID lookups during traversal
    std::unordered_set<Individual<>*> node_set;
    std::vector<Individual<>*> q; // Queue for BFS
    
    for (int pid : proband_ids) {
        if (pedigree.individuals.count(pid)) {
            Individual<>* ind = pedigree.individuals.at(pid);
            if (node_set.find(ind) == node_set.end()) {
                node_set.insert(ind);
                q.push_back(ind);
            }
        }
    }

    // BFS to gather ancestors
    size_t q_idx = 0;
    while(q_idx < q.size()) {
        Individual<>* curr = q[q_idx++];
        if (curr->father && node_set.find(curr->father) == node_set.end()) {
            node_set.insert(curr->father);
            q.push_back(curr->father);
        }
        if (curr->mother && node_set.find(curr->mother) == node_set.end()) {
            node_set.insert(curr->mother);
            q.push_back(curr->mother);
        }
    }

    // Sort by rank (generation) -> Oldest first
    std::sort(q.begin(), q.end(), [](Individual<>* a, Individual<>* b) {
        return a->rank < b->rank;
    });

    // Build Mapping
    std::unordered_map<Individual<>*, int> ptr_to_idx;
    int n = q.size();
    data.num_individuals = n;
    data.original_ids.resize(n);
    data.sires.assign(n, -1);
    data.dams.assign(n, -1);

    // We need to recover original IDs for the inbreeding function
    // This is expensive ($O(N)$ map scan) if ID is not in struct. 
    // Optimization: Build a reverse map once.
    std::unordered_map<Individual<>*, int> rev_lookup;
    for (auto const& [id, ptr] : pedigree.individuals) {
        rev_lookup[ptr] = id;
    }

    for (int i = 0; i < n; ++i) {
        ptr_to_idx[q[i]] = i;
        data.original_ids[i] = rev_lookup[q[i]];
    }

    for (int i = 0; i < n; ++i) {
        if (q[i]->father) data.sires[i] = ptr_to_idx[q[i]->father];
        if (q[i]->mother) data.dams[i] = ptr_to_idx[q[i]->mother];
    }

    for (int pid : proband_ids) {
        if (pedigree.individuals.count(pid)) {
            data.proband_indices.push_back(ptr_to_idx[pedigree.individuals.at(pid)]);
        }
    }

    return data;
}

// Replaces the original compute_kinships_sparse with Colleau's method
SparseResult compute_kinships_sparse(
    Pedigree<> &pedigree, 
    std::vector<int> proband_ids, 
    bool verbose, 
    bool symmetric_coo
) {
    if (proband_ids.empty()) {
        proband_ids = get_proband_ids(pedigree);
    }

    if (verbose) std::cout << "Preparing pedigree for Colleau's method..." << std::endl;

    // 1. Prepare Pedigree (Topological Sort & Renumbering)
    RemappedPedigree data = prepare_colleau_pedigree(pedigree, proband_ids);
    int N = data.num_individuals;
    int M = data.proband_indices.size();

    if (verbose) std::cout << "Ancestors: " << N << ", Probands: " << M << std::endl;

    // 2. Compute Inbreeding (F) to get Diagonal (D)
    // We use the existing highly efficient compute_inbreedings function (Sargolzaei).
    // We calculate F for *all* ancestors in our subset.
    if (verbose) std::cout << "Computing inbreeding coefficients..." << std::endl;
    std::vector<double> F = compute_inbreedings(pedigree, data.original_ids);

    // 3. Compute D vector
    // D[i] is the within-family segregation variance.
    // Base formula (Colleau/Quaas): 
    // D[i] = 1 (if both parents unknown)
    // D[i] = 0.75 - 0.25*F_p (if one parent p known)
    // D[i] = 0.5 - 0.25*(F_s + F_d) (if both known)
    data.D.resize(N);
    for (int i = 0; i < N; ++i) {
        int s = data.sires[i];
        int d = data.dams[i];
        
        double f_s = (s != -1) ? F[s] : 0.0;
        double f_d = (d != -1) ? F[d] : 0.0;
        
        if (s == -1 && d == -1) {
            data.D[i] = 1.0;
        } else if (s != -1 && d != -1) {
            data.D[i] = 0.5 - 0.25 * (f_s + f_d);
        } else {
            // One parent known
            double f_known = (s != -1) ? f_s : f_d;
            data.D[i] = 0.75 - 0.25 * f_known;
        }
    }

    if (verbose) std::cout << "Running indirect calculation (Ax = y)..." << std::endl;

    // 4. Colleau's Algorithm (Parallelized by Proband)
    // We compute one column of the relationship matrix at a time.
    // Column j corresponds to setting x = e_j.
    
    // We need to output a SparseResult. 
    // To minimize memory contention, each thread will collect its own triplets.
    
    struct Triplet { int r; int c; float v; };
    std::vector<std::vector<Triplet>> thread_results;
    
    #pragma omp parallel
    {
        // Thread-local storage
        std::vector<double> w(N);
        std::vector<Triplet> local_triplets;
        
        #pragma omp single
        {
            thread_results.resize(omp_get_num_threads());
        }
        
        #pragma omp for schedule(dynamic)
        for (int k = 0; k < M; ++k) {
            int proband_idx = data.proband_indices[k];
            
            // --- Step 1: Upward Pass ---
            // Solve (I - T') * w = x
            // x has a single 1.0 at proband_idx.
            // We propagate contributions from progeny to parents.
            // Loop backwards from proband_idx to 0.
            
            std::fill(w.begin(), w.end(), 0.0);
            w[proband_idx] = 1.0;
            
            // Optimization: Only iterate from the proband down to the oldest ancestor
            for (int i = proband_idx; i >= 0; --i) {
                double val = w[i];
                if (val == 0.0) continue;
                
                int s = data.sires[i];
                int d = data.dams[i];
                // T' implies moving from child (i) to parent (s, d)
                // The coefficient in T is 0.5
                if (s != -1) w[s] += 0.5 * val;
                if (d != -1) w[d] += 0.5 * val;
            }
            
            // --- Step 2: Scaling ---
            // w = D * w
            for (int i = 0; i <= proband_idx; ++i) {
                if (w[i] != 0.0) w[i] *= data.D[i];
            }
            
            // --- Step 3: Downward Pass ---
            // Solve (I - T) * y = w
            // We propagate contributions from parents to progeny.
            // y = w + T * y
            // We can reuse the 'w' vector to store 'y'.
            // Iterate forward from 0 to N-1.
            
            for (int i = 0; i < N; ++i) {
                double val = w[i]; // Starts with D*w_up
                
                int s = data.sires[i];
                int d = data.dams[i];
                
                double inherited = 0.0;
                if (s != -1) inherited += w[s];
                if (d != -1) inherited += w[d];
                
                w[i] = val + 0.5 * inherited;
            }
            
            // --- Step 4: Extract Results ---
            // w now contains the full column 'k' of the relationship matrix A.
            // w[p] is the relationship between proband k and individual p.
            // We only save w[p] where p is also a proband.
            
            // For symmetric COO (full matrix): Save all (row, col) pairs.
            // For standard/CSR (lower triangle): Save if row >= col.
            
            // Note: 'k' is the column index (proband k).
            // We iterate all other probands 'l' as rows.
            for (int l = 0; l < M; ++l) {
                int row_proband_idx = data.proband_indices[l];
                double val = w[row_proband_idx];
                
                if (std::abs(val) > 1e-9) { // Sparsity check
                    if (symmetric_coo) {
                        local_triplets.push_back({l, k, (float)val});
                    } else {
                        // Store Lower Triangle only (Row >= Col)
                        if (l >= k) {
                            local_triplets.push_back({l, k, (float)val});
                        }
                    }
                }
            }
        }
        
        // Save local results
        thread_results[omp_get_thread_num()] = std::move(local_triplets);
    }

    // 5. Aggregate Results
    if (verbose) std::cout << "Aggregating results..." << std::endl;
    
    SparseResult result;
    size_t total_nnz = 0;
    for (const auto& vec : thread_results) total_nnz += vec.size();
    
    // For COO format (implied by symmetric_coo based on previous code usage)
    if (symmetric_coo) {
        result.rows.reserve(total_nnz);
        result.indices.reserve(total_nnz);
        result.data.reserve(total_nnz);
        
        for (const auto& vec : thread_results) {
            for (const auto& t : vec) {
                result.rows.push_back(t.r);
                result.indices.push_back(t.c);
                result.data.push_back(t.v / 2.0f);
            }
        }
    } else {
        // Construct CSR for Lower Triangle
        // We have triplets. We need to sort them by Row then Col.
        std::vector<Triplet> all_triplets;
        all_triplets.reserve(total_nnz);
        for (const auto& vec : thread_results) {
            all_triplets.insert(all_triplets.end(), vec.begin(), vec.end());
        }
        
        // Sort
        std::sort(all_triplets.begin(), all_triplets.end(), 
            [](const Triplet& a, const Triplet& b) {
                if (a.r != b.r) return a.r < b.r;
                return a.c < b.c;
            });
            
        // Convert to CSR
        result.data.reserve(total_nnz);
        result.indices.reserve(total_nnz);
        result.indptr.reserve(M + 1);
        result.indptr.push_back(0);
        
        int current_row = 0;
        for (const auto& t : all_triplets) {
            while (current_row < t.r) {
                result.indptr.push_back(result.data.size());
                current_row++;
            }
            result.data.push_back(t.v / 2.0f);
            result.indices.push_back(t.c);
        }
        // Finish remaining rows
        while (current_row < M) {
            result.indptr.push_back(result.data.size());
            current_row++;
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
