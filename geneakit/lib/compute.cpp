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
std::vector<int> get_previous_generation(Pedigree<> &pedigree,
    std::vector<int> &ids) {
    phmap::flat_hash_set<int> set;
    for (const int id : ids) {
        Individual<> *individual = pedigree.individuals.at(id);
        if (individual->father) {
            set.insert(individual->father->id);
        }
        if (individual->mother) {
            set.insert(individual->mother->id);
        }
    }
    std::vector<int> previous_generation(set.begin(), set.end());
    // Sort the individuals by ID
    sort(previous_generation.begin(), previous_generation.end());
    return previous_generation;
}

// Go from the bottom to the top of the pedigree
std::vector<std::vector<int>> get_generations(Pedigree<> &pedigree,
    std::vector<int> &proband_ids) {
    std::vector<std::vector<int>> generations;
    std::vector<int> generation;
    for (const int id : proband_ids) {
        generation.push_back(id);
    }
    while (!generation.empty()) {
        generations.push_back(generation);
        generation = get_previous_generation(pedigree, generation);
    }
    return generations;
}

// Drag the individuals up
std::vector<std::vector<int>> copy_bottom_up(
    std::vector<std::vector<int>> &generations) {
    std::vector<std::vector<int>> bottom_up;
    std::vector<int> ids;
    for (const int id : generations[0]) {
        ids.push_back(id);
    }
    bottom_up.push_back(ids);
    for (int i = 0; i < (int) generations.size() - 1; i++) {
        std::vector<int> next_generation;
        std::set<int> set1(bottom_up[i].begin(), bottom_up[i].end());
        std::set<int> set2;
        for (const int id : generations[i + 1]) {
            set2.insert(id);
        }
        set_union(
            set1.begin(), set1.end(),
            set2.begin(), set2.end(),
            std::back_inserter(next_generation)
        );
        bottom_up.push_back(next_generation);
    }
    reverse(bottom_up.begin(), bottom_up.end());
    return bottom_up;
}

// Drag the individuals down
std::vector<std::vector<int>> copy_top_down(
    std::vector<std::vector<int>> &generations) {
    reverse(generations.begin(), generations.end());
    std::vector<std::vector<int>> top_down;
    std::vector<int> ids;
    for (const int id : generations[0]) {
        ids.push_back(id);
    }
    sort(ids.begin(), ids.end());
    top_down.push_back(ids);
    for (int i = 0; i < (int) generations.size() - 1; i++) {
        std::vector<int> next_generation;
        std::set<int> set1, set2;
        for (int id : top_down[i]) {
            set1.insert(id);
        }
        for (const int id : generations[i + 1]) {
            set2.insert(id);
        }
        set_union(
            set1.begin(), set1.end(),
            set2.begin(), set2.end(),
            std::back_inserter(next_generation)
        );
        top_down.push_back(next_generation);
    }
    return top_down;
}

// Find the intersection of the two sets
std::vector<std::vector<int>> intersect_both_directions(
    std::vector<std::vector<int>> &bottom_up,
    std::vector<std::vector<int>> &top_down) {
    std::vector<std::vector<int>> vertex_cuts;
    for (int i = 0; i < (int) bottom_up.size(); i++) {
        std::vector<int> vertex_cut;
        std::set<int> set1(bottom_up[i].begin(), bottom_up[i].end());
        std::set<int> set2(top_down[i].begin(), top_down[i].end());
        std::vector<int> intersection_result;
        set_intersection(
            set1.begin(), set1.end(),
            set2.begin(), set2.end(),
            std::back_inserter(intersection_result)
        );
        for (int id : intersection_result) {
            vertex_cut.push_back(id);
        }
        vertex_cuts.push_back(vertex_cut);
    }
    return vertex_cuts;
}

// Separate the individuals into generations where individuals who appear
// in two non-contiguous generations are dragged in the generations in-between.
// Based on the recursive-cut algorithm from Kirkpatrick et al.
std::vector<std::vector<int>> cut_vertices(
    Pedigree<> &pedigree, std::vector<int> &proband_ids) {
    std::vector<std::vector<int>> generations, vertex_cuts;
    generations = get_generations(pedigree, proband_ids);
    std::vector<std::vector<int>> bottom_up, top_down;
    bottom_up = copy_bottom_up(generations);
    top_down = copy_top_down(generations);
    vertex_cuts = intersect_both_directions(bottom_up, top_down);
    // Set the last vertex cut to the probands
    std::vector<int> probands;
    for (const int id : proband_ids) {
        probands.push_back(id);
    }
    vertex_cuts[vertex_cuts.size() - 1] = probands;
    return vertex_cuts;
}

// Returns the kinship coefficient between two individuals.
// A modified version of the recursive algorithm from Karigl.
double compute_kinship(
    const Individual<Index> *individual1,
    const Individual<Index> *individual2,
    Matrix<double> &founder_matrix) {
    double kinship = 0.0;
    const int founder_index1 = individual1->data.index;
    const int founder_index2 = individual2->data.index;
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
    std::vector<Individual<Index> *> &vertex_cut,
    Matrix<double> &founder_matrix,
    Matrix<double> &proband_matrix) {
    #pragma omp parallel for
    for (int i = 0; i < (int) vertex_cut.size(); i++) {
        const Individual<Index> *individual = vertex_cut[i];
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
    std::vector<Individual<Index> *> &vertex_cut,
    Matrix<double> &founder_matrix,
    Matrix<double> &proband_matrix) {
    #pragma omp parallel for
    for (int i = 0; i < (int) vertex_cut.size(); i++) {
        Individual<Index> *individual1 = vertex_cut[i];
        #pragma omp parallel for
        for (int j = 0; j < i; j++) {
            Individual<Index> *individual2 = vertex_cut[j];
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
    // Cut the vertices (a vertex corresponds to an individual)
    std::vector<std::vector<int>> vertex_cuts = cut_vertices(pedigree, proband_ids);
    
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
    Pedigree<Index> kinship_pedigree(pedigree);
    // Cut the vertices (a vertex corresponds to an individual)
    std::vector<std::vector<int>> vertex_cuts;
    vertex_cuts = cut_vertices(pedigree, proband_ids);
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
        for (std::vector<int> vertex_cut : vertex_cuts) {
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
        for (const int id : vertex_cuts[i]) {
            Individual<Index> *individual = kinship_pedigree.individuals.at(id);
            individual->data.index = founder_index++;
        }
        // Initialize the probands' kinship matrix
        Matrix<double> proband_matrix(
            vertex_cuts[i + 1].size(), vertex_cuts[i + 1].size()
        );
        std::vector<Individual<Index> *> probands;
        for (const int id : vertex_cuts[i + 1]) {
            probands.push_back(kinship_pedigree.individuals.at(id));
        }
        compute_kinship_with_oneself(
            probands, founder_matrix, proband_matrix
        );
        compute_kinship_between_probands(
            probands, founder_matrix, proband_matrix
        );
        // The current generation becomes the previous generation
        founder_matrix = std::move(proband_matrix);
    }
    return founder_matrix;
}

// Convert CSR matrix to hashmap (ONLY used between generations, not at the end)
phmap::flat_hash_map<int, phmap::flat_hash_map<int, float>> csr_to_hashmap(const CSRMatrix& csr) {
    phmap::flat_hash_map<int, phmap::flat_hash_map<int, float>> hashmap_matrix;
    
    #pragma omp parallel
    {
        // Each thread builds its own local hashmap
        phmap::flat_hash_map<int, phmap::flat_hash_map<int, float>> local_hashmap;
        
        #pragma omp for
        for (int i = 0; i < csr.n_rows; i++) {
            if (csr.indptr[i + 1] > csr.indptr[i]) {
                local_hashmap.emplace(i, phmap::flat_hash_map<int, float>());
                for (int j = csr.indptr[i]; j < csr.indptr[i + 1]; j++) {
                    if (csr.data[j] != 0.0f) {
                        local_hashmap[i].emplace(csr.indices[j], csr.data[j]);
                    }
                }
            }
        }
        
        // Merge local hashmaps into global hashmap
        #pragma omp critical
        {
            for (auto& pair : local_hashmap) {
                hashmap_matrix[pair.first] = std::move(pair.second);
            }
        }
    }
    
    return hashmap_matrix;
}

// Sparse version of compute_kinship function for fast retrieval from hashmap
float sparse_compute_kinship(
    const Individual<Index> *individual1,
    const Individual<Index> *individual2,
    const phmap::flat_hash_map<int, phmap::flat_hash_map<int, float>>& founder_matrix) {
    
    float kinship = 0.0f;
    const int founder_index1 = individual1->data.index;
    const int founder_index2 = individual2->data.index;
    
    if (founder_index1 != -1 && founder_index2 != -1) {
        // Look up kinship coefficient in founder matrix
        auto it1 = founder_matrix.find(founder_index1);
        if (it1 != founder_matrix.end()) {
            auto it2 = it1->second.find(founder_index2);
            if (it2 != it1->second.end()) {
                kinship = it2->second;
            }
        }
        // Try symmetric lookup if not found
        if (kinship == 0.0f && founder_index1 != founder_index2) {
            auto it1_sym = founder_matrix.find(founder_index2);
            if (it1_sym != founder_matrix.end()) {
                auto it2_sym = it1_sym->second.find(founder_index1);
                if (it2_sym != it1_sym->second.end()) {
                    kinship = it2_sym->second;
                }
            }
        }
    } else if (founder_index1 != -1) {
        // Recursive case: individual2 is not a founder
        if (individual2->father) {
            kinship += 0.5f * sparse_compute_kinship(
                individual1, individual2->father, founder_matrix);
        }
        if (individual2->mother) {
            kinship += 0.5f * sparse_compute_kinship(
                individual1, individual2->mother, founder_matrix);
        }
    } else if (founder_index2 != -1) {
        // Recursive case: individual1 is not a founder
        if (individual1->father) {
            kinship += 0.5f * sparse_compute_kinship(
                individual1->father, individual2, founder_matrix);
        }
        if (individual1->mother) {
            kinship += 0.5f * sparse_compute_kinship(
                individual1->mother, individual2, founder_matrix);
        }
    } else if (individual1->rank == individual2->rank) {
        // Same individual
        kinship = 0.5f;
        if (individual1->father && individual1->mother) {
            kinship += 0.5f * sparse_compute_kinship(
                individual1->father, individual1->mother, founder_matrix);
        }
    } else if (individual1->rank < individual2->rank) {
        // Karigl's recursive algorithm
        if (individual2->father) {
            kinship += 0.5f * sparse_compute_kinship(
                individual1, individual2->father, founder_matrix);
        }
        if (individual2->mother) {
            kinship += 0.5f * sparse_compute_kinship(
                individual1, individual2->mother, founder_matrix);
        }
    } else {
        // individual1->rank > individual2->rank
        if (individual1->father) {
            kinship += 0.5f * sparse_compute_kinship(
                individual1->father, individual2, founder_matrix);
        }
        if (individual1->mother) {
            kinship += 0.5f * sparse_compute_kinship(
                individual1->mother, individual2, founder_matrix);
        }
    }
    return kinship;
}

// Build CSR matrix for kinship with oneself
void sparse_compute_kinship_with_oneself(
    std::vector<Individual<Index> *> &vertex_cut,
    const phmap::flat_hash_map<int, phmap::flat_hash_map<int, float>>& founder_matrix,
    CSRMatrix& result_matrix) {
    
    int n = static_cast<int>(vertex_cut.size());
    std::vector<std::vector<std::pair<int, float>>> temp_rows(n);
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        const Individual<Index> *individual = vertex_cut[i];
        float kinship = 0.5f;
        if (individual->father && individual->mother) {
            kinship += 0.5f * sparse_compute_kinship(
                individual->father, individual->mother, founder_matrix
            );
        }
        
        // Store diagonal element
        if (kinship != 0.0f) {
            temp_rows[i].emplace_back(i, kinship);
        }
    }
    
    // Build final CSR matrix
    result_matrix.indptr[0] = 0;
    for (int i = 0; i < n; i++) {
        result_matrix.indptr[i + 1] = result_matrix.indptr[i] + static_cast<int>(temp_rows[i].size());
    }
    
    int total_nnz = result_matrix.indptr[n];
    result_matrix.indices.resize(total_nnz);
    result_matrix.data.resize(total_nnz);
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int start = result_matrix.indptr[i];
        for (size_t j = 0; j < temp_rows[i].size(); j++) {
            result_matrix.indices[start + static_cast<int>(j)] = temp_rows[i][j].first;
            result_matrix.data[start + static_cast<int>(j)] = temp_rows[i][j].second;
        }
    }
}

// Build CSR matrix for kinship between probands
void sparse_compute_kinship_between_probands(
    std::vector<Individual<Index> *> &vertex_cut,
    const phmap::flat_hash_map<int, phmap::flat_hash_map<int, float>>& founder_matrix,
    CSRMatrix& result_matrix) {
    
    int n = static_cast<int>(vertex_cut.size());
    std::vector<std::vector<std::pair<int, float>>> temp_rows(n);
    
    // Step 1: Compute the lower triangle of the kinship matrix in parallel.
    // Each thread only writes to its own row temp_rows[i], so this is race-free.
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        Individual<Index> *individual1 = vertex_cut[i];
        for (int j = 0; j < i; j++) {
            Individual<Index> *individual2 = vertex_cut[j];
            float kinship = sparse_compute_kinship(
                individual1, individual2, founder_matrix
            );
            
            if (kinship != 0.0f) {
                // Store pair (column_index, value) for the lower triangle
                temp_rows[i].emplace_back(j, kinship);
            }
        }
    }
    
    // Step 2: Create the full symmetric matrix from the lower triangle.
    // This part is serial to avoid race conditions when writing to temp_rows[j].
    for (int i = 0; i < n; i++) {
        for (const auto& pair : temp_rows[i]) {
            int j = pair.first;
            float kinship = pair.second;
            // Add the symmetric entry to the upper triangle part.
            temp_rows[j].emplace_back(i, kinship);
        }
    }

    // Step 3: Sort each row by column index in parallel.
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        std::sort(temp_rows[i].begin(), temp_rows[i].end());
    }
    
    // Step 4: Build the final CSR matrix from the sorted rows.
    result_matrix.indptr[0] = 0;
    int total_nnz = 0;
    for (int i = 0; i < n; i++) {
        total_nnz += static_cast<int>(temp_rows[i].size());
        result_matrix.indptr[i + 1] = total_nnz;
    }
    
    result_matrix.indices.resize(total_nnz);
    result_matrix.data.resize(total_nnz);
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int start = result_matrix.indptr[i];
        for (size_t j = 0; j < temp_rows[i].size(); j++) {
            result_matrix.indices[start + j] = temp_rows[i][j].first;
            result_matrix.data[start + j] = temp_rows[i][j].second;
        }
    }
}

// Merge two CSR matrices
CSRMatrix merge_csr_matrices(const CSRMatrix& matrix1, const CSRMatrix& matrix2) {
    CSRMatrix result(matrix1.n_rows, matrix1.n_cols);
    
    std::vector<phmap::flat_hash_map<int, float>> temp_rows(matrix1.n_rows);
    
    // Add elements from matrix1
    #pragma omp parallel for
    for (int i = 0; i < matrix1.n_rows; i++) {
        for (int j = matrix1.indptr[i]; j < matrix1.indptr[i + 1]; j++) {
            temp_rows[i][matrix1.indices[j]] += matrix1.data[j];
        }
    }
    
    // Add elements from matrix2
    #pragma omp parallel for
    for (int i = 0; i < matrix2.n_rows; i++) {
        for (int j = matrix2.indptr[i]; j < matrix2.indptr[i + 1]; j++) {
            temp_rows[i][matrix2.indices[j]] += matrix2.data[j];
        }
    }
    
    // Build final CSR matrix
    result.indptr[0] = 0;
    for (int i = 0; i < matrix1.n_rows; i++) {
        result.indptr[i + 1] = result.indptr[i] + static_cast<int>(temp_rows[i].size());
    }
    
    int total_nnz = result.indptr[matrix1.n_rows];
    result.indices.resize(total_nnz);
    result.data.resize(total_nnz);
    
    #pragma omp parallel for
    for (int i = 0; i < matrix1.n_rows; i++) {
        int start = result.indptr[i];
        int idx = 0;
        
        std::vector<std::pair<int, float>> row_data;
        row_data.reserve(temp_rows[i].size());
        for (const auto& pair : temp_rows[i]) {
            if (pair.second != 0.0f) {
                row_data.emplace_back(pair.first, pair.second);
            }
        }
        std::sort(row_data.begin(), row_data.end());
        
        for (const auto& pair : row_data) {
            result.indices[start + idx] = pair.first;
            result.data[start + idx] = pair.second;
            idx++;
        }
    }
    
    return result;
}

// Returns a sparse matrix of the kinship coefficients.
std::tuple<std::vector<int>, std::vector<int>, std::vector<float>>
compute_sparse_kinships(Pedigree<> &pedigree,
    std::vector<int> proband_ids, bool verbose) {
    
    // Get the proband IDs if they are not provided
    if (proband_ids.empty()) {
        proband_ids = get_proband_ids(pedigree);
    }
    
    // Handle edge cases
    if (proband_ids.empty()) {
        return {{}, {0}, {}};
    }
    
    if (proband_ids.size() == 1) {
        return {{0}, {0, 1}, {0.5f}};
    }
    
    // Convert the pedigree to a kinship pedigree
    Pedigree<Index> kinship_pedigree(pedigree);
    
    // Cut the vertices
    std::vector<std::vector<int>> vertex_cuts = cut_vertices(pedigree, proband_ids);
    
    // Initialize founder matrix as hashmap
    phmap::flat_hash_map<int, phmap::flat_hash_map<int, float>> founder_matrix;
    
    // Initialize founders with self-kinship of 0.5
    if (!vertex_cuts.empty()) {
        for (int i = 0; i < static_cast<int>(vertex_cuts[0].size()); i++) {
            founder_matrix[i][i] = 0.5f;
        }
    }
    
    if (verbose) {
        int count = 0;
        for (const std::vector<int>& vertex_cut : vertex_cuts) {
            printf("Cut size %d/%d: %d\n",
                ++count, static_cast<int>(vertex_cuts.size()), static_cast<int>(vertex_cut.size()));
        }
        printf("Computing the sparse kinship matrix:\n");
    }
    
    int cut_count = 0;
    
    // Go from the top to the bottom of the pedigree
    for (int i = 0; i < static_cast<int>(vertex_cuts.size()) - 1; i++) {
        if (verbose) {
            printf("Cut %d out of %d\n", ++cut_count, static_cast<int>(vertex_cuts.size()));
        }
        
        // Index the founders
        int founder_index = 0;
        for (const int id : vertex_cuts[i]) {
            Individual<Index> *individual = kinship_pedigree.individuals.at(id);
            individual->data.index = founder_index++;
        }
        
        // Get probands for this generation
        std::vector<Individual<Index> *> probands;
        for (const int id : vertex_cuts[i + 1]) {
            probands.push_back(kinship_pedigree.individuals.at(id));
        }
        
        // Build CSR matrices for diagonal and off-diagonal elements
        CSRMatrix diagonal_matrix(static_cast<int>(probands.size()), static_cast<int>(probands.size()));
        CSRMatrix offdiagonal_matrix(static_cast<int>(probands.size()), static_cast<int>(probands.size()));
        
        // Compute kinships
        sparse_compute_kinship_with_oneself(probands, founder_matrix, diagonal_matrix);
        sparse_compute_kinship_between_probands(probands, founder_matrix, offdiagonal_matrix);
        
        // Merge diagonal and off-diagonal matrices
        CSRMatrix proband_matrix = merge_csr_matrices(diagonal_matrix, offdiagonal_matrix);
        
        // If this is the last generation, return the CSR matrix directly
        if (i == static_cast<int>(vertex_cuts.size()) - 2) {
            if (verbose) {
                printf("Returning final CSR matrix directly\n");
            }
            return {std::move(proband_matrix.indices), 
                    std::move(proband_matrix.indptr), 
                    std::move(proband_matrix.data)};
        }
        
        // Otherwise, convert CSR proband_matrix to hashmap founder_matrix for next iteration
        founder_matrix = csr_to_hashmap(proband_matrix);
    }
    
    // This should never be reached, but handle empty case
    return {{}, {0}, {}};
}

// Returns the mean kinship coefficient of a kinship matrix.
double compute_mean_kinship(Matrix<double> &kinship_matrix) {
    // Initialize the mean kinship coefficient
    double total = 0.0;
    // Compute the mean kinship coefficient
    for (int i = 0; i < (int) kinship_matrix.size(); i++) {
        for (int j = 0; j < i; j++) {
            total += kinship_matrix[i][j];
        }
    }
    uint64_t count = (kinship_matrix.size() * (kinship_matrix.size() - 1)) / 2;
    return total / count;
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

// Returns the meioses between two individuals.
// A modified version of the recursive kinship algorithm from Karigl.
double compute_meioses(
    const Individual<Index> *individual1,
    const Individual<Index> *individual2,
    Matrix<char> &founder_matrix) {
    char minimum, paternal = 127, maternal = 127;
    const int founder_index1 = individual1->data.index;
    const int founder_index2 = individual2->data.index;
    if (founder_index1 != -1 && founder_index2 != -1) {
        // The kinship coefficient is stored in the founder matrix.
        return founder_matrix[founder_index1][founder_index2];
    } else if (founder_index1 != -1) {
        // The kinship coefficient was computed between individual 1
        // and the parents of individual 2.
        if (individual2->father) {
            paternal = compute_meioses(
                individual1, individual2->father, founder_matrix);
        }
        if (individual2->mother) {
            maternal = compute_meioses(
                individual1, individual2->mother, founder_matrix);
        }
    } else if (founder_index2 != -1) {
        // Vice versa.
        if (individual1->father) {
            paternal = compute_meioses(
                individual1->father, individual2, founder_matrix);
        }
        if (individual1->mother) {
            maternal = compute_meioses(
                individual1->mother, individual2, founder_matrix);
        }
    } else if (individual1->rank == individual2->rank) {
        // It's the same individual.
        return 0;
    } else if (individual1->rank < individual2->rank) {
        // Karigl's recursive algorithm.
        if (individual2->father) {
            paternal = compute_meioses(
                individual1, individual2->father, founder_matrix);
        }
        if (individual2->mother) {
            maternal = compute_meioses(
                individual1, individual2->mother, founder_matrix);
        }
    } else {
        if (individual1->father) {
            paternal = compute_meioses(
                individual1->father, individual2, founder_matrix);
        }
        if (individual1->mother) {
            maternal = compute_meioses(
                individual1->mother, individual2, founder_matrix);
        }
    }
    if (paternal == 127 && maternal == 127) return 127;
    minimum = paternal < maternal ? paternal : maternal;
    return 1 + minimum;
}

// Compute the meioses between the individuals
void compute_meioses_between_probands(
    std::vector<Individual<Index> *> &vertex_cut,
    Matrix<char> &founder_matrix,
    Matrix<char> &proband_matrix) {
    #pragma omp parallel for
    for (int i = 0; i < (int) vertex_cut.size(); i++) {
        Individual<Index> *individual1 = vertex_cut[i];
        #pragma omp parallel for
        for (int j = 0; j < i; j++) {
            Individual<Index> *individual2 = vertex_cut[j];
            char meioses = compute_meioses(
                individual1, individual2, founder_matrix
            );
            proband_matrix[i][j] = meioses;
            proband_matrix[j][i] = meioses;
        }
    }
}

// Returns the meioses matrix using the algorithm from Morin et al.
Matrix<char> compute_meiotic_distances(Pedigree<> &pedigree,
    std::vector<int> proband_ids, bool verbose) {
    // Get the proband IDs if they are not provided
    if (proband_ids.empty()) {
        proband_ids = get_proband_ids(pedigree);
    }
    // Convert the pedigree to a kinship pedigree
    Pedigree<Index> meioses_pedigree(pedigree);
    // Cut the vertices (a vertex corresponds to an individual)
    std::vector<std::vector<int>> vertex_cuts;
    vertex_cuts = cut_vertices(pedigree, proband_ids);
    // Initialize the founders' kinship matrix
    Matrix<char> founder_matrix = zeros<char>(
        vertex_cuts[0].size(), vertex_cuts[0].size()
    );
    if (verbose) {
        // Print the vertex cuts
        int count = 0;
        for (std::vector<int> vertex_cut : vertex_cuts) {
            printf("Cut size %d/%d: %d\n",
                ++count, (int) vertex_cuts.size(), (int) vertex_cut.size());
        }
        printf("Computing the meioses matrix:\n");
    }
    int cut_count = 0;
    // Go from the top to the bottom of the pedigree
    for (int i = 0; i < (int) vertex_cuts.size() - 1; i++) {
        if (verbose) {
            printf("Cut %d out of %d\n", ++cut_count, (int) vertex_cuts.size());
        }
        // Index the founders
        int founder_index = 0;
        for (const int id : vertex_cuts[i]) {
            Individual<Index> *individual = meioses_pedigree.individuals.at(id);
            individual->data.index = founder_index++;
        }
        // Initialize the probands' kinship matrix
        Matrix<char> proband_matrix = zeros<char>(
            vertex_cuts[i + 1].size(), vertex_cuts[i + 1].size()
        );
        std::vector<Individual<Index> *> probands;
        for (const int id : vertex_cuts[i + 1]) {
            probands.push_back(meioses_pedigree.individuals.at(id));
        }
        compute_meioses_between_probands(
            probands, founder_matrix, proband_matrix
        );
        // The current generation becomes the previous generation
        founder_matrix = std::move(proband_matrix);
    }
    return founder_matrix;
}

// Returns the matrix of correlations between individuals.
Matrix<double> compute_correlations(Pedigree<> &pedigree,
    std::vector<int> proband_ids, bool verbose) {
    Matrix<double> correlation_matrix = compute_kinships(
        pedigree, proband_ids, verbose);
    #pragma omp parallel for
    for (int i = 0; i < (int) correlation_matrix.size(); i++) {
        #pragma omp parallel for
        for (int j = 0; j < i; j++) {
            double correlation = correlation_matrix[i][j] /
                sqrt(correlation_matrix[i][i] * correlation_matrix[j][j]);
            correlation_matrix[i][j] = correlation;
            correlation_matrix[j][i] = correlation;
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < (int) correlation_matrix.size(); i++) {
        correlation_matrix[i][i] = 1.0;
    }
    return correlation_matrix;
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
