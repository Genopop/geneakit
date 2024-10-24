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
    std::unordered_set<int> set;
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
    Pedigree<> &pedigree,
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
    vertex_cuts = intersect_both_directions(pedigree, bottom_up, top_down);
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
    // Extract the relevant individuals from the pedigree
    Pedigree<> extracted_pedigree = extract_pedigree(pedigree, proband_ids);
    // Cut the vertices (a vertex corresponds to an individual)
    std::vector<std::vector<int>> vertex_cuts;
    vertex_cuts = cut_vertices(extracted_pedigree, proband_ids);
    // Calculate the size of each pair of vertex cuts
    std::vector<int> sizes;
    for (int i = 0; i < (int) vertex_cuts.size() - 1; i++) {
        sizes.push_back(vertex_cuts[i].size() + vertex_cuts[i + 1].size());
    }
    // Get the maximum size
    int max_size = *max_element(sizes.begin(), sizes.end());
    // Get the two sizes that when summed give the maximum size
    int size1 = 0, size2 = 0;
    for (int i = 0; i < (int) sizes.size(); i++) {
        if (vertex_cuts[i].size() + vertex_cuts[i + 1].size() == max_size) {
            size1 = vertex_cuts[i].size();
            size2 = vertex_cuts[i + 1].size();
            break;
        }
    }
    // Calculate the required memory
    double requirement = (size1 * size1 + size2 * size2) * sizeof(double) / 1e9;
    return requirement;
}

// Returns the kinship matrix using the algorithm from Morin et al.
Matrix<double> compute_kinships(Pedigree<> &pedigree,
    std::vector<int> proband_ids, bool verbose) {
    // Get the proband IDs if they are not provided
    if (proband_ids.empty()) {
        proband_ids = get_proband_ids(pedigree);
    }
    // Extract the relevant individuals from the pedigree
    Pedigree<> extracted_pedigree = extract_pedigree(pedigree, proband_ids);
    // Convert the pedigree to a kinship pedigree
    Pedigree<Index> kinship_pedigree(extracted_pedigree);
    // Cut the vertices (a vertex corresponds to an individual)
    std::vector<std::vector<int>> vertex_cuts;
    vertex_cuts = cut_vertices(extracted_pedigree, proband_ids);
    // Initialize the founders' kinship matrix
    Matrix<double> founder_matrix = zeros<double>(
        vertex_cuts[0].size(), vertex_cuts[0].size()
    );
    #pragma omp parallel for
    for (size_t i = 0; i < vertex_cuts[0].size(); i++) {
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

// Returns the kinship coefficient between two individuals.
// An implementation of the recursive algorithm from Karigl.
double compute_kinship(const Individual<> *individual1,
    const Individual<> *individual2) {
    double kinship = 0.0;
    if (individual1->rank == individual2->rank) {
        // It's the same individual.
        kinship = 0.5;
        if (individual1->father && individual2->mother) {
            kinship += 0.5 * compute_kinship(
                individual1->father, individual2->mother);
        }
    } else if (individual1->rank < individual2->rank) {
        // Karigl's recursive algorithm.
        if (individual2->father) {
            kinship += 0.5 * compute_kinship(individual1, individual2->father);
        }
        if (individual2->mother) {
            kinship += 0.5 * compute_kinship(individual1, individual2->mother);
        }
    } else {
        if (individual1->father) {
            kinship += 0.5 * compute_kinship(individual1->father, individual2);
        }
        if (individual1->mother) {
            kinship += 0.5 * compute_kinship(individual1->mother, individual2);
        }
    }
    return kinship;
}

// Returns the inbreeding coefficients of a vector of individuals.
std::vector<double> compute_inbreedings(Pedigree<> &pedigree,
    std::vector<int> proband_ids) {
    std::vector<double> inbreeding_coefficients;
    if (proband_ids.empty()) {
        proband_ids = get_proband_ids(pedigree);
    }
    for (const int id : proband_ids) {
        Individual<> *individual = pedigree.individuals.at(id);
        if (!individual->father || !individual->mother) {
            double inbreeding = 0.0;
            inbreeding_coefficients.push_back(inbreeding);
        } else {
            double inbreeding = compute_kinship(
                individual->father, individual->mother
            );
            inbreeding_coefficients.push_back(inbreeding);
        }
    }
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
    // Extract the relevant individuals from the pedigree
    Pedigree new_pedigree = extract_pedigree(
        pedigree, proband_ids, ancestor_ids
    );
    // Convert the pedigree to a kinship pedigree
    Pedigree<Contribution> contribution_pedigree(new_pedigree);
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