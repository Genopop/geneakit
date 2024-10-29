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

#include "../include/describe.hpp"

// Returns the number of individuals in the pedigree
int get_number_of_individuals(Pedigree<> &pedigree) {
    return pedigree.ids.size();
}

// Returns the number of men in the pedigree
int get_number_of_men(Pedigree<> &pedigree) {
    int number_of_men = 0;
    for (int id : pedigree.ids) {
        Individual<> *individual = pedigree.individuals.at(id);
        if (individual->sex == MALE) {
            number_of_men++;
        }
    }
    return number_of_men;
}

// Returns the number of women in the pedigree
int get_number_of_women(Pedigree<> &pedigree) {
    int number_of_women = 0;
    for (int id : pedigree.ids) {
        Individual<> *individual = pedigree.individuals.at(id);
        if (individual->sex == FEMALE) {
            number_of_women++;
        }
    }
    return number_of_women;
}

// Returns the number of parent-child relationships in the pedigree
int get_number_of_parent_child_relations(Pedigree<> &pedigree) {
    int number_of_parent_child_relations = 0;
    for (int id : pedigree.ids) {
        Individual<> *individual = pedigree.individuals.at(id);
        if (individual->father) {
            number_of_parent_child_relations++;
        }
        if (individual->mother) {
            number_of_parent_child_relations++;
        }
    }
    return number_of_parent_child_relations;
}

// Returns the maximum depth of an individual's pedigree
int get_individual_depth(Individual<> *individual) {
    int maximum_depth = 0;
    if (individual->father) {
        int father_depth = get_individual_depth(individual->father);
        if (father_depth > maximum_depth) {
            maximum_depth = father_depth;
        }
    }
    if (individual->mother) {
        int mother_depth = get_individual_depth(individual->mother);
        if (mother_depth > maximum_depth) {
            maximum_depth = mother_depth;
        }
    }
    return maximum_depth + 1;
}

// Returns the depth of a pedigree
int get_pedigree_depth(Pedigree<> &pedigree) {
    int depth = 0;
    for (int id : pedigree.ids) {
        Individual<> *individual = pedigree.individuals.at(id);
        int individual_depth = get_individual_depth(individual);
        if (individual_depth > depth) {
            depth = individual_depth;
        }
    }
    return depth;
}

// Returns the mean depth of an individual's pedigree
double get_mean_individual_depth(Individual<> *individual) {
    double mean_depth = 0;
    if (individual->father) {
        mean_depth += get_mean_individual_depth(individual->father);
    }
    if (individual->mother) {
        mean_depth += get_mean_individual_depth(individual->mother);
    }
    mean_depth /= 2;
    return mean_depth + 1;
}

// Returns the mean depths of the pedigree for the specified probands.
std::vector<double> get_mean_pedigree_depths(Pedigree<> &pedigree,
    std::vector<int> proband_ids) {
    std::vector<double> mean_depths;
    if (proband_ids.empty()) {
        proband_ids = get_proband_ids(pedigree);
    }
    for (const int id : proband_ids) {
        Individual<> *individual = pedigree.individuals.at(id);
        double mean_depth = get_mean_individual_depth(individual) - 1;
        mean_depths.push_back(mean_depth);
    }
    return mean_depths;
}

// Recursively climb the pedigree to find the ancestors.
void climb_pedigree(Individual<PathLength> *individual, int depth) {
    if (individual->data.is_ancestor) {
        individual->data.min = std::min(individual->data.min, depth);
        individual->data.max = std::max(individual->data.max, depth);
        individual->data.sum += depth;
        individual->data.count++;
        return;
    }
    if (individual->father) {
        climb_pedigree(individual->father, depth + 1);
    }
    if (individual->mother) {
        climb_pedigree(individual->mother, depth + 1);
    }
}

// Returns the minimum path length to a vector of ancestors.
std::vector<int> get_min_ancestor_path_lengths(Pedigree<> &pedigree,
    std::vector<int> ancestor_ids) {
    Pedigree<PathLength> path_pedigree(pedigree);
    for (const int id : ancestor_ids) {
        Individual<PathLength> *individual = path_pedigree.individuals.at(id);
        individual->data.is_ancestor = true;
    }
    std::vector<int> proband_ids = get_proband_ids(pedigree);
    for (const int id : proband_ids) {
        Individual<PathLength> *individual = path_pedigree.individuals.at(id);
        climb_pedigree(individual, 0);
    }
    std::vector<int> min_path_lengths;
    for (const int id : ancestor_ids) {
        min_path_lengths.push_back(
            path_pedigree.individuals.at(id)->data.min
        );
    }
    return min_path_lengths;
}

// Returns the mean path length to a vector of ancestors.
std::vector<double> get_mean_ancestor_path_lengths(Pedigree<> &pedigree,
    std::vector<int> ancestor_ids) {
    Pedigree<PathLength> path_pedigree(pedigree);
    for (const int id : ancestor_ids) {
        Individual<PathLength> *individual = path_pedigree.individuals.at(id);
        individual->data.is_ancestor = true;
    }
    std::vector<int> proband_ids = get_proband_ids(pedigree);
    for (const int id : proband_ids) {
        Individual<PathLength> *individual = path_pedigree.individuals.at(id);
        climb_pedigree(individual, 0);
    }
    std::vector<double> mean_path_lengths;
    for (const int id : ancestor_ids) {
        Individual<PathLength> *individual = path_pedigree.individuals.at(id);
        mean_path_lengths.push_back(
            (double) individual->data.sum / individual->data.count
        );
    }
    return mean_path_lengths;
}

// Returns the maximum path length to a vector of ancestors.
std::vector<int> get_max_ancestor_path_lengths(Pedigree<> &pedigree,
    std::vector<int> ancestor_ids) {
    Pedigree<PathLength> path_pedigree(pedigree);
    for (const int id : ancestor_ids) {
        Individual<PathLength> *individual = path_pedigree.individuals.at(id);
        individual->data.is_ancestor = true;
    }
    std::vector<int> proband_ids = get_proband_ids(pedigree);
    for (const int id : proband_ids) {
        Individual<PathLength> *individual = path_pedigree.individuals.at(id);
        climb_pedigree(individual, 0);
    }
    std::vector<int> max_path_lengths;
    for (const int id : ancestor_ids) {
        max_path_lengths.push_back(
            path_pedigree.individuals.at(id)->data.max
        );
    }
    return max_path_lengths;
}

// Returns the number of children of a vector of individuals
std::vector<int> get_number_of_children(Pedigree<> &pedigree,
    std::vector<int> individual_ids) {
    std::vector<int> number_of_children;
    for (int id : individual_ids) {
        Individual<> *individual = pedigree.individuals.at(id);
        number_of_children.push_back(individual->children.size());
    }
    return number_of_children;
}

// Recursively adds the completeness of an individual's pedigree
void add_individual_completeness(Individual<> *individual, int depth,
    std::vector<int> &completeness) {
    if (individual->father || individual->mother) {
        if ((int) completeness.size() < depth + 2) {
            completeness.push_back(0);
        }
    }
    if (individual->father) {
        completeness[depth + 1] += 1;
        add_individual_completeness(
            individual->father, depth + 1, completeness
        );
    }
    if (individual->mother) {
        completeness[depth + 1] += 1;
        add_individual_completeness(
            individual->mother, depth + 1, completeness
        );
    }
}

// Returns the completeness of a pedigree.
// Each row corresponds to a proband.
// Each column corresponds to a generation (0 is the proband).
Matrix<double> compute_individual_completeness(Pedigree<> &pedigree,
    std::vector<int> proband_ids) {
    if (proband_ids.empty()) {
        proband_ids = get_proband_ids(pedigree);
    }
    Pedigree new_pedigree = extract_pedigree(pedigree, proband_ids);
    int depth = get_pedigree_depth(new_pedigree);
    Matrix<double> completeness = zeros<double>(depth, proband_ids.size());
    #pragma omp parallel for
    for (int i = 0; i < (int) proband_ids.size(); i++) {
        const int id = proband_ids[i];
        Individual<> *individual = pedigree.individuals.at(id);
        std::vector<int> individual_completeness(1, 1);
        add_individual_completeness(
            individual, 0, individual_completeness
        );
        for (int j = 0; j < (int) individual_completeness.size(); j++) {
            completeness[j][i] = (double) individual_completeness[j] / pow(2, j);
        }
    }
    // Multiply by 100 to get percentages
    for (int i = 0; i < depth; i++) {
        for (int j = 0; j < (int) proband_ids.size(); j++) {
            completeness[i][j] *= 100.0;
        }
    }
    return completeness;
}

// Returns the mean completeness of a pedigree per generation.
// Each row corresponds to a generation (0 is the probands').
std::vector<double> compute_mean_completeness(Pedigree<> &pedigree,
    std::vector<int> proband_ids) {
    std::vector<double> mean_completeness;
    if (proband_ids.empty()) {
        proband_ids = get_proband_ids(pedigree);
    }
    Pedigree new_pedigree = extract_pedigree(pedigree, proband_ids);
    int depth = get_pedigree_depth(new_pedigree);
    Matrix<double> completeness = zeros<double>(proband_ids.size(), depth);
    for (int i = 0; i < (int) proband_ids.size(); i++) {
        const int id = proband_ids[i];
        Individual<> *individual = pedigree.individuals.at(id);
        std::vector<int> individual_completeness(1, 1);
        add_individual_completeness(
            individual, 0, individual_completeness
        );
        for (int j = 0; j < (int) individual_completeness.size(); j++) {
            completeness[i][j] = (double) individual_completeness[j] / pow(2, j);
        }
    }
    for (int j = 0; j < depth; j++) {
        double sum = 0;
        for (int i = 0; i < (int) proband_ids.size(); i++) {
            sum += completeness[i][j];
        }
        mean_completeness.push_back(sum / proband_ids.size());
    }
    // Multiply by 100 to get percentages
    for (int i = 0; i < depth; i++) {
        mean_completeness[i] *= 100.0;
    }
    return mean_completeness;
}

// Recursively adds the implex of an individual's pedigree
void add_individual_implex(Individual<> *individual, int depth,
    std::vector<std::set<int>> &implex) {
    if (individual->father || individual->mother) {
        if ((int) implex.size() < depth + 2) {
            implex.push_back(std::set<int>());
        }
    }
    if (individual->father) {
        implex[depth + 1].insert(individual->father->id);
        add_individual_implex(
            individual->father, depth + 1, implex
        );
    }
    if (individual->mother) {
        implex[depth + 1].insert(individual->mother->id);
        add_individual_implex(
            individual->mother, depth + 1, implex
        );
    }
}

// Returns the implex of a pedigree.
// Each row corresponds to a proband.
// Each column corresponds to a generation (0 is the proband).
Matrix<double> compute_individual_implex(Pedigree<> &pedigree,
    std::vector<int> proband_ids, bool only_new_ancestors) {
    if (proband_ids.empty()) {
        proband_ids = get_proband_ids(pedigree);
    }
    Pedigree new_pedigree = extract_pedigree(pedigree, proband_ids);
    int depth = get_pedigree_depth(new_pedigree);
    Matrix<double> implex(proband_ids.size(), depth);
    #pragma omp parallel for
    for (int i = 0; i < (int) proband_ids.size(); i++) {
        const int id = proband_ids[i];
        Individual<> *individual = pedigree.individuals.at(id);
        std::set<int> set;
        set.insert(id);
        std::vector<std::set<int>> individual_implex(1, set);
        add_individual_implex(individual, 0, individual_implex);
        if (only_new_ancestors) {
            for (int j = 0; j < (int) individual_implex.size(); j++) {
                for (int k = j + 1; k < (int) individual_implex.size(); k++) {
                    for (int ancestor_id : individual_implex[j]) {
                        individual_implex[k].erase(ancestor_id);
                    }
                }
            }
        }
        for (int j = 0; j < (int) individual_implex.size(); j++) {
            implex[i][j] = (double) individual_implex[j].size() /
                pow(2, j) * 100.0;
        }
    }
    return implex;
}

// Returns the mean implex of a pedigree per generation.
// Each row corresponds to a generation (0 is the probands').
std::vector<double> compute_mean_implex(Pedigree<> &pedigree,
    std::vector<int> proband_ids, bool only_new_ancestors) {
    if (proband_ids.empty()) {
        proband_ids = get_proband_ids(pedigree);
    }
    Pedigree new_pedigree = extract_pedigree(pedigree, proband_ids);
    int depth = get_pedigree_depth(new_pedigree);
    std::vector<double> mean_implex;
    Matrix<double> implex(proband_ids.size(), depth);
    for (int i = 0; i < (int) proband_ids.size(); i++) {
        const int id = proband_ids[i];
        Individual<> *individual = pedigree.individuals.at(id);
        std::set<int> set;
        set.insert(id);
        std::vector<std::set<int>> individual_implex(1, set);
        add_individual_implex(individual, 0, individual_implex);
        if (only_new_ancestors) {
            for (int j = 0; j < (int) individual_implex.size(); j++) {
                for (int k = j + 1; k < (int) individual_implex.size(); k++) {
                    for (int ancestor_id : individual_implex[j]) {
                        individual_implex[k].erase(ancestor_id);
                    }
                }
            }
        }
        for (int j = 0; j < (int) individual_implex.size(); j++) {
            implex[i][j] = (double) individual_implex[j].size() /
                pow(2, j) * 100.0;
        }
    }
    for (int j = 0; j < depth; j++) {
        double sum = 0;
        for (int i = 0; i < (int) proband_ids.size(); i++) {
            sum += implex[i][j];
        }
        mean_implex.push_back(sum / proband_ids.size());
    }
    return mean_implex;
}

// Returns the number of probands that descend from a vector of ancestors.
std::vector<int> get_number_of_descendants(Pedigree<> &pedigree,
    std::vector<int> ancestor_ids, std::vector<int> proband_ids) {
    if (ancestor_ids.empty()) {
        ancestor_ids = get_founder_ids(pedigree);
    }
    if (proband_ids.empty()) {
        proband_ids = get_proband_ids(pedigree);
    }
    std::vector<int> number_of_descendants;
    std::set<int> set1;
    for (const int id : proband_ids) {
        set1.insert(id);
    }
    for (const int id : ancestor_ids) {
        std::vector<int> descendant_ids = get_descendant_ids(
            pedigree, std::vector<int>(1, id)
        );
        std::set<int> set2;
        std::vector<int> intersection_result;
        for (const int id : descendant_ids) {
            set2.insert(id);
        }
        set_intersection(
            set1.begin(), set1.end(),
            set2.begin(), set2.end(),
            std::back_inserter(intersection_result)
        );
        number_of_descendants.push_back(intersection_result.size());
    }
    return number_of_descendants;
}

// Add the count of an individual to the count matrix.
void add_count(Individual<Count> *individual) {
    if (individual->data.is_ancestor) {
        individual->data.count += 1;
    }
    if (individual->father) {
        add_count(individual->father);
    }
    if (individual->mother) {
        add_count(individual->mother);
    }
}

// Count the occurrences of a vector of ancestors in the probands' pedigrees.
// Each row corresponds to an ancestor.
// Each column corresponds to the number of occurrences for a given proband.
Matrix<int> count_individual_occurrences(Pedigree<> &pedigree,
    std::vector<int> ancestor_ids, std::vector<int> proband_ids) {
    if (ancestor_ids.empty()) {
        ancestor_ids = get_founder_ids(pedigree);
    }
    if (proband_ids.empty()) {
        proband_ids = get_proband_ids(pedigree);
    }
    Matrix<int> occurrences = zeros<int>(
        ancestor_ids.size(), proband_ids.size());
    Pedigree new_pedigree = extract_pedigree(
        pedigree, proband_ids, ancestor_ids
    );
    Pedigree<Count> count_pedigree(new_pedigree);
    for (const int id : ancestor_ids) {
        Individual<Count> *individual = count_pedigree.individuals.at(id);
        individual->data.is_ancestor = true;
    }
    for (int j = 0; j < (int) proband_ids.size(); j++) {
        const int proband_id = proband_ids[j];
        Individual<Count> *proband = count_pedigree.individuals.at(proband_id);
        add_count(proband);
        for (int i = 0; i < (int) ancestor_ids.size(); i++) {
            const int ancestor_id = ancestor_ids[i];
            Individual<Count> *ancestor = count_pedigree.
                individuals.at(ancestor_id);
            occurrences[i][j] = ancestor->data.count;
            ancestor->data.count = 0;
        }
    }
    return occurrences;
}

// Count the total occurrences of ancestors in the probands' pedigrees.
std::vector<int> count_total_occurrences(Pedigree<> &pedigree,
    std::vector<int> ancestor_ids, std::vector<int> proband_ids) {
    std::vector<int> total_occurrences;
    if (ancestor_ids.empty()) {
        ancestor_ids = get_founder_ids(pedigree);
    }
    if (proband_ids.empty()) {
        proband_ids = get_proband_ids(pedigree);
    }
    Matrix<int> occurrences(ancestor_ids.size(), proband_ids.size());
    Pedigree new_pedigree = extract_pedigree(
        pedigree, proband_ids, ancestor_ids
    );
    Pedigree<Count> count_pedigree(new_pedigree);
    for (const int id : ancestor_ids) {
        Individual<Count> *individual = count_pedigree.individuals.at(id);
        individual->data.is_ancestor = true;
    }
    for (int j = 0; j < (int) proband_ids.size(); j++) {
        const int proband_id = proband_ids[j];
        Individual<Count> *proband = count_pedigree.individuals.at(proband_id);
        add_count(proband);
        for (int i = 0; i < (int) ancestor_ids.size(); i++) {
            const int ancestor_id = ancestor_ids[i];
            Individual<Count> *ancestor = count_pedigree.
                individuals.at(ancestor_id);
            occurrences[i][j] = ancestor->data.count;
            ancestor->data.count = 0;
        }
    }
    for (int i = 0; i < (int) ancestor_ids.size(); i++) {
        int total_occurrence = 0;
        for (int j = 0; j < (int) proband_ids.size(); j++) {
            total_occurrence += occurrences[i][j];
        }
        total_occurrences.push_back(total_occurrence);
    }
    return total_occurrences;
}

// Return all paths from an individual to their ancestors.
std::vector<std::vector<int>> get_ancestor_paths(Individual<> *individual) {
    std::vector<std::vector<int>> all_paths(
        1, std::vector<int>(1, individual->id));
    if (individual->father) {
        std::vector<std::vector<int>> father_paths = get_ancestor_paths(
            individual->father
        );
        for (std::vector<int> path : father_paths) {
            path.push_back(individual->id);
            all_paths.push_back(path);
        }
    }
    if (individual->mother) {
        std::vector<std::vector<int>> mother_paths = get_ancestor_paths(
            individual->mother
        );
        for (std::vector<int> path : mother_paths) {
            path.push_back(individual->id);
            all_paths.push_back(path);
        }
    }
    return all_paths;
}

// Returns the lengths of the paths from an individual to a given ancestor.
std::vector<int> get_ancestor_path_lengths(Pedigree<> &pedigree,
    int proband_id, int ancestor_id) {
    Individual<> *individual = pedigree.individuals.at(proband_id);
    std::vector<std::vector<int>> all_paths = get_ancestor_paths(individual);
    std::vector<int> path_lengths;
    for (std::vector<int> path : all_paths) {
        if (path.front() == ancestor_id) {
            path_lengths.push_back((int) path.size() - 1);
        }
    }
    return path_lengths;
}

// Returns the minimum path length from an individual to a given ancestor.
int get_min_ancestor_path_length(Pedigree<> &pedigree,
    int proband_id, int ancestor_id) {
    std::vector<int> path_lengths = get_ancestor_path_lengths(
        pedigree, proband_id, ancestor_id
    );
    if (path_lengths.empty()) {
        return -1;
    }
    return *std::min_element(path_lengths.begin(), path_lengths.end());
}

// Returns the minimum distance between two probands and a common ancestor.
int get_min_common_ancestor_path_length(Pedigree<> &pedigree,
    int proband1_id, int proband2_id, int common_ancestor_id) {
    int min_path_length1 = get_min_ancestor_path_length(
        pedigree, proband1_id, common_ancestor_id
    );
    int min_path_length2 = get_min_ancestor_path_length(
        pedigree, proband2_id, common_ancestor_id
    );
    return min_path_length1 + min_path_length2;
}

// Returns the number of meioses between probands and MRCAs.
Matrix<int> get_mrca_meioses(Pedigree<> &pedigree,
    std::vector<int> proband_ids, std::vector<int> ancestor_ids) {
    if (ancestor_ids.empty()) {
        ancestor_ids = get_mrca_ids(pedigree, proband_ids);
    }
    Matrix<int> meioses_matrix(proband_ids.size(), ancestor_ids.size());
    #pragma omp parallel for
    for (int i = 0; i < (int) proband_ids.size(); i++) {
        #pragma omp parallel for
        for (int j = 0; j < (int) ancestor_ids.size(); j++) {
            const int proband_id = proband_ids[i];
            const int ancestor_id = ancestor_ids[j];
            int min_path_length = get_min_ancestor_path_length(
                pedigree, proband_id, ancestor_id
            );
            meioses_matrix[i][j] = min_path_length;
        }
    }
    return meioses_matrix;
}

// Returns the number of probands that descend from a vector of ancestors.
std::vector<int> count_coverage(Pedigree<> &pedigree,
    std::vector<int> proband_ids, std::vector<int> ancestor_ids) {
    if (proband_ids.empty()) {
        proband_ids = get_proband_ids(pedigree);
    }
    if (ancestor_ids.empty()) {
        ancestor_ids = get_founder_ids(pedigree);
    }
    Pedigree<Count> count_pedigree(pedigree);
    std::vector<Individual<Count> *> probands;
    for (const int id : proband_ids) {
        Individual<Count> *individual = count_pedigree.individuals.at(id);
        probands.push_back(individual);
    }
    std::vector<int> coverage(ancestor_ids.size(), 0);
    for (int i = 0; i < (int) ancestor_ids.size(); i++) {
        const int ancestor_id = ancestor_ids[i];
        Individual<Count> *ancestor = count_pedigree.individuals.at(ancestor_id);
        std::deque<Individual<Count> *> queue(1, ancestor);
        while (!queue.empty()) {
            Individual<Count> *individual = queue.front();
            queue.pop_front();
            individual->data.is_ancestor = true;
            for (Individual<Count> *child : individual->children) {
                queue.push_back(child);
            }
        }
        for (Individual<Count> *proband : probands) {
            if (proband->data.is_ancestor) {
                coverage[i] += 1;
                proband->data.is_ancestor = false;
            }
        }
    }
    return coverage;
}