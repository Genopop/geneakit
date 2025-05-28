#ifndef DESCRIBE_H
#define DESCRIBE_H

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

#include <cmath>
#include "extract.hpp"
#include "identify.hpp"
#include "matrix.hpp"

// Returns the number of individuals in the pedigree
int get_number_of_individuals(Pedigree<> &pedigree);

// Returns the number of men in the pedigree
int get_number_of_men(Pedigree<> &pedigree);

// Returns the number of women in the pedigree
int get_number_of_women(Pedigree<> &pedigree);

// Returns the number of parent-child relationships in the pedigree
int get_number_of_parent_child_relations(Pedigree<> &pedigree);

// Returns the maximum depth of an individual's pedigree
int get_individual_depth(Individual<> *individual);

// Returns the depth of a pedigree
int get_pedigree_depth(Pedigree<> &pedigree);

// Returns the mean depth of an individual's pedigree
double get_mean_individual_depth(Individual<> *individual);

// Returns the mean depths of the pedigree for the specified probands.
std::vector<double> get_mean_pedigree_depths(Pedigree<> &pedigree,
    std::vector<int> proband_ids = {});

// Recursively climb the pedigree to find the ancestors.
void climb_pedigree(Individual<PathLength> *individual, int depth);

// Returns the minimum path length to a vector of ancestors.
std::vector<int> get_min_ancestor_path_lengths(Pedigree<> &pedigree,
    std::vector<int> ancestor_ids);

// Returns the mean path length to a vector of ancestors.
std::vector<double> get_mean_ancestor_path_lengths(Pedigree<> &pedigree,
    std::vector<int> ancestor_ids);

// Returns the maximum path length to a vector of ancestors.
std::vector<int> get_max_ancestor_path_lengths(Pedigree<> &pedigree,
    std::vector<int> ancestor_ids);

// Returns the number of children of a vector of individuals
std::vector<int> get_number_of_children(Pedigree<> &pedigree,
    std::vector<int> individual_ids);

// Recursively adds the completeness of an individual's pedigree
void add_individual_completeness(Individual<> *individual, int depth,
    std::vector<int> &completeness);

// Returns the completeness of a pedigree.
// Each row corresponds to a proband.
// Each column corresponds to a generation (0 is the proband).
Matrix<double> compute_individual_completeness(Pedigree<> &pedigree,
    std::vector<int> proband_ids);

// Returns the mean completeness of a pedigree per generation.
// Each row corresponds to a generation (0 is the probands').
std::vector<double> compute_mean_completeness(Pedigree<> &pedigree,
    std::vector<int> proband_ids);

// Recursively adds the implex of an individual's pedigree
void add_individual_implex(Individual<> *individual, int depth,
    std::vector<std::set<int>> &implex);

// Returns the implex of a pedigree.
// Each row corresponds to a proband.
// Each column corresponds to a generation (0 is the proband).
Matrix<double> compute_individual_implex(Pedigree<> &pedigree,
    std::vector<int> proband_ids, bool only_new_ancestors = false);

// Returns the mean implex of a pedigree per generation.
// Each row corresponds to a generation (0 is the probands').
std::vector<double> compute_mean_implex(Pedigree<> &pedigree,
    std::vector<int> proband_ids, bool only_new_ancestors = false);

// Returns the number of probands that descend from a vector of ancestors.
std::vector<int> get_number_of_descendants(Pedigree<> &pedigree,
    std::vector<int> ancestor_ids, std::vector<int> proband_ids);

// Add the count of an individual to the count matrix.
void add_count(Individual<Count> *individual);

// Count the occurrences of a vector of ancestors in the probands' pedigrees.
// Each row corresponds to an ancestor.
// Each column corresponds to the number of occurrences for a given proband.
Matrix<int> count_individual_occurrences(Pedigree<> &pedigree,
    std::vector<int> ancestor_ids, std::vector<int> proband_ids);

// Count the total occurrences of ancestors in the probands' pedigrees.
std::vector<int> count_total_occurrences(Pedigree<> &pedigree,
    std::vector<int> ancestor_ids, std::vector<int> proband_ids);

// Return all paths from an individual to their ancestors.
std::vector<std::vector<int>> get_ancestor_paths(Individual<> *individual);

// Returns the lengths of the paths from an individual to a given ancestor.
std::vector<int> get_ancestor_path_lengths(Pedigree<> &pedigree,
    int proband_id, int ancestor_id);

// Returns the minimum path length from an individual to a given ancestor.
int get_min_ancestor_path_length(Pedigree<> &pedigree,
    int proband_id, int ancestor_id);

// Returns the minimum distance between two probands and a common ancestor.
int get_min_common_ancestor_path_length(Pedigree<> &pedigree,
    int proband1_id, int proband2_id, int common_ancestor_id);

// Returns the number of meioses between probands and MRCAs.
Matrix<int> get_mrca_meioses(Pedigree<> &pedigree,
    std::vector<int> proband_ids, std::vector<int> ancestor_ids = {});

// Returns the number of probands that descend from a vector of ancestors.
std::vector<int> count_coverage(Pedigree<> &pedigree,
    std::vector<int> proband_ids = {}, std::vector<int> ancestor_ids = {});

#endif