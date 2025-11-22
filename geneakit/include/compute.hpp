#ifndef COMPUTE_H
#define COMPUTE_H

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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <math.h>
#include <omp.h>
#include <queue>
#include <set>
#include <tuple>
#include <vector>
#include <parallel_hashmap/phmap.h>
#include "identify.hpp"
#include "extract.hpp"
#include "matrix.hpp"

// Returns the sparse kinship matrix
std::tuple<std::vector<float>, std::vector<int>, std::vector<int64_t>>
compute_kinships_sparse(Pedigree<> &pedigree, std::vector<int> proband_ids, bool verbose);

// Returns the previous generation of a set of individuals.
phmap::flat_hash_set<int> get_previous_generation(Pedigree<> &pedigree,
    phmap::flat_hash_set<int> &ids);

// Go from the bottom to the top of the pedigree
std::vector<phmap::flat_hash_set<int>> get_generations(
    Pedigree<> &pedigree, std::vector<int> &proband_ids);

// Drag the individuals up
std::vector<phmap::flat_hash_set<int>> copy_bottom_up(
    std::vector<phmap::flat_hash_set<int>> &generations);

// Drag the individuals down
std::vector<phmap::flat_hash_set<int>> copy_top_down(
    std::vector<phmap::flat_hash_set<int>> &generations);

// Find the intersection of the two sets
std::vector<std::vector<int>> intersect_both_directions(
    std::vector<phmap::flat_hash_set<int>> &bottom_up,
    std::vector<phmap::flat_hash_set<int>> &top_down);

// Separate the individuals into generations where individuals who appear
// in two non-contiguous generations are dragged in the generations in-between.
// Based on the recursive-cut algorithm from Kirkpatrick et al.
std::vector<std::vector<int>> cut_vertices(
    Pedigree<> &pedigree, std::vector<int> &proband_ids);

// Returns the kinship coefficient between two individuals.
// A modified version of the recursive algorithm from Karigl.
double compute_kinship(
    const Individual<Index> *individual1,
    const Individual<Index> *individual2,
    Matrix<double> &founder_matrix);

// Compute the kinship coefficients with oneself
void compute_kinship_with_oneself(
    std::vector<Individual<Index> *> &vertex_cut,
    Matrix<double> &founder_matrix,
    Matrix<double> &proband_matrix);

// Compute the kinship coefficients between the individuals
void compute_kinship_between_probands(
    std::vector<Individual<Index> *> &vertex_cut,
    Matrix<double> &founder_matrix,
    Matrix<double> &proband_matrix);

// Returns the required memory for kinship calculations.
double get_required_memory_for_kinships(
    Pedigree<> &pedigree, std::vector<int> proband_ids = {});

// Returns the kinship matrix using the algorithm from Morin et al.
Matrix<double> compute_kinships(
    Pedigree<> &pedigree, std::vector<int> proband_ids = {},
    bool verbose = false);

// Returns the mean kinship coefficient of a kinship matrix.
double compute_mean_kinship(Matrix<double> &kinship_matrix);

// Returns the inbreeding coefficients of a vector of individuals.
// Copied from the article by M Sargolzaei, H Iwaisaki & J-J Colleau (2005).
std::vector<double> compute_inbreedings(Pedigree<> &pedigree,
    std::vector<int> proband_ids);

// Adds the contribution of an individual.
void add_contribution(const Individual<Contribution> *individual,
    const int depth);

// Returns the genetic contributions of a pedigree.
// Each row corresponds to a proband.
// Each column corresponds to an ancestor.
Matrix<double> compute_genetic_contributions(Pedigree<> &pedigree,
    std::vector<int> proband_ids = {}, std::vector<int> ancestor_ids = {});

#endif