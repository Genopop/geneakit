#ifndef IDENTIFY_H
#define IDENTIFY_H

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

#include <vector>
#include <queue>
#include <set>
#include <algorithm>
#include "pedigree.hpp"
#include "matrix.hpp"

// Returns the founder IDs of a pedigree.
std::vector<int> get_founder_ids(Pedigree<> &pedigree);

// Returns the half founder IDs of a pedigree.
std::vector<int> get_half_founder_ids(Pedigree<> &pedigree);

// Returns the proband IDs of a pedigree.
std::vector<int> get_proband_ids(Pedigree<> &pedigree);

// Returns the IDs of the fathers of a vector of individuals.
std::vector<int> get_father_ids(Pedigree<> &pedigree, std::vector<int> ids);

// Returns the IDs of the mothers of a vector of individuals.
std::vector<int> get_mother_ids(Pedigree<> &pedigree, std::vector<int> ids);

// Returns the IDs of the siblings of a vector of individuals.
std::vector<int> get_sibling_ids(
    Pedigree<> &pedigree, std::vector<int> ids,
    bool include_half_siblings = true
);

// Returns the IDs of the children of a vector of individuals.
std::vector<int> get_children_ids(Pedigree<> &pedigree, std::vector<int> ids);

// Returns the IDs of the ancestors of a vector of individuals.
std::vector<int> get_ancestor_ids(Pedigree<> &pedigree, std::vector<int> ids);

// Returns the IDs of all occurrences of ancestors of a vector of individuals.
std::vector<int> get_all_ancestor_ids(
    Pedigree<> &pedigree, std::vector<int> ids
);

// Returns the IDs of the descendants of a vector of individuals.
std::vector<int> get_descendant_ids(Pedigree<> &pedigree, std::vector<int> ids);

// Returns the IDs of all occurrences of descendants of a vector of individuals.
std::vector<int> get_all_descendant_ids(
    Pedigree<> &pedigree, std::vector<int> ids
);

// Returns the IDs of the common ancestors of a vector of individuals.
std::vector<int> get_common_ancestor_ids(
    Pedigree<> &pedigree, std::vector<int> ids);

// Returns the IDs of the common founders of a vector of individuals.
std::vector<int> get_common_founder_ids(
    Pedigree<> &pedigree, std::vector<int> ids);

// Returns the IDs of the most recent common ancestors of a vector of IDs.
std::vector<int> get_mrca_ids(Pedigree<> &pedigree, std::vector<int> ids);

#endif