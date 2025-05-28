#ifndef EXTRACT_H
#define EXTRACT_H

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

#include "identify.hpp"

// Mark the descendants of an individual.
void mark_descendants(Individual<Status> *individual);

// Mark the ancestors of an individual.
void mark_ancestors(Individual<Status> *individual);

// Recursively mark the paternal lineage of an individual.
void mark_paternal_lineage(Individual<Status> *individual);

// Recursively mark the maternal lineage of an individual.
void mark_maternal_lineage(Individual<Status> *individual);

// Return the status of an individual.
bool both_ancestor_and_descendant(Individual<Status> *individual);
bool at_least_ancestor(Individual<Status> *individual);
bool at_least_descendant(Individual<Status> *individual);

// Extracts the relevant individuals from the pedigree.
Pedigree<> extract_pedigree(Pedigree<> &pedigree,
    std::vector<int> proband_ids = {}, std::vector<int> ancestor_ids = {});

// Extracts the lineages of individuals.
Pedigree<> extract_lineages(Pedigree<> &pedigree,
    std::vector<int> proband_ids = {}, bool maternal = true);

#endif