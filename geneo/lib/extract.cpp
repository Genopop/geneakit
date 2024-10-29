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

#include "../include/extract.hpp"

// Mark the descendants of an individual.
void mark_descendants(Individual<Status> *individual) {
    if (!individual->data.is_descendant) {
        individual->data.is_descendant = true;
        for (Individual<Status> *child : individual->children) {
            mark_descendants(child);
        }
    }
}

// Mark the ancestors of an individual.
void mark_ancestors(Individual<Status> *individual) {
    if (!individual->data.is_ancestor) {
        individual->data.is_ancestor = true;
        if (individual->father) {
            mark_ancestors(individual->father);
        }
        if (individual->mother) {
            mark_ancestors(individual->mother);
        }
    }
}

// Recursively mark the paternal lineage of an individual.
void mark_paternal_lineage(Individual<Status> *individual) {
    if (!individual->data.is_ancestor) {
        individual->data.is_ancestor = true;
        if (individual->father) {
            mark_paternal_lineage(individual->father);
        }
    }
}

// Recursively mark the maternal lineage of an individual.
void mark_maternal_lineage(Individual<Status> *individual) {
    if (!individual->data.is_ancestor) {
        individual->data.is_ancestor = true;
        if (individual->mother) {
            mark_maternal_lineage(individual->mother);
        }
    }
}

// Return the status of an individual.
bool both_ancestor_and_descendant(Individual<Status> *individual) {
    return individual->data.is_ancestor && individual->data.is_descendant;
}
bool at_least_ancestor(Individual<Status> *individual) {
    return individual->data.is_ancestor;
}
bool at_least_descendant(Individual<Status> *individual) {
    return individual->data.is_descendant;
}

// Extracts the relevant individuals from the pedigree.
Pedigree<> extract_pedigree(Pedigree<> &pedigree,
    std::vector<int> proband_ids, std::vector<int> ancestor_ids) {
    // Initialize the pedigree
    Pedigree new_pedigree;
    // Create a pedigree for extraction
    Pedigree<Status> extract_pedigree(pedigree);
    // Mark the probands
    for (const int id : proband_ids) {
        Individual<Status> *proband =
            extract_pedigree.individuals.at(id);
        mark_ancestors(proband);
    }
    // Mark the ancestors
    for (const int id : ancestor_ids) {
        Individual<Status> *ancestor =
            extract_pedigree.individuals.at(id);
        mark_descendants(ancestor);
    }
    // Add the relevant individuals to the new pedigree
    std::vector<int> ids;
    phmap::flat_hash_map<int, Individual<> *> individuals;
    int rank = 0;
    bool (*condition)(Individual<Status> *);
    if (!proband_ids.empty() && !ancestor_ids.empty()) {
        condition = both_ancestor_and_descendant;
    } else if (!proband_ids.empty()) {
        condition = at_least_ancestor;
    } else if (!ancestor_ids.empty()) {
        condition = at_least_descendant;
    } else {
        return pedigree;
    }
    for (const int id : extract_pedigree.ids) {
        Individual<Status> *individual = extract_pedigree.individuals.at(id);
        if (condition(individual)) {
            Individual<> *father = nullptr;
            if (individual->father && condition(individual->father)) {
                father = individuals.at(individual->father->id);
            }
            Individual<> *mother = nullptr;
            if (individual->mother && condition(individual->mother)) {
                mother = individuals.at(individual->mother->id);
            }
            Individual<> *new_individual = new Individual(
                rank++, individual->id, father, mother, individual->sex
            );
            individuals.emplace(id, new_individual);
            ids.push_back(id);
        }
    }
    // Add the individuals to the new pedigree
    new_pedigree.ids = ids;
    new_pedigree.individuals = individuals;
    // Return the new pedigree
    return new_pedigree;
}

// Extracts the lineages of individuals.
Pedigree<> extract_lineages(Pedigree<> &pedigree,
    std::vector<int> proband_ids, bool maternal) {
    // Initialize the pedigree
    Pedigree new_pedigree;
    if (proband_ids.empty()) {
        proband_ids = get_proband_ids(pedigree);
    }
    // Create a pedigree for extraction
    Pedigree<Status> extract_pedigree(pedigree);
    // Mark the probands
    for (const int id : proband_ids) {
        Individual<Status> *proband = extract_pedigree.individuals.at(id);
        if (maternal) {
            mark_maternal_lineage(proband);
        } else {
            mark_paternal_lineage(proband);
        }
    }
    // Add the relevant individuals to the new pedigree
    std::vector<int> ids;
    phmap::flat_hash_map<int, Individual<> *> individuals;
    int rank = 0;
    for (const int id : extract_pedigree.ids) {
        Individual<Status> *individual = extract_pedigree.individuals.at(id);
        if (individual->data.is_ancestor) {
            Individual<> *father = nullptr;
            if (individual->father) {
                if (individual->father->data.is_ancestor) {
                    father = individuals.at(individual->father->id);
                }
            }
            Individual<> *mother = nullptr;
            if (individual->mother) {
                if (individual->mother->data.is_ancestor) {
                    mother = individuals.at(individual->mother->id);
                }
            }
            Individual<> *new_individual = new Individual(
                rank++, individual->id, father, mother, individual->sex
            );
            individuals.emplace(id, new_individual);
            if (father) {
                father->children.push_back(individuals.at(id));
            }
            if (mother) {
                mother->children.push_back(individuals.at(id));
            }
            ids.push_back(id);
        }
    }
    // Add the individuals to the new pedigree
    new_pedigree.ids = ids;
    new_pedigree.individuals = individuals;
    // Return the new pedigree
    return new_pedigree;
}