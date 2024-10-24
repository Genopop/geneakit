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

#include "../include/identify.hpp"

// Returns the founder IDs of a pedigree.
std::vector<int> get_founder_ids(Pedigree<> &pedigree) {
    std::vector<int> founder_ids;
    for (auto const& [id, individual] : pedigree.individuals) {
        if (!individual->father && !individual->mother) {
            founder_ids.push_back(id);
        }
    }
    // Sort the founders by ID
    sort(founder_ids.begin(), founder_ids.end());
    return founder_ids;
}

// Returns the half founder IDs of a pedigree.
std::vector<int> get_half_founder_ids(Pedigree<> &pedigree) {
    std::vector<int> half_founder_ids;
    for (auto const& [id, individual] : pedigree.individuals) {
        if ((!individual->father && individual->mother) ||
            (individual->father && !individual->mother)) {
            half_founder_ids.push_back(id);
        }
    }
    // Sort the half founders by ID
    sort(half_founder_ids.begin(), half_founder_ids.end());
    return half_founder_ids;
}

// Returns the proband IDs of a pedigree.
std::vector<int> get_proband_ids(Pedigree<> &pedigree) {
    std::vector<int> proband_ids;
    for (auto const& [id, individual] : pedigree.individuals) {
        if (!individual->children.size()) {
            proband_ids.push_back(id);
        }
    }
    // Sort the probands by ID
    sort(proband_ids.begin(), proband_ids.end());
    return proband_ids;
}

// Returns the IDs of the fathers of a vector of individuals.
std::vector<int> get_father_ids(
    Pedigree<> &pedigree, std::vector<int> ids
) {
    std::vector<int> father_ids;
    std::set<int> set;
    for (const int id : ids) {
        Individual<> *individual = pedigree.individuals.at(id);
        if (individual->father) {
            set.insert(individual->father->id);
        }
    }
    for (const int id : set) {
        father_ids.push_back(id);
    }
    // Sort the fathers by ID
    sort(father_ids.begin(), father_ids.end());
    return father_ids;
}

// Returns the IDs of the mothers of a vector of individuals.
std::vector<int> get_mother_ids(
    Pedigree<> &pedigree, std::vector<int> ids
) {
    std::vector<int> mother_ids;
    std::set<int> set;
    for (const int id : ids) {
        Individual<> *individual = pedigree.individuals.at(id);
        if (individual->mother) {
            set.insert(individual->mother->id);
        }
    }
    for (const int id : set) {
        mother_ids.push_back(id);
    }
    // Sort the mothers by ID
    sort(mother_ids.begin(), mother_ids.end());
    return mother_ids;
}

// Returns the IDs of the siblings of a vector of individuals.
std::vector<int> get_sibling_ids(Pedigree<> &pedigree,
    std::vector<int> ids, bool include_half_siblings) {
    std::vector<int> sibling_ids;
    std::set<int> set1, set2, result;
    for (const int id : ids) {
        set2.insert(id);
        std::set<int> fathers_children, mothers_children;
        Individual<> *individual = pedigree.individuals.at(id);
        if (individual->father) {
            for (Individual<> *child : individual->father->children) {
                fathers_children.insert(child->id);
            }
        }
        if (individual->mother) {
            for (Individual<> *child : individual->mother->children) {
                mothers_children.insert(child->id);
            }
        }
        if (include_half_siblings) {
            set_union(
                fathers_children.begin(), fathers_children.end(),
                mothers_children.begin(), mothers_children.end(),
                std::inserter(set1, set1.begin())
            );
        } else {
            set_intersection(
                fathers_children.begin(), fathers_children.end(),
                mothers_children.begin(), mothers_children.end(),
                std::inserter(set2, set2.begin())
            );
        }
    }
    set_difference(
        set1.begin(), set1.end(),
        set2.begin(), set2.end(),
        std::inserter(result, result.begin())
    );
    for (const int id : result) {
        sibling_ids.push_back(id);
    }
    // Sort the siblings by ID
    sort(sibling_ids.begin(), sibling_ids.end());
    return sibling_ids;
}

// Returns the IDs of the children of a vector of individuals.
std::vector<int> get_children_ids(Pedigree<> &pedigree,
    std::vector<int> ids) {
    std::vector<int> children_ids;
    std::unordered_set<int> set;
    for (int id : ids) {
        Individual<> *individual = pedigree.individuals.at(id);
        for (Individual<> *child : individual->children) {
            set.insert(child->id);
        }
    }
    for (int id : set) {
        children_ids.push_back(id);
    }
    // Sort the children by ID
    sort(children_ids.begin(), children_ids.end());
    return children_ids;
}

// Returns the IDs of the ancestors of a vector of individuals.
std::vector<int> get_ancestor_ids(Pedigree<> &pedigree,
    std::vector<int> ids) {
    std::vector<int> ancestor_ids;
    std::unordered_set<int> set;
    std::deque<int> queue;
    for (const int id : ids) {
        queue.push_back(id);
    }
    while (!queue.empty()) {
        const int id = queue.front();
        queue.pop_front();
        Individual<> *individual = pedigree.individuals.at(id);
        if (individual->father) {
            set.insert(individual->father->id);
            queue.push_back(individual->father->id);
        }
        if (individual->mother) {
            set.insert(individual->mother->id);
            queue.push_back(individual->mother->id);
        }
    }
    for (const int id : set) {
        ancestor_ids.push_back(id);
    }
    // Sort the ancestors by ID
    sort(ancestor_ids.begin(), ancestor_ids.end());
    return ancestor_ids;
}

// Returns the IDs of all occurrences of ancestors of a vector of individuals.
std::vector<int> get_all_ancestor_ids(Pedigree<> &pedigree,
    std::vector<int> ids) {
    std::vector<int> ancestor_ids;
    std::deque<int> queue;
    for (const int id : ids) {
        queue.push_back(id);
    }
    while (!queue.empty()) {
        const int id = queue.front();
        queue.pop_front();
        Individual<> *individual = pedigree.individuals.at(id);
        if (individual->father) {
            ancestor_ids.push_back(individual->father->id);
            queue.push_back(individual->father->id);
        }
        if (individual->mother) {
            ancestor_ids.push_back(individual->mother->id);
            queue.push_back(individual->mother->id);
        }
    }
    // Sort the ancestors by ID
    sort(ancestor_ids.begin(), ancestor_ids.end());
    return ancestor_ids;
}

// Returns the IDs of the descendants of a vector of individuals.
std::vector<int> get_descendant_ids(Pedigree<> &pedigree,
    std::vector<int> ids) {
    std::vector<int> descendant_ids;
    std::unordered_set<int> set;
    std::deque<int> queue;
    for (const int id : ids) {
        queue.push_back(id);
    }
    while (!queue.empty()) {
        const int id = queue.front();
        queue.pop_front();
        Individual<> *individual = pedigree.individuals.at(id);
        for (Individual<> *child : individual->children) {
            set.insert(child->id);
            queue.push_back(child->id);
        }
    }
    for (const int id : set) {
        descendant_ids.push_back(id);
    }
    // Sort the descendants by ID
    sort(descendant_ids.begin(), descendant_ids.end());
    return descendant_ids;
}

// Returns the IDs of all occurrences of descendants of a vector of individuals.
std::vector<int> get_all_descendant_ids(Pedigree<> &pedigree,
    std::vector<int> ids) {
    std::vector<int> descendant_ids;
    std::deque<int> queue;
    for (const int id : ids) {
        queue.push_back(id);
    }
    while (!queue.empty()) {
        const int id = queue.front();
        queue.pop_front();
        Individual<> *individual = pedigree.individuals.at(id);
        for (Individual<> *child : individual->children) {
            descendant_ids.push_back(child->id);
            queue.push_back(child->id);
        }
    }
    // Sort the descendants by ID
    sort(descendant_ids.begin(), descendant_ids.end());
    return descendant_ids;
}

// Returns the IDs of the common ancestors of a vector of individuals.
std::vector<int> get_common_ancestor_ids(Pedigree<> &pedigree,
    std::vector<int> ids) {
    std::vector<int> common_ancestor_ids = get_ancestor_ids(pedigree, ids);
    std::set<int> set1, set2;
    std::vector<int> intersection_result;
    for (int id : common_ancestor_ids) {
        set1.insert(id);
    }
    common_ancestor_ids.clear();
    for (int id : ids) {
        std::vector<int> ancestor_ids = get_ancestor_ids(
            pedigree, std::vector<int>(1, id)
        );
        for (int ancestor_id : ancestor_ids) {
            set2.insert(ancestor_id);
        }
        set_intersection(
            set1.begin(), set1.end(),
            set2.begin(), set2.end(),
            std::back_inserter(intersection_result)
        );
        set1.clear();
        for (const int id : intersection_result) {
            set1.insert(id);
        }
        set2.clear();
        intersection_result.clear();
    }
    for (int id : set1) {
        common_ancestor_ids.push_back(id);
    }
    // Sort the common ancestors by ID
    sort(common_ancestor_ids.begin(), common_ancestor_ids.end());
    return common_ancestor_ids;
}

// Returns the IDs of the common founders of a vector of individuals.
std::vector<int> get_common_founder_ids(Pedigree<> &pedigree,
    std::vector<int> ids) {
    std::vector<int> common_ancestor_ids =
        get_common_ancestor_ids(pedigree, ids);
    std::set<int> set1(common_ancestor_ids.begin(), common_ancestor_ids.end());
    std::vector<int> founder_ids = get_founder_ids(pedigree);
    std::set<int> set2(founder_ids.begin(), founder_ids.end());
    std::set<int> intersection_result;
    set_intersection(
        set2.begin(), set2.end(),
        set1.begin(), set1.end(),
        std::inserter(intersection_result, intersection_result.begin())
    );
    std::vector<int> common_founder_ids(
        intersection_result.begin(),
        intersection_result.end()
    );
    // Sort the common founders by ID
    sort(common_founder_ids.begin(), common_founder_ids.end());
    return common_founder_ids;
}

// Returns the IDs of the most recent common ancestors of a vector of individuals.
std::vector<int> get_mrca_ids(Pedigree<> &pedigree,
    std::vector<int> ids) {
    std::vector<int> most_recent_common_ancestor_ids;
    std::vector<int> common_ancestor_ids =
        get_common_ancestor_ids(pedigree, ids);
    std::vector<int> ancestors_of_common_ancestors =
        get_ancestor_ids(pedigree, common_ancestor_ids);
    std::set<int> set1(common_ancestor_ids.begin(), common_ancestor_ids.end());
    std::set<int> set2(
        ancestors_of_common_ancestors.begin(),
        ancestors_of_common_ancestors.end()
    );
    set_difference(
        set1.begin(), set1.end(),
        set2.begin(), set2.end(),
        std::back_inserter(most_recent_common_ancestor_ids)
    );
    // Sort the most recent common ancestors by ID
    sort(
        most_recent_common_ancestor_ids.begin(), 
        most_recent_common_ancestor_ids.end()
    );
    return most_recent_common_ancestor_ids;
}