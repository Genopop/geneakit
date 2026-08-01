/*------------------------------------------------------------------------------
MIT License

Copyright (c) 2026 Gilles-Philippe Morin

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

#include "../include/edit.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace {

// Removes a single occurrence of `child` from `parent`'s children vector.
// The order of a `children` vector is never relied upon elsewhere, so an
// O(1) swap-and-pop is used instead of an order-preserving erase.
void unlink_child(Individual<> *parent, Individual<> *child) {
    if (!parent) {
        return;
    }
    std::vector<Individual<> *> &children = parent->children;
    auto it = std::find(children.begin(), children.end(), child);
    if (it != children.end()) {
        *it = children.back();
        children.pop_back();
    }
}

// Looks up a parent by ID (0 means "no parent"). Throws std::out_of_range
// if the ID does not resolve to an existing individual.
Individual<> *find_parent(Pedigree<> &pedigree, int parent_id,
    const char *role) {
    if (parent_id == 0) {
        return nullptr;
    }
    auto it = pedigree.individuals.find(parent_id);
    if (it == pedigree.individuals.end()) {
        throw std::out_of_range(
            std::string(role) + " with ID " + std::to_string(parent_id) +
            " does not exist in the pedigree; add parents before their " +
            "children");
    }
    return it->second;
}

}  // namespace

void set_individual(Pedigree<> &pedigree, int id, int father_id,
    int mother_id, int sex) {
    if (id == 0) {
        throw std::invalid_argument(
            "ID 0 is reserved to mean \"no individual\" and cannot be "
            "assigned to an individual");
    }
    if (sex != UNKNOWN && sex != MALE && sex != FEMALE) {
        throw std::invalid_argument(
            "sex must be 0 (unknown), 1 (male) or 2 (female), got " +
            std::to_string(sex));
    }
    if (father_id == id) {
        throw std::invalid_argument(
            "individual " + std::to_string(id) +
            " cannot be their own father");
    }
    if (mother_id == id) {
        throw std::invalid_argument(
            "individual " + std::to_string(id) +
            " cannot be their own mother");
    }
    if (father_id != 0 && father_id == mother_id) {
        throw std::invalid_argument(
            "individual " + std::to_string(father_id) + " cannot be both " +
            "the father and the mother of individual " + std::to_string(id));
    }
    Individual<> *father = find_parent(pedigree, father_id, "father");
    if (father && father->sex == FEMALE) {
        throw std::invalid_argument(
            "individual " + std::to_string(father_id) + " is recorded as " +
            "female and cannot be set as a father");
    }
    Individual<> *mother = find_parent(pedigree, mother_id, "mother");
    if (mother && mother->sex == MALE) {
        throw std::invalid_argument(
            "individual " + std::to_string(mother_id) + " is recorded as " +
            "male and cannot be set as a mother");
    }

    auto it = pedigree.individuals.find(id);
    if (it == pedigree.individuals.end()) {
        // Brand-new individual. A founder (no known parents) is pushed to
        // the front with a rank below the current minimum: nothing
        // constrains it to come after anyone, so this leaves it free to
        // be attached as a parent of any individual already in the
        // pedigree later, without renumbering or moving anyone. An
        // individual with a known parent is appended at the back with a
        // rank past the current maximum, which keeps it after that
        // parent (already confirmed to exist somewhere in the deque).
        // Either way this only touches the new individual: O(1).
        bool is_founder = !father && !mother;
        int rank;
        if (pedigree.ids.empty()) {
            rank = 0;
        } else if (is_founder) {
            rank = pedigree.individuals.at(pedigree.ids.front())->rank - 1;
        } else {
            rank = pedigree.individuals.at(pedigree.ids.back())->rank + 1;
        }
        Individual<> *individual = new Individual<>(
            rank, id, father, mother, (Sex) sex);
        pedigree.individuals.emplace(id, individual);
        if (is_founder) {
            pedigree.ids.push_front(id);
        } else {
            pedigree.ids.push_back(id);
        }
        return;
    }

    // Update an individual already in the pedigree, in place, so that any
    // children already holding a raw pointer to it keep seeing a valid
    // individual.
    Individual<> *individual = it->second;
    if (father && father->rank >= individual->rank) {
        throw std::invalid_argument(
            "individual " + std::to_string(father_id) +
            " was added after individual " + std::to_string(id) +
            "; a parent must have been added before their child");
    }
    if (mother && mother->rank >= individual->rank) {
        throw std::invalid_argument(
            "individual " + std::to_string(mother_id) +
            " was added after individual " + std::to_string(id) +
            "; a parent must have been added before their child");
    }
    if ((Sex) sex != individual->sex) {
        // Changing sex could contradict the role this individual already
        // plays for their existing children.
        for (Individual<> *child : individual->children) {
            if (child->father == individual && sex == FEMALE) {
                throw std::invalid_argument(
                    "individual " + std::to_string(id) + " is the father " +
                    "of individual " + std::to_string(child->id) +
                    " and cannot be marked female");
            }
            if (child->mother == individual && sex == MALE) {
                throw std::invalid_argument(
                    "individual " + std::to_string(id) + " is the mother " +
                    "of individual " + std::to_string(child->id) +
                    " and cannot be marked male");
            }
        }
        individual->sex = (Sex) sex;
    }
    if (father != individual->father) {
        unlink_child(individual->father, individual);
        individual->father = father;
        if (father) {
            father->children.push_back(individual);
        }
    }
    if (mother != individual->mother) {
        unlink_child(individual->mother, individual);
        individual->mother = mother;
        if (mother) {
            mother->children.push_back(individual);
        }
    }
}

void remove_individual(Pedigree<> &pedigree, int id) {
    auto it = pedigree.individuals.find(id);
    if (it == pedigree.individuals.end()) {
        throw std::out_of_range(
            "individual " + std::to_string(id) +
            " does not exist in the pedigree");
    }
    Individual<> *individual = it->second;
    // Detach from the parents' children lists.
    unlink_child(individual->father, individual);
    unlink_child(individual->mother, individual);
    // Orphan any children that pointed to this individual: the side of
    // their lineage that went through it becomes unknown again.
    for (Individual<> *child : individual->children) {
        if (child->father == individual) {
            child->father = nullptr;
        }
        if (child->mother == individual) {
            child->mother = nullptr;
        }
    }
    pedigree.individuals.erase(it);
    auto ids_it = std::find(pedigree.ids.begin(), pedigree.ids.end(), id);
    pedigree.ids.erase(ids_it);
    delete individual;
}
