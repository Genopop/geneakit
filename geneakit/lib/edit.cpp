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
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

// Removes a single occurrence of `child` from `parent`'s children vector.
// The erase is order-preserving: `children` is visible from Python and is
// written out in that order, so it must not depend on the history of edits
// that produced the pedigree.
void unlink_child(Individual<> *parent, Individual<> *child) {
    if (!parent) {
        return;
    }
    std::vector<Individual<> *> &children = parent->children;
    auto it = std::find(children.begin(), children.end(), child);
    if (it != children.end()) {
        children.erase(it);
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
            " does not exist in the pedigree");
    }
    return it->second;
}

// Rejects the parts of a record that are wrong on their own, without any
// reference to the rest of the pedigree.
void check_record(int id, int father_id, int mother_id, int sex) {
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
}

// ---------------------------------------------------------------------------
// Incremental topological ordering
//
// The pedigree is a DAG whose vertices carry a rank that is both a valid
// topological numbering and the exact position of the vertex in `ids`.
// Adding the link parent -> child when the parent already comes first costs
// nothing. When it does not, the order has to be repaired, and the whole
// point of the routines below is to repair it without renumbering the
// pedigree: only the individuals that the new link actually constrains are
// moved, and they are moved into the set of positions they already
// occupied, so the numbering stays dense. This is the algorithm of Pearce &
// Kelly (2007).
// ---------------------------------------------------------------------------

// Collects the descendants of `start` that come strictly before `bound`,
// i.e. the individuals that the new link may force to move later. Returns
// false as soon as `target` is reached, which means `target` is a
// descendant of `start` and the requested link would close a cycle.
//
// Pruning at `bound` is safe precisely because ranks increase along every
// link: once a descendant sits at or after `bound`, so does everyone below
// it.
bool collect_descendants(Individual<> *start, int bound,
    const Individual<> *target, std::vector<Individual<> *> &region) {
    std::vector<Individual<> *> stack;
    std::unordered_set<const Individual<> *> seen;
    stack.push_back(start);
    seen.insert(start);
    region.push_back(start);
    while (!stack.empty()) {
        Individual<> *current = stack.back();
        stack.pop_back();
        for (Individual<> *child : current->children) {
            if (child == target) {
                return false;
            }
            if (child->rank >= bound) {
                continue;
            }
            if (seen.insert(child).second) {
                region.push_back(child);
                stack.push_back(child);
            }
        }
    }
    return true;
}

// Collects the ancestors of `start` that come strictly after `bound`, i.e.
// the individuals that the new link may force to move earlier.
void collect_ancestors(Individual<> *start, int bound,
    std::vector<Individual<> *> &region) {
    std::vector<Individual<> *> stack;
    std::unordered_set<const Individual<> *> seen;
    stack.push_back(start);
    seen.insert(start);
    region.push_back(start);
    while (!stack.empty()) {
        Individual<> *current = stack.back();
        stack.pop_back();
        Individual<> *parents[2] = {current->father, current->mother};
        for (Individual<> *parent : parents) {
            if (!parent || parent->rank <= bound) {
                continue;
            }
            if (seen.insert(parent).second) {
                region.push_back(parent);
                stack.push_back(parent);
            }
        }
    }
}

// Rewrites the order of the affected region. `earlier` (the ancestors that
// must end up in front) and `later` (the descendants that must end up
// behind) are placed back into the very positions they already occupied
// between them, keeping their own relative order. Nobody outside the two
// sets moves, and no position is created or destroyed, so `ids` stays a
// permutation of itself and the ranks stay dense.
void reorder_region(Pedigree<> &pedigree,
    std::vector<Individual<> *> &earlier,
    std::vector<Individual<> *> &later) {
    auto by_rank = [] (const Individual<> *a, const Individual<> *b) {
        return a->rank < b->rank;
    };
    std::sort(earlier.begin(), earlier.end(), by_rank);
    std::sort(later.begin(), later.end(), by_rank);
    std::vector<int> positions;
    positions.reserve(earlier.size() + later.size());
    for (const Individual<> *individual : earlier) {
        positions.push_back(individual->rank);
    }
    for (const Individual<> *individual : later) {
        positions.push_back(individual->rank);
    }
    std::sort(positions.begin(), positions.end());
    size_t next = 0;
    for (Individual<> *individual : earlier) {
        individual->rank = positions[next];
        pedigree.ids[positions[next]] = individual->id;
        next++;
    }
    for (Individual<> *individual : later) {
        individual->rank = positions[next];
        pedigree.ids[positions[next]] = individual->id;
        next++;
    }
}

// True if `parent` is a descendant of `child`, in which case making it the
// parent of `child` would close a cycle. The rank invariant makes the
// common case free: an individual that already comes before another cannot
// possibly be one of its descendants.
bool would_create_cycle(Individual<> *parent, Individual<> *child) {
    if (parent->rank < child->rank) {
        return false;
    }
    std::vector<Individual<> *> region;
    return !collect_descendants(child, parent->rank, parent, region);
}

// Restores the invariant for the link parent -> child, moving only the
// individuals that the link genuinely constrains. The link must already be
// in place and must have been shown not to close a cycle.
void order_parent_first(Pedigree<> &pedigree, Individual<> *parent,
    Individual<> *child) {
    if (parent->rank < child->rank) {
        return;
    }
    std::vector<Individual<> *> later;
    std::vector<Individual<> *> earlier;
    collect_descendants(child, parent->rank, nullptr, later);
    collect_ancestors(parent, child->rank, earlier);
    reorder_region(pedigree, earlier, later);
}

// Renumbers `ids` from `from` onwards so that rank == position again.
void renumber_from(Pedigree<> &pedigree, int from) {
    for (int position = from; position < (int) pedigree.ids.size();
        position++) {
        pedigree.individuals.at(pedigree.ids[position])->rank = position;
    }
}

}  // namespace

void set_individual(Pedigree<> &pedigree, int id, int father_id,
    int mother_id, int sex) {
    check_record(id, father_id, mother_id, sex);
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
        // Brand-new individual: appended at the back with the next rank.
        // Both parents, if any, are already in the pedigree and therefore
        // already come before, so the invariants hold with no work at all
        // and nobody else is touched: O(1).
        Individual<> *individual = new Individual<>(
            (int) pedigree.ids.size(), id, father, mother, (Sex) sex);
        pedigree.individuals.emplace(id, individual);
        pedigree.ids.push_back(id);
        return;
    }

    // Update an individual already in the pedigree, in place, so that any
    // children already holding a raw pointer to it keep seeing a valid
    // individual.
    Individual<> *individual = it->second;

    // Everything that can be refused is refused here, before the first
    // mutation, so that a rejected edit is a no-op. Cycles are checked
    // against the pedigree as it stands: a new link can only close a cycle
    // through paths that already exist, so the two links can be checked
    // independently of each other.
    if (father && father != individual->father &&
        would_create_cycle(father, individual)) {
        throw std::invalid_argument(
            "individual " + std::to_string(father_id) + " is a descendant " +
            "of individual " + std::to_string(id) + " and cannot also be " +
            "their father");
    }
    if (mother && mother != individual->mother &&
        would_create_cycle(mother, individual)) {
        throw std::invalid_argument(
            "individual " + std::to_string(mother_id) + " is a descendant " +
            "of individual " + std::to_string(id) + " and cannot also be " +
            "their mother");
    }
    if ((Sex) sex != individual->sex) {
        // Changing sex could contradict the role this individual already
        // plays for their existing children.
        for (const Individual<> *child : individual->children) {
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
    }

    individual->sex = (Sex) sex;
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
    // Restore the order. Each call is O(1) unless that particular link is
    // the one that broke it.
    if (father) {
        order_parent_first(pedigree, father, individual);
    }
    if (mother) {
        order_parent_first(pedigree, mother, individual);
    }
}

void set_individuals(Pedigree<> &pedigree,
    const std::vector<PedigreeEntry> &entries) {
    if (entries.empty()) {
        return;
    }

    // 1. Validate every record on its own and collapse duplicates, the
    //    last record for an ID winning as in `dict.update`.
    std::unordered_map<int, PedigreeEntry> pending;
    std::vector<int> pending_ids;
    pending.reserve(entries.size());
    for (const PedigreeEntry &entry : entries) {
        check_record(entry.id, entry.father_id, entry.mother_id, entry.sex);
        auto inserted = pending.emplace(entry.id, entry);
        if (inserted.second) {
            pending_ids.push_back(entry.id);
        } else {
            inserted.first->second = entry;
        }
    }

    // 2. Give every individual a provisional position: those already in the
    //    pedigree keep the one they have, newcomers are provisionally
    //    appended in the order they first appear in the batch. Ties in the
    //    sort below are broken by this position, so the pedigree comes out
    //    as close as possible to the order it already had.
    std::unordered_map<int, int> position;
    position.reserve(pedigree.ids.size() + pending_ids.size());
    std::vector<int> all_ids(pedigree.ids.begin(), pedigree.ids.end());
    for (int index = 0; index < (int) all_ids.size(); index++) {
        position.emplace(all_ids[index], index);
    }
    std::vector<int> new_ids;
    for (const int id : pending_ids) {
        if (pedigree.individuals.find(id) == pedigree.individuals.end()) {
            position.emplace(id, (int) all_ids.size() + (int) new_ids.size());
            new_ids.push_back(id);
        }
    }
    all_ids.insert(all_ids.end(), new_ids.begin(), new_ids.end());
    const int total = (int) all_ids.size();

    // The record every individual will end up with once the batch is
    // applied: the pending one if there is one, the current one otherwise.
    auto final_record = [&] (int id) {
        auto it = pending.find(id);
        if (it != pending.end()) {
            return it->second;
        }
        const Individual<> *individual = pedigree.individuals.at(id);
        return PedigreeEntry{id,
            individual->father ? individual->father->id : 0,
            individual->mother ? individual->mother->id : 0,
            (int) individual->sex};
    };

    // 3. Check every parent reference and every parent's role against the
    //    state the pedigree will be in after the batch, not the state it is
    //    in now: a record may name a parent that another record in the same
    //    batch introduces, or whose sex another record in the same batch
    //    corrects. Every individual is checked, not only the ones in the
    //    batch, so that changing somebody's sex cannot silently contradict
    //    a child that the batch does not mention.
    for (const int id : all_ids) {
        const PedigreeEntry record = final_record(id);
        if (record.father_id != 0) {
            if (position.find(record.father_id) == position.end()) {
                throw std::out_of_range(
                    "father with ID " + std::to_string(record.father_id) +
                    " does not exist in the pedigree and is not part of "
                    "this batch");
            }
            if (final_record(record.father_id).sex == FEMALE) {
                throw std::invalid_argument(
                    "individual " + std::to_string(record.father_id) +
                    " is recorded as female and cannot be set as the "
                    "father of individual " + std::to_string(id));
            }
        }
        if (record.mother_id != 0) {
            if (position.find(record.mother_id) == position.end()) {
                throw std::out_of_range(
                    "mother with ID " + std::to_string(record.mother_id) +
                    " does not exist in the pedigree and is not part of "
                    "this batch");
            }
            if (final_record(record.mother_id).sex == MALE) {
                throw std::invalid_argument(
                    "individual " + std::to_string(record.mother_id) +
                    " is recorded as male and cannot be set as the mother "
                    "of individual " + std::to_string(id));
            }
        }
    }

    // 4. One topological sort of the whole post-batch graph, ready
    //    individuals being emitted in provisional-position order so the
    //    result is deterministic and stays close to the previous order.
    std::unordered_map<int, std::vector<int>> children_of;
    std::unordered_map<int, int> remaining_parents;
    children_of.reserve(total);
    remaining_parents.reserve(total);
    for (const int id : all_ids) {
        remaining_parents.emplace(id, 0);
    }
    for (const int id : all_ids) {
        const PedigreeEntry record = final_record(id);
        if (record.father_id != 0) {
            children_of[record.father_id].push_back(id);
            remaining_parents.at(id)++;
        }
        if (record.mother_id != 0) {
            children_of[record.mother_id].push_back(id);
            remaining_parents.at(id)++;
        }
    }
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>,
        std::greater<std::pair<int, int>>> ready;
    for (const int id : all_ids) {
        if (remaining_parents.at(id) == 0) {
            ready.emplace(position.at(id), id);
        }
    }
    std::vector<int> sorted_ids;
    sorted_ids.reserve(total);
    while (!ready.empty()) {
        const int id = ready.top().second;
        ready.pop();
        sorted_ids.push_back(id);
        auto it = children_of.find(id);
        if (it == children_of.end()) {
            continue;
        }
        for (const int child_id : it->second) {
            if (--remaining_parents.at(child_id) == 0) {
                ready.emplace(position.at(child_id), child_id);
            }
        }
    }
    if ((int) sorted_ids.size() != total) {
        int blocked = 0;
        for (const int id : all_ids) {
            if (remaining_parents.at(id) > 0) {
                blocked = id;
                break;
            }
        }
        throw std::invalid_argument(
            "this batch would make individual " + std::to_string(blocked) +
            " their own ancestor");
    }

    // 5. Commit. Nothing above touched the pedigree, so every rejection
    //    above left it exactly as it was.
    for (const int id : new_ids) {
        pedigree.individuals.emplace(id,
            new Individual<>(0, id, nullptr, nullptr, UNKNOWN));
    }
    for (const int id : all_ids) {
        pedigree.individuals.at(id)->children.clear();
    }
    pedigree.ids.assign(sorted_ids.begin(), sorted_ids.end());
    for (int rank = 0; rank < total; rank++) {
        Individual<> *individual = pedigree.individuals.at(sorted_ids[rank]);
        const PedigreeEntry record = final_record(sorted_ids[rank]);
        individual->rank = rank;
        individual->sex = (Sex) record.sex;
        individual->father = record.father_id
            ? pedigree.individuals.at(record.father_id) : nullptr;
        individual->mother = record.mother_id
            ? pedigree.individuals.at(record.mother_id) : nullptr;
        // Children are rebuilt in topological order, so their order is a
        // property of the pedigree rather than of the edit history.
        if (individual->father) {
            individual->father->children.push_back(individual);
        }
        if (individual->mother) {
            individual->mother->children.push_back(individual);
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
    // rank is the position in `ids`, so there is nothing to search for.
    const int position = individual->rank;
    pedigree.individuals.erase(it);
    pedigree.ids.erase(pedigree.ids.begin() + position);
    delete individual;
    renumber_from(pedigree, position);
}

void remove_individuals(Pedigree<> &pedigree, const std::vector<int> &ids) {
    std::unordered_set<int> removed;
    removed.reserve(ids.size());
    for (const int id : ids) {
        if (pedigree.individuals.find(id) == pedigree.individuals.end()) {
            throw std::out_of_range(
                "individual " + std::to_string(id) +
                " does not exist in the pedigree");
        }
        removed.insert(id);
    }
    if (removed.empty()) {
        return;
    }
    for (const int id : removed) {
        Individual<> *individual = pedigree.individuals.at(id);
        // Only survivors need their children list cleaned up; the others
        // are about to be freed anyway.
        if (individual->father && !removed.count(individual->father->id)) {
            unlink_child(individual->father, individual);
        }
        if (individual->mother && !removed.count(individual->mother->id)) {
            unlink_child(individual->mother, individual);
        }
        for (Individual<> *child : individual->children) {
            if (child->father == individual) {
                child->father = nullptr;
            }
            if (child->mother == individual) {
                child->mother = nullptr;
            }
        }
    }
    std::deque<int> survivors;
    for (const int id : pedigree.ids) {
        if (!removed.count(id)) {
            survivors.push_back(id);
        }
    }
    pedigree.ids = std::move(survivors);
    for (const int id : removed) {
        delete pedigree.individuals.at(id);
        pedigree.individuals.erase(id);
    }
    renumber_from(pedigree, 0);
}
