#ifndef EDIT_H
#define EDIT_H

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

#include "pedigree.hpp"
#include <vector>

// One (individual, father, mother, sex) record, as accepted by
// `set_individuals`. A parent ID of 0 means "unknown".
struct PedigreeEntry {
    int id;
    int father_id;
    int mother_id;
    int sex;
};

// Adds a new individual to the pedigree, or updates the sex and the parents
// of an individual that is already present.
//
// The pedigree keeps two invariants at all times, and this function is
// responsible for restoring both after every edit:
//
//   (1) `ids` is topologically ordered: a parent always appears before
//       their children;
//   (2) `rank` is exactly the position in `ids`, so ranks are always the
//       dense range 0..N-1, with no gaps and no negative values.
//
// A brand-new individual is appended at the back with rank N, which is
// O(1): both of its parents (if any) must already be in the pedigree and
// therefore already come before it.
//
// Re-parenting an individual that is already present may genuinely require
// moving people. Setting individual A as the parent of individual B when A
// currently comes after B is a legitimate correction as long as A is not a
// descendant of B, and it is accepted: the order is repaired incrementally
// with the algorithm of Pearce & Kelly (2007), "A Dynamic Topological Sort
// Algorithm for Directed Acyclic Graphs", ACM Journal of Experimental
// Algorithmics 11. Only the individuals that lie between A and B in the
// current order and that are actually constrained by the new link are
// touched; everybody outside that window keeps their rank, which is what
// allows the ranks to stay dense without renumbering the whole pedigree.
// The cost is O(K log K) where K is the number of individuals that really
// have to move, not O(N).
//
// The individual's identity (its pointer) never changes on update, so any
// raw pointers already held by its children stay valid. Every check is
// performed before anything is modified, so a rejected edit leaves the
// pedigree exactly as it was.
//
// Throws std::invalid_argument for data that would make the pedigree
// inconsistent (unknown sex code, an individual named as their own parent,
// the same individual given as both parents, a parent whose recorded sex
// conflicts with the requested role, or a parent that is a descendant of
// the individual, which would create a cycle). Throws std::out_of_range if
// a referenced parent ID is not in the pedigree.
void set_individual(Pedigree<> &pedigree, int id, int father_id,
    int mother_id, int sex);

// Applies a whole batch of records at once, then repairs the order with a
// single topological sort: O(N + M) for the entire batch, instead of one
// incremental repair per record.
//
// Besides being faster for bulk corrections, the batch form is strictly
// more expressive than repeated calls to `set_individual`: records may
// refer to parents that are themselves defined later in the same batch, so
// a whole sub-pedigree can be spliced in in one go without having to sort
// it by hand first. Ties are broken in favour of the order the pedigree
// already had, so the result stays as close as possible to the previous
// ordering, and individuals new to the pedigree keep the order in which
// they first appear in the batch.
//
// If the same ID appears more than once, the last record wins, as in
// `dict.update`. Nothing is modified until every record has been validated
// and the resulting graph has been shown to be acyclic, so a rejected batch
// leaves the pedigree exactly as it was.
//
// Throws the same exceptions as `set_individual`, plus std::invalid_argument
// naming an individual on the cycle if the batch would create one.
void set_individuals(Pedigree<> &pedigree,
    const std::vector<PedigreeEntry> &entries);

// Removes an individual from the pedigree: it is unlinked from its own
// parents' children lists, any children pointing to it are orphaned on
// that side (their father/mother pointer is cleared, since the individual
// they pointed to no longer exists), and its memory is freed.
//
// The position of the individual is read straight off its rank, so no
// search through `ids` is needed; the remaining cost is the O(N - rank)
// pass that closes the gap and keeps the ranks dense.
//
// Throws std::out_of_range if the ID is not in the pedigree.
void remove_individual(Pedigree<> &pedigree, int id);

// Removes several individuals in a single pass, closing every gap at once:
// O(N) for the whole batch rather than O(N) per individual. IDs may be
// given in any order and may include parents and children of one another.
//
// Throws std::out_of_range, without modifying the pedigree, if any of the
// IDs is not in the pedigree.
void remove_individuals(Pedigree<> &pedigree, const std::vector<int> &ids);

#endif
