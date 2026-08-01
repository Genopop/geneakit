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

// Adds a new individual to the pedigree, or updates the sex and parents of
// an individual that is already present.
//
// A new individual with at least one known parent is appended after it
// (parents must already exist in the pedigree), so a rank past the current
// maximum keeps it ordered: O(1) amortized, and nobody else is touched. A
// new individual with no known parents (a founder) is instead inserted
// with a rank below the current minimum, which leaves it free to become
// the parent of any individual already in the pedigree later - including
// one added long before it - since it starts out ordered before all of
// them. Updating an existing individual's parents only ever needs to check
// that the new parent's rank is already lower, so retroactively attaching
// a founder as someone's parent is also O(1); attaching an individual that
// itself has parents (and so has a fixed position from when it was added)
// is only allowed if it already comes before the child, which is what
// keeps the whole pedigree topologically ordered without ever having to
// renumber it, and also rules out creating a cycle. The individual's
// identity (its pointer) never changes on update, so any raw pointers
// already held by its children (via child->father / child->mother) stay
// valid.
//
// Throws std::invalid_argument for data that would make the pedigree
// inconsistent (unknown sex code, an individual named as their own parent,
// a parent whose recorded sex conflicts with the requested role, or a
// non-founder parent that comes after the individual being updated).
// Throws std::out_of_range if a referenced parent ID is not in the
// pedigree.
void set_individual(Pedigree<> &pedigree, int id, int father_id,
    int mother_id, int sex);

// Removes an individual from the pedigree: it is unlinked from its own
// parents' children lists, any children pointing to it are orphaned on
// that side (their father/mother pointer is cleared, since the individual
// they pointed to no longer exists), and its memory is freed.
//
// Throws std::out_of_range if the ID is not in the pedigree.
void remove_individual(Pedigree<> &pedigree, int id);

#endif
