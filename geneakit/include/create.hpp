#ifndef CREATE_H
#define CREATE_H

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

#include <fstream>
#include <iostream>
#include <sstream>
#include <regex>
#include "pedigree.hpp"

// Creates an unordered pedigree from a file
Pedigree<ParentIDs> create_unsorted_pedigree(std::string pedigree_file);

// Creates an unordered pedigree from vectors
Pedigree<ParentIDs> create_unsorted_pedigree(std::vector<int> ids,
    std::vector<int> father_ids, std::vector<int> mother_ids,
    std::vector<int> sexes);

// Converts an unordered pedigree to a sorted pedigree using depth-first search
void populate_pedigree_dfs(Pedigree<> &pedigree,
    Pedigree<ParentIDs> &unsorted_pedigree, int id, int &rank);

// Converts an unordered pedigree to a sorted pedigree
Pedigree<> convert_pedigree(Pedigree<ParentIDs> &unsorted_pedigree,
    bool sorted);

// Creates a pedigree from a file
Pedigree<> load_pedigree_from_file(std::string pedigree_file, bool sorted);

// Creates a pedigree from vectors
Pedigree<> load_pedigree_from_vectors(std::vector<int> ids,
    std::vector<int> father_ids, std::vector<int> mother_ids,
    std::vector<int> sexes, bool sorted);

#endif