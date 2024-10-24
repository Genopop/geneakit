#ifndef DATA_H
#define DATA_H

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

#include <limits>

// An empty structure.
struct Empty {
    // Default constructor
    Empty() {};
};

// A structure used to create a pedigree from IDs.
struct ParentIDs {
    mutable bool visited;
    mutable int father_id;
    mutable int mother_id;
    ParentIDs() {
        visited = false;
        father_id = -1;
        mother_id = -1;
    }
};

// A structure used to find a founder in a founder kinship matrix.
struct Index {
    mutable int index;
    Index() {
        index = -1;
    }
};

// A structure used when extracting a subpedigree.
struct Status {
    mutable bool is_ancestor;
    mutable bool is_descendant;
    Status() {
        is_ancestor = false;
        is_descendant = false;
    }
};

// A structure used to count the number of occurrences.
struct Count {
    mutable int count;
    mutable bool is_ancestor;
    Count() {
        count = 0;
        is_ancestor = false;
    }
};

// A structure used to compute the genetic contribution.
struct Contribution {
    mutable double contribution;
    mutable bool is_proband;
    Contribution() {
        contribution = 0.0;
        is_proband = false;
    }
};

// A structure to compute the minimum, mean, and maximum path lengths.
struct PathLength {
    mutable bool is_ancestor;
    mutable int min;
    mutable double sum;
    mutable int count;
    mutable int max;
    PathLength() {
        is_ancestor = false;
        min = std::numeric_limits<int>::max();
        sum = 0.0;
        count = 0;
        max = -1;
    }
};

#endif