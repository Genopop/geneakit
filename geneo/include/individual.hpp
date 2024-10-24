#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

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
#include "data.hpp"

enum Sex {
    UNKNOWN,
    MALE,
    FEMALE
};

// A structure to represent an individual.
template <typename T = Empty>
struct Individual {
    int rank;
    int id;
    Individual<T> *father;
    Individual<T> *mother;
    Sex sex;
    std::vector<Individual<T> *> children;
    mutable T data;
    // Constructor
    Individual(int rank, int id, Individual<T> *father,
        Individual<T> *mother, Sex sex) {
        this->rank = rank;
        this->id = id;
        this->father = father;
        this->mother = mother;
        this->sex = sex;
        this->data = T();
        if (father) {
            father->children.push_back(this);
        }
        if (mother) {
            mother->children.push_back(this);
        }
    }
};

#endif