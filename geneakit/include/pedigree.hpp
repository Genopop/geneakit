#ifndef PEDIGREE_H
#define PEDIGREE_H

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

#include <parallel_hashmap/phmap.h>
#include "individual.hpp"

// A structure to represent a pedigree.
template <typename T = Empty>
struct Pedigree {
    std::vector<int> ids;
    phmap::flat_hash_map<int, Individual<T> *> individuals;
    // Constructors
    Pedigree() {};
    Pedigree(std::vector<int> ids,
        phmap::flat_hash_map<int, Individual<T> *> individuals) :
        ids(ids), individuals(individuals) {};
    // Copy constructor
    Pedigree(const Pedigree<T>& other) {
        ids = other.ids;
        for (const int id : ids) {
            Individual<T> *individual = other.individuals.at(id);
            int rank = individual->rank;
            Individual<T> *father = individual->father ?
                individuals.at(individual->father->id) : nullptr;
            Individual<T> *mother = individual->mother ?
                individuals.at(individual->mother->id) : nullptr;
            Sex sex = individual->sex;
            Individual<T> *individual_copy = new Individual<T>(
                rank, id, father, mother, sex);
            individuals.emplace(id, individual_copy);
        }
    }
    // Conversion constructor
    Pedigree(Pedigree<> &other) {
        this->ids = other.ids;
        for (const int id : other.ids) {
            Individual<> *individual = other.individuals.at(id);
            Individual<T> *father = individual->father ?
                this->individuals.at(individual->father->id) : nullptr;
            Individual<T> *mother = individual->mother ?
                this->individuals.at(individual->mother->id) : nullptr;
            Individual<T> *data_individual =
                new Individual<T>(individual->rank, individual->id,
                    father, mother, individual->sex);
            this->individuals.emplace(id, data_individual);
        }
    }
    // Move assignment operator
    Pedigree& operator=(Pedigree&& other) noexcept {
        if (this != &other) {
            ids = std::move(other.ids);
            for (auto const& [id, individual] : individuals) {
                delete individual;
            }
            individuals = std::move(other.individuals);
            other.individuals.clear();
        }
        return *this;
    }
    // Destructor
    ~Pedigree() {
        for (auto const& [id, individual] : individuals) {
            delete individual;
        }
    };
};

#endif