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

#include "../include/output.hpp"

// Saves the pedigree to a file
void save_pedigree(Pedigree<> &pedigree, std::string pedigree_file) {
    // Open the file
    std::ofstream file(pedigree_file);
    // Write the header
    file << "ind    father  mother  sex\n";
    // Write the individuals
    for (int id : pedigree.ids) {
        Individual<> *individual = pedigree.individuals.at(id);
        file << individual->id << "\t";
        if (individual->father) {
            file << individual->father->id << "\t";
        } else {
            file << "0\t";
        }
        if (individual->mother) {
            file << individual->mother->id << "\t";
        } else {
            file << "0\t";
        }
        file << (int) individual->sex << "\n";
    }
    // Close the file
    file.close();
}

// Returns the pedigree as a vector of IDs, father IDs, mother IDs and sexes
Matrix<int> output_pedigree(Pedigree<> &pedigree) {
    Matrix<int> output = Matrix<int>(pedigree.ids.size(), 4);
    #pragma omp parallel for
    for (int i = 0; i < (int) pedigree.ids.size(); i++) {
        const int id = pedigree.ids.at(i);
        Individual<> *individual = pedigree.individuals.at(id);
        output[i][0] = individual->id;
        if (individual->father) {
            output[i][1] = individual->father->id;
        } else {
            output[i][1] = 0;
        }
        if (individual->mother) {
            output[i][2] = individual->mother->id;
        } else {
            output[i][2] = 0;
        }
        output[i][3] = (int) individual->sex;
    }
    return output;
}