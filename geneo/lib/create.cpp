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

#include "../include/create.hpp"

// Creates an unordered pedigree from a file
Pedigree<ParentIDs> create_unsorted_pedigree(std::string pedigree_file) {
    // Initialize the pedigree
    Pedigree<ParentIDs> unsorted_pedigree = Pedigree<ParentIDs>();
    std::unordered_map<int, Individual<ParentIDs> *> individuals;
    std::vector<int> ids;
    int rank = 0;
    // Read the pedigree file
    std::ifstream file(pedigree_file);
    // Check if the file can be opened
    if (!file) {
		std::cerr << "Unable to open " << pedigree_file << std::endl;
        exit(1);
	}
    std::string line;
    // Ignore the first line
    getline(file, line);
    std::regex pattern("\\D+|(\\d+)\\D*(\\d+)\\D*(\\d+)\\D*(\\d+)\\D*");
    while (getline(file, line)) {
        // Parse the line using a regular expression to handle various separators
        std::smatch match;
        if (!std::regex_search(line, match, pattern)) {
            std::cerr << "Invalid line format in pedigree file: " <<
                line << std::endl;
            exit(1);
        }
        int id = std::stoi(match[1]);
        int father_id = std::stoi(match[2]);
        int mother_id = std::stoi(match[3]);
        int sex = std::stoi(match[4]);
        // Create the individual
        Individual<ParentIDs> *individual = new Individual<ParentIDs>(
            rank++, id, nullptr, nullptr, (Sex) sex
        );
        // Add the parents to the individual
        individual->data.father_id = father_id;
        individual->data.mother_id = mother_id;
        // Add the individual to the pedigree
        individuals.emplace(id, individual);
        ids.push_back(id);
    }
    // Add the individuals to the pedigree
    unsorted_pedigree.individuals = individuals;
    unsorted_pedigree.ids = ids;
    return unsorted_pedigree;
}

// Creates an unordered pedigree from vectors
Pedigree<ParentIDs> create_unsorted_pedigree(std::vector<int> ids,
    std::vector<int> father_ids, std::vector<int> mother_ids,
    std::vector<int> sexes) {
    // Initialize the pedigree
    Pedigree<ParentIDs> unsorted_pedigree = Pedigree<ParentIDs>();
    std::unordered_map<int, Individual<ParentIDs> *> individuals;
    // Fill the pedigree
    for (int i = 0; i < (int) ids.size(); i++) {
        // Create the individual
        Individual<ParentIDs> *individual = new Individual<ParentIDs>(
            i, ids[i], nullptr, nullptr, (Sex) sexes[i]
        );
        // Add the parents to the individual
        individual->data.father_id = father_ids[i];
        individual->data.mother_id = mother_ids[i];
        // Add the individual to the pedigree
        individuals.emplace(ids[i], individual);
    }
    // Add the individuals to the pedigree
    unsorted_pedigree.individuals = individuals;
    unsorted_pedigree.ids = ids;
    return unsorted_pedigree;
}

// Converts an unordered pedigree to a sorted pedigree using depth-first search
void populate_pedigree_dfs(Pedigree<> &pedigree,
    Pedigree<ParentIDs> &unsorted_pedigree, int id, int &rank) {
    Individual<ParentIDs> *unsorted_individual =
        unsorted_pedigree.individuals.at(id);
    if (unsorted_individual->data.visited) {
        return;
    }
    if (unsorted_individual->data.father_id) {
        populate_pedigree_dfs(pedigree, unsorted_pedigree,
            unsorted_individual->data.father_id, rank);
    }
    if (unsorted_individual->data.mother_id) {
        populate_pedigree_dfs(pedigree, unsorted_pedigree,
            unsorted_individual->data.mother_id, rank);
    }
    // Create the individual
    Individual<> *father;
    if (unsorted_individual->data.father_id) {
        father = pedigree.individuals.at(unsorted_individual->data.father_id);
    } else {
        father = nullptr;
    }
    Individual<> *mother;
    if (unsorted_individual->data.mother_id) {
        mother = pedigree.individuals.at(unsorted_individual->data.mother_id);
    } else {
        mother = nullptr;
    }
    Individual<> *individual = new Individual(
        rank++, id, father, mother, unsorted_individual->sex);
    // Add the individual to the pedigree
    pedigree.ids.push_back(id);
    pedigree.individuals.emplace(id, individual);
    unsorted_individual->data.visited = true;
}

// Converts an unordered pedigree to a sorted pedigree
Pedigree<> convert_pedigree(Pedigree<ParentIDs> &unsorted_pedigree,
    bool sorted) {
    // Initialize the pedigree
    Pedigree pedigree;
    std::unordered_map<int, Individual<> *> individuals;
    // Check if the pedigree is already sorted
    if (sorted) {
        // Fill the pedigree
        int rank = 0;
        for (const int id : unsorted_pedigree.ids) {
            Individual<ParentIDs> *individual =
                unsorted_pedigree.individuals.at(id);
            // Create the individual
            Individual<> *father;
            if (individual->data.father_id) {
                father = individuals.at(individual->data.father_id);
            } else {
                father = nullptr;
            }
            Individual<> *mother;
            if (individual->data.mother_id) {
                mother = individuals.at(individual->data.mother_id);
            } else {
                mother = nullptr;
            }
            Individual<> *sorted_individual = new Individual(
                rank++, id, father, mother, individual->sex
            );
            // Add the individual to the pedigree
            pedigree.individuals.emplace(id, sorted_individual);
        }
        // Add the individuals to the pedigree
        pedigree.ids = unsorted_pedigree.ids;
        pedigree.individuals = individuals;
    } else {
        // Fill the pedigree using depth-first search
        int rank = 0;
        for (int id : unsorted_pedigree.ids) {
            populate_pedigree_dfs(pedigree, unsorted_pedigree, id, rank);
        }
    }
    return pedigree;
}

// Creates a pedigree from a file
Pedigree<> load_pedigree_from_file(std::string pedigree_file,
    bool sorted) {
    // Construct an unordered pedigree from integers
    Pedigree<ParentIDs> unsorted_pedigree = create_unsorted_pedigree(
        pedigree_file);
    return convert_pedigree(unsorted_pedigree, sorted);
}

// Creates a pedigree from vectors
Pedigree<> load_pedigree_from_vectors(std::vector<int> ids,
    std::vector<int> father_ids, std::vector<int> mother_ids,
    std::vector<int> sexes, bool sorted) {
    // Construct an unordered pedigree from integers
    Pedigree<ParentIDs> unsorted_pedigree = create_unsorted_pedigree(
        ids, father_ids, mother_ids, sexes);
    return convert_pedigree(unsorted_pedigree, sorted);
}