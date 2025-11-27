#include "../include/cgeneakit.hpp"

NB_MODULE(cgeneakit, m) {
    m.doc() = "A C++/Python module for genealogical analysis.";

    nb::class_<Pedigree<>>(m, "Pedigree")
        .def(nb::init<std::vector<int>, std::unordered_map<int, Individual<> *>>())
        .def("__str__", [] (Pedigree<> &pedigree) {
            return "A pedigree with:\n - " +
                std::to_string(get_number_of_individuals(pedigree)) +
                    " individuals;\n - " +
                std::to_string(get_number_of_parent_child_relations(pedigree)) +
                    " parent-child relations;\n - " +
                std::to_string(get_number_of_men(pedigree)) + " men;\n - " +
                std::to_string(get_number_of_women(pedigree)) + " women;\n - " +
                std::to_string(get_proband_ids(pedigree).size()) +
                    " probands;\n - " +
                std::to_string(get_pedigree_depth(pedigree)) + " generations.";
        })
        .def("__repr__", [] (Pedigree<> &pedigree) {
            return "A pedigree with:\n - " +
                std::to_string(get_number_of_individuals(pedigree)) +
                    " individuals;\n - " +
                std::to_string(get_number_of_parent_child_relations(pedigree)) +
                    " parent-child relations;\n - " +
                std::to_string(get_number_of_men(pedigree)) + " men;\n - " +
                std::to_string(get_number_of_women(pedigree)) + " women;\n - " +
                std::to_string(get_proband_ids(pedigree).size()) +
                    " probands;\n - " +
                std::to_string(get_pedigree_depth(pedigree)) + " generations.";
        })
        .def("__getitem__", [] (Pedigree<> &pedigree, int id) {
            if (id == 0) {
                return Individual<>(0, 0, nullptr, nullptr, UNKNOWN);
            } else {
                return *pedigree.individuals.at(id);
            }
        })
        .def("__len__", &get_number_of_individuals)
        .def("__iter__", [] (Pedigree<> &pedigree) {
            return nb::make_iterator(
                nb::type<Pedigree<>>(),
                "keys_iterator",
                pedigree.ids.begin(),
                pedigree.ids.end()
            );
        }, nb::keep_alive<0,1>())
        .def("keys", [] (Pedigree<> &pedigree) {
            return pedigree.ids;
        })
        .def("items", [] (Pedigree<> &pedigree) {
            return nb::make_iterator(
                nb::type<Pedigree<>>(),
                "items_iterator",
                pedigree.individuals.begin(),
                pedigree.individuals.end()
            );
        }, nb::keep_alive<0,1>())
        .def("values", [] (Pedigree<> &pedigree) {
            return nb::make_value_iterator(
                nb::type<Pedigree<>>(),
                "values_iterator",
                pedigree.individuals.begin(),
                pedigree.individuals.end()
            );
        }, nb::keep_alive<0,1>());

    nb::class_<Individual<>>(m, "Individual")
        .def(nb::init<int, int, Individual<> *, Individual<> *, Sex>())
        .def("__str__", [] (Individual<> &individual) {
            return "ind: " + std::to_string(individual.id) +
                "\nfather: " + std::to_string(
                    individual.father ? individual.father->id : 0
                ) +
                "\nmother: " + std::to_string(
                    individual.mother ? individual.mother->id : 0
                ) +
                "\nsex: " + std::to_string((int) individual.sex);
        })
        .def("__repr__", [] (Individual<> &individual) {
            return "ind: " + std::to_string(individual.id) +
                "\nfather: " + std::to_string(
                    individual.father ? individual.father->id : 0
                ) +
                "\nmother: " + std::to_string(
                    individual.mother ? individual.mother->id : 0
                ) +
                "\nsex: " + std::to_string((int) individual.sex);
        })
        .def_ro("ind", &Individual<>::id)
        .def_prop_ro("father", [] (Individual<> &individual) {
            if (individual.father) {
                return *individual.father;
            } else {
                return Individual<>(0, 0, nullptr, nullptr, UNKNOWN);
            }
        })
        .def_prop_ro("mother", [] (Individual<> &individual) {
            if (individual.mother) {
                return *individual.mother;
            } else {
                return Individual<>(0, 0, nullptr, nullptr, UNKNOWN);
            }
        })
        .def_prop_ro("sex", [] (Individual<> &individual) {
            return (int) individual.sex;    
        })
        .def_prop_ro("rank", [] (Individual<> &individual) {
            return (int) individual.rank;    
        })
        .def_prop_ro("children", [] (Individual<> &individual) {
            std::vector<Individual<>> children;
            for (Individual<> *child : individual.children) {
                children.push_back(*child);
            }
            return children;
        });

    m.def("load_pedigree_from_file", &load_pedigree_from_file,
        "Returns a pedigree loaded from a file.");

    m.def("load_pedigree_from_vectors", &load_pedigree_from_vectors,
        "Returns a pedigree loaded from vectors.");

    m.def("extract_pedigree", &extract_pedigree,
        "Returns an extracted pedigree.");

    m.def("extract_lineages", &extract_lineages,
        "Returns the lineages of individuals.");

    m.def("output_pedigree", [] (Pedigree<> &pedigree) {
        Matrix<int> output = output_pedigree(pedigree);
        size_t rows = output.rows();
        size_t cols = output.cols();
        int *data = output.data();
        
        // Move the Matrix into the capsule to keep it alive
        nb::capsule owner(new Matrix<int>(std::move(output)), 
            [](void *ptr) noexcept {
                delete static_cast<Matrix<int>*>(ptr);
            });
        
        return nb::ndarray<nb::numpy, int, nb::ndim<2>>(
            data,
            {rows, cols},
            owner
        );
    },
    "Outputs a pedigree to a vector of vectors.");

    m.def("get_proband_ids", &get_proband_ids,
        "Returns the proband IDs of a pedigree.");

    m.def("get_founder_ids", &get_founder_ids,
        "Returns the founder IDs of a pedigree.");

    m.def("get_half_founder_ids", &get_half_founder_ids,
        "Returns the half founder IDs of a pedigree.");

    m.def("get_father_ids", &get_father_ids,
        "Returns the IDs of the fathers of a vector of individuals.");
    
    m.def("get_mother_ids", &get_mother_ids,
        "Returns the IDs of the mothers of a vector of individuals.");

    m.def("get_sibling_ids", &get_sibling_ids,
        "Returns the IDs of the siblings of a vector of individuals.");

    m.def("get_children_ids", &get_children_ids,
        "Returns the IDs of the children of a vector of individuals.");

    m.def("get_ancestor_ids", &get_ancestor_ids,
        "Returns the IDs of the ancestors of a vector of individuals.");

    m.def("get_all_ancestor_ids", &get_all_ancestor_ids,
        "Returns the IDs of all occurrences of ancestors of a vector of individuals.");

    m.def("get_descendant_ids", &get_descendant_ids,
        "Returns the IDs of the descendants of a vector of individuals.");

    m.def("get_all_descendant_ids", &get_all_descendant_ids,
        "Returns the IDs of all occurrences of descendants of a vector of individuals.");
    
    m.def("get_common_ancestor_ids", &get_common_ancestor_ids,
        "Returns the IDs of the common ancestors of a vector of individuals.");

    m.def("get_common_founder_ids", &get_common_founder_ids,
        "Returns the IDs of the common founders of a vector of individuals.");

    m.def("get_mrca_ids", &get_mrca_ids,
        "Returns the IDs of the MRCAs of a vector of IDs.");

    m.def("get_mrca_meioses", [] (Pedigree<> &pedigree,
        std::vector<int> proband_ids, std::vector<int> ancestor_ids) {
            Matrix<int> meioses = get_mrca_meioses(
                pedigree, proband_ids, ancestor_ids
            );
            size_t rows = meioses.rows();
            size_t cols = meioses.cols();
            int *data = meioses.data();
            
            nb::capsule owner(new Matrix<int>(std::move(meioses)), 
                [](void *ptr) noexcept {
                    delete static_cast<Matrix<int>*>(ptr);
                });
            
            return nb::ndarray<nb::numpy, int, nb::ndim<2>>(
                data,
                {rows, cols},
                owner
            );
        },
        "Returns the number of meioses between probands and MRCAs.");

    m.def("get_number_of_individuals", &get_number_of_individuals,
        "Returns the number of individuals in the pedigree.");

    m.def("get_number_of_men", &get_number_of_men,
        "Returns the number of men in the pedigree.");

    m.def("get_number_of_women", &get_number_of_women,
        "Returns the number of women in the pedigree.");

    m.def("get_pedigree_depth", &get_pedigree_depth,
        "Returns the depth of the pedigree.");

    m.def("get_mean_pedigree_depths", &get_mean_pedigree_depths,
        "Returns the mean depths of the pedigree for the specified probands.");

    m.def("get_min_ancestor_path_lengths", &get_min_ancestor_path_lengths,
        "Returns the minimum path length to a vector of ancestors.");

    m.def("get_mean_ancestor_path_lengths", &get_mean_ancestor_path_lengths,
        "Returns the mean path length to a vector of ancestors.");

    m.def("get_max_ancestor_path_lengths", &get_max_ancestor_path_lengths,
        "Returns the maximum path length to a vector of ancestors.");

    m.def("get_number_of_children", &get_number_of_children,
        "Returns the number of children of a vector of individuals.");

    m.def("compute_mean_completeness", &compute_mean_completeness,
        "Returns the completeness of the pedigree.");

    m.def("compute_individual_completeness", [] (Pedigree<> &pedigree,
    std::vector<int> proband_ids) {
        Matrix<double> completeness_matrix =
            compute_individual_completeness(pedigree, proband_ids);
        size_t rows = completeness_matrix.rows();
        size_t cols = completeness_matrix.cols();
        double *data = completeness_matrix.data();
        
        nb::capsule owner(new Matrix<double>(std::move(completeness_matrix)), 
            [](void *ptr) noexcept {
                delete static_cast<Matrix<double>*>(ptr);
            });
        
        return nb::ndarray<nb::numpy, double, nb::ndim<2>>(
            data,
            {rows, cols},
            owner
        );
    },
    "Returns the completeness of the pedigree.");

    m.def("compute_mean_implex", &compute_mean_implex,
        "Returns the mean implex of the pedigree per generation.");

    m.def("compute_individual_implex", [] (Pedigree<> &pedigree,
    std::vector<int> proband_ids, bool only_new_ancestors) {
        Matrix<double> implex_matrix = compute_individual_implex(
            pedigree, proband_ids, only_new_ancestors
        );
        size_t rows = implex_matrix.rows();
        size_t cols = implex_matrix.cols();
        double *data = implex_matrix.data();
        
        nb::capsule owner(new Matrix<double>(std::move(implex_matrix)), 
            [](void *ptr) noexcept {
                delete static_cast<Matrix<double>*>(ptr);
            });
        
        return nb::ndarray<nb::numpy, double, nb::ndim<2>>(
            data,
            {rows, cols},
            owner
        );
    },
    "Returns the implex of the pedigree.");

    m.def("count_total_occurrences", &count_total_occurrences,
        "Returns the total occurrences of ancestors in the probands' pedigrees.");

    m.def("count_individual_occurrences", [] (Pedigree<> &pedigree,
    std::vector<int> ancestor_ids, std::vector<int> proband_ids) {
        Matrix<int> occurrences = count_individual_occurrences(
            pedigree, ancestor_ids, proband_ids
        );
        size_t rows = occurrences.rows();
        size_t cols = occurrences.cols();
        int *data = occurrences.data();
        
        nb::capsule owner(new Matrix<int>(std::move(occurrences)), 
            [](void *ptr) noexcept {
                delete static_cast<Matrix<int>*>(ptr);
            });
        
        return nb::ndarray<nb::numpy, int, nb::ndim<2>>(
            data,
            {rows, cols},
            owner
        );
    },
    "Returns the occurrences of ancestors in the probands' pedigrees.");

    m.def("get_ancestor_path_lengths", &get_ancestor_path_lengths,
        "Returns the lengths of the paths from an individual to an ancestor.");

    m.def("get_min_common_ancestor_path_length",
        &get_min_common_ancestor_path_length,
        "Returns the minimum distance between two probands and an ancestor.");

    m.def("count_coverage", &count_coverage,
        "Returns how many probands descend from a vector of ancestors.");

    m.def("get_required_memory_for_kinships", &get_required_memory_for_kinships,
        "Returns the required memory for kinship calculations.");

    m.def("compute_kinships", [] (Pedigree<> &pedigree,
    std::vector<int> proband_ids, bool verbose) {
        Matrix<double> kinship_matrix = compute_kinships(
            pedigree, proband_ids, verbose
        );
        size_t rows = kinship_matrix.rows();
        size_t cols = kinship_matrix.cols();
        double *data = kinship_matrix.data();
        
        nb::capsule owner(new Matrix<double>(std::move(kinship_matrix)), 
            [](void *ptr) noexcept {
                delete static_cast<Matrix<double>*>(ptr);
            });
        
        return nb::ndarray<nb::numpy, double, nb::ndim<2>>(
            data,
            {rows, cols},
            owner
        );
    },
    "Returns the kinship matrix of a pedigree.");

    m.def("compute_kinships_sparse", [] (Pedigree<> &pedigree,
        std::vector<int> proband_ids, bool verbose, bool symmetric_coo) {
            
        SparseResult result = compute_kinships_sparse(
            pedigree, proband_ids, verbose, symmetric_coo
        );
        
        size_t nnz = result.data.size();

        // Data Vector (Common)
        auto *data_ptr = new std::vector<float>(std::move(result.data));
        nb::capsule data_owner(data_ptr, [](void *p) noexcept { 
            delete static_cast<std::vector<float>*>(p); 
        });
        auto np_data = nb::ndarray<nb::numpy, float, nb::ndim<1>>(
            data_ptr->data(), {nnz}, data_owner);

        // Cols/Indices Vector (Common)
        auto *cols_ptr = new std::vector<int>(std::move(result.indices));
        nb::capsule cols_owner(cols_ptr, [](void *p) noexcept { 
            delete static_cast<std::vector<int>*>(p); 
        });
        auto np_cols = nb::ndarray<nb::numpy, int, nb::ndim<1>>(
            cols_ptr->data(), {nnz}, cols_owner);

        if (symmetric_coo) {
            // Return (data, rows, cols) for COO
            auto *rows_ptr = new std::vector<int>(std::move(result.rows));
            nb::capsule rows_owner(rows_ptr, [](void *p) noexcept { 
                delete static_cast<std::vector<int>*>(p); 
            });
            auto np_rows = nb::ndarray<nb::numpy, int, nb::ndim<1>>(
                rows_ptr->data(), {nnz}, rows_owner);
            
            return nb::make_tuple(np_data, np_rows, np_cols);

        } else {
            // Return (data, indices, indptr) for CSR
            size_t n_rows = result.indptr.size() - 1;
            auto *indptr_ptr = new std::vector<int64_t>(std::move(result.indptr));
            nb::capsule indptr_owner(indptr_ptr, [](void *p) noexcept { 
                delete static_cast<std::vector<int64_t>*>(p); 
            });
            auto np_indptr = nb::ndarray<nb::numpy, int64_t, nb::ndim<1>>(
                indptr_ptr->data(), {n_rows + 1}, indptr_owner);
            
            return nb::make_tuple(np_data, np_cols, np_indptr);
        }
    },
    "Returns sparse kinship matrix vectors (zero-copy).",
    nb::arg("pedigree"), 
    nb::arg("proband_ids") = std::vector<int>(), 
    nb::arg("verbose") = false,
    nb::arg("symmetric_coo") = false);

    m.def("compute_inbreedings", &compute_inbreedings,
        "Returns the inbreeding coefficients of probands.");
    
    m.def("compute_genetic_contributions", [] (Pedigree<> &pedigree,
    std::vector<int> proband_ids, std::vector<int> ancestor_ids) {
        Matrix<double> contribution_matrix = compute_genetic_contributions(
            pedigree, proband_ids, ancestor_ids
        );
        size_t rows = contribution_matrix.rows();
        size_t cols = contribution_matrix.cols();
        double *data = contribution_matrix.data();
        
        nb::capsule owner(new Matrix<double>(std::move(contribution_matrix)), 
            [](void *ptr) noexcept {
                delete static_cast<Matrix<double>*>(ptr);
            });
        
        return nb::ndarray<nb::numpy, double, nb::ndim<2>>(
            data,
            {rows, cols},
            owner
        );
    },
    "Returns the genetic contributions of ancestors.");
}