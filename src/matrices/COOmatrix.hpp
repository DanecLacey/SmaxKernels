#ifndef COOMATRIX_HPP
#define COOMATRIX_HPP

#include <vector>

#include "sparse_matrix_base.hpp"

template <typename VT, typename IT>

class COOMatrix : sparsematrix<IT> {
    using base = sparsematrix<IT>;
    using base::ncols_;
    using base::nnz_;
    using base::nrows_;
    using base::is_sorted_;
    using base::is_symmetric_;

  public:
    COOMatrix(IT n_rows = 0, IT n_cols = 0, IT n_nzs = 0,
              bool is_sorted = false, bool is_symmetric = false)
        : sparsematrix<IT>(n_rows, n_cols, n_nzs, is_sorted, is_symmetric),
          I_(n_nzs), J_(n_nzs), val_(n_nzs) {}

    void print() const override {
        base::print();
        std::cout << "Values: ";
        for (IT i = 0; i < this->nnz; ++i)
            std::cout << this->val_[i] << " ";

        std::cout << "\nCol: ";
        for (IT i = 0; i < this->nnz; ++i)
            std::cout << this->J_[i] << " ";

        std::cout << "\nRow: ";
        for (IT i = 0; i < this->nnz; ++i)
            std::cout << this->I_[i] << " ";

        std::cout << "\n" << std::endl;
    }

    void read_from_mtx(const std::string &matrix_file_name) {
#ifdef DEBUG_MODE
        std::cout << "Reading matrix from file: " << matrix_file_name
                  << std::endl;
#endif
#ifdef USE_FAST_MMIO
        std::vector<int> original_rows;
        std::vector<int> original_cols;
        std::vector<double> original_vals;

        fmm::matrix_market_header header;

        // Load
        {
            fmm::read_options options;
            options.generalize_symmetry = true;
            std::ifstream f(matrix_file_name);
            fmm::read_matrix_market_triplet(f, header, original_rows,
                                            original_cols, original_vals,
                                            options);
        }

        // Find sort permutation
        auto perm = compute_sort_permutation(original_rows, original_cols);

        // Apply permutation
        this->I = apply_permutation(perm, original_rows);
        this->J = apply_permutation(perm, original_cols);
        this->val = apply_permutation(perm, original_vals);

        this->n_rows = header.nrows;
        this->n_cols = header.ncols;
        this->nnz = this->val.size();
        this->is_sorted = true;
        this->is_symmetric = (header.symmetry != fmm::symmetry_type::general);
#else
        MM_typecode matcode;
        FILE *f = fopen(matrix_file_name.c_str(), "r");
        if (!f) {
            throw std::runtime_error("Unable to open file: " +
                                     matrix_file_name);
        }

        if (mm_read_banner(f, &matcode) != 0) {
            fclose(f);
            throw std::runtime_error(
                "Could not process Matrix Market banner in file: " +
                matrix_file_name);
        }

        fclose(f);

        if (!(mm_is_sparse(matcode) &&
              (mm_is_real(matcode) || mm_is_pattern(matcode) ||
               mm_is_integer(matcode)) &&
              (mm_is_symmetric(matcode) || mm_is_general(matcode)))) {
            throw std::runtime_error("Unsupported matrix format in file: " +
                                     matrix_file_name);
        }

        int nrows, ncols, nnz;
        int *row_unsorted = nullptr;
        int *col_unsorted = nullptr;
        double *val_unsorted = nullptr;

        if (mm_read_unsymmetric_sparse<double, int>(
                matrix_file_name.c_str(), &nrows, &ncols, &nnz, &val_unsorted,
                &row_unsorted, &col_unsorted) < 0) {
            throw std::runtime_error("Error reading matrix from file: " +
                                     matrix_file_name);
        }

        if (nrows != ncols) {
            throw std::runtime_error("Matrix must be square.");
        }

        bool symm_flag = mm_is_symmetric(matcode);

        std::vector<int> row_data, col_data;
        std::vector<double> val_data;

        // Unpacks symmetric matrices
        // TODO: You should be able to work with symmetric matrices!
        if (symm_flag) {
            for (int i = 0; i < nnz; ++i) {
                row_data.push_back(row_unsorted[i]);
                col_data.push_back(col_unsorted[i]);
                val_data.push_back(val_unsorted[i]);
                if (row_unsorted[i] != col_unsorted[i]) {
                    row_data.push_back(col_unsorted[i]);
                    col_data.push_back(row_unsorted[i]);
                    val_data.push_back(val_unsorted[i]);
                }
            }
            free(row_unsorted);
            free(col_unsorted);
            free(val_unsorted);
            nnz = static_cast<ULL>(val_data.size());
        } else {
            row_data.assign(row_unsorted, row_unsorted + nnz);
            col_data.assign(col_unsorted, col_unsorted + nnz);
            val_data.assign(val_unsorted, val_unsorted + nnz);
            free(row_unsorted);
            free(col_unsorted);
            free(val_unsorted);
        }

        std::vector<int> perm(nnz);
        std::iota(perm.begin(), perm.end(), 0);
        sort_perm(row_data.data(), perm.data(), nnz);

        this->I.resize(nnz);
        this->J.resize(nnz);
        this->val.resize(nnz);

        for (int i = 0; i < nnz; ++i) {
            this->I[i] = row_data[perm[i]];
            this->J[i] = col_data[perm[i]];
            this->val[i] = val_data[perm[i]];
        }

        this->n_rows = nrows;
        this->n_cols = ncols;
        this->nnz = nnz;
        this->is_sorted = 1;    // TODO: verify
        this->is_symmetric = 0; // TODO: determine based on matcode?
#endif

#ifdef DEBUG_MODE
        std::cout << "Completed reading matrix from file: " << matrix_file_name
                  << std::endl;
#endif
    }

  private:
    std::vector<IT> I_;
    std::vector<IT> J_;
    std::vector<VT> val_;
};

#endif