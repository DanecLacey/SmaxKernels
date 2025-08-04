#include <iostream>
#include <numeric>
#include <vector>

#ifdef USE_FAST_MMIO
#include "mmio.hpp"
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <fstream>
namespace fmm = fast_matrix_market;
#else
#include "mmio.hpp"
#endif

#include "permutation_helpers.hpp"

using ULL = unsigned long long int;

struct COOMatrix {
    ULL n_rows{};
    ULL n_cols{};
    ULL nnz{};

    bool is_sorted{};
    bool is_symmetric{};

    std::vector<int> I;
    std::vector<int> J;
    std::vector<double> val;

    COOMatrix()
        : n_rows(0), n_cols(0), nnz(0), is_sorted(false), is_symmetric(false),
          I(), J(), val() {}

    void write_to_mtx(int my_rank, std::string file_out_name) {
        std::string file_name =
            file_out_name + "_rank_" + std::to_string(my_rank) + ".mtx";

        for (ULL nz_idx = 0; nz_idx < nnz; ++nz_idx) {
            ++I[nz_idx];
            ++J[nz_idx];
        }

        char arg_str[] = "MCRG";

        mm_write_mtx_crd(&file_name[0], n_rows, n_cols, nnz, &(I)[0], &(J)[0],
                         &(val)[0],
                         arg_str // TODO: <- make more general, i.e. flexible
                                 // based on the matrix. Read from original mtx?
        );
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

    void print(void) {
        std::cout << "is_sorted = " << this->is_sorted << std::endl;
        std::cout << "is_symmetric = " << this->is_symmetric << std::endl;
        std::cout << "NNZ: " << this->nnz << std::endl;
        std::cout << "N_rows: " << this->n_rows << std::endl;
        std::cout << "N_cols: " << this->n_cols << std::endl;

        std::cout << "Values: ";
        for (ULL i = 0; i < this->nnz; ++i)
            std::cout << this->val[i] << " ";

        std::cout << "\nCol: ";
        for (ULL i = 0; i < this->nnz; ++i)
            std::cout << this->J[i] << " ";

        std::cout << "\nRow: ";
        for (ULL i = 0; i < this->nnz; ++i)
            std::cout << this->I[i] << " ";

        std::cout << std::endl;
        std::cout << std::endl;
    }
};

template <typename IT, typename VT> struct CRSMatrix {
    ULL nnz;
    ULL n_rows;
    ULL n_cols;
    VT *val;
    IT *col;
    IT *row_ptr;

    CRSMatrix() {
        this->n_rows = 0;
        this->nnz = 0;
        this->n_cols = 0;

        val = nullptr;
        col = nullptr;
        row_ptr = nullptr;
    }

    CRSMatrix(ULL n_rows, ULL n_cols, ULL nnz) {
        this->n_rows = n_rows;
        this->nnz = nnz;
        this->n_cols = n_cols;

        // TODO: Actually shouldn't be necessary
        val = new VT[nnz];
        col = new IT[nnz];
        row_ptr = new IT[n_rows + 1];
    }

    // --- Copy assignment operator ---
    CRSMatrix &operator=(CRSMatrix const &other) {
        if (this != &other) {
            // 1) Free existing storage
            delete[] val;
            delete[] col;
            delete[] row_ptr;

            // 2) Copy sizes
            nnz = other.nnz;
            n_rows = other.n_rows;
            n_cols = other.n_cols;

            // 3) Allocate new storage
            if (nnz > 0) {
                val = new VT[nnz];
                col = new IT[nnz];
                row_ptr = new IT[n_rows + 1];

                // 4) Copy data
                std::copy(other.val, other.val + nnz, val);
                std::copy(other.col, other.col + nnz, col);
                std::copy(other.row_ptr, other.row_ptr + n_rows + 1, row_ptr);
            } else {
                val = col = row_ptr = nullptr;
            }
        }
        return *this;
    }

    ~CRSMatrix() {
        delete[] val;
        delete[] col;
        delete[] row_ptr;
    }

    // Useful for benchmarking
    void clear() {
        delete[] val;
        delete[] col;
        delete[] row_ptr;
        val = nullptr;
        col = nullptr;
        row_ptr = nullptr;
        nnz = 0;
    }

    void write_to_mtx_file(std::string file_out_name) {
        // Convert csr back to coo for mtx format printing
        std::vector<int> temp_rows(nnz);
        std::vector<int> temp_cols(nnz);
        std::vector<double> temp_values(nnz);

        ULL elem_num = 0;
        for (ULL row = 0; row < n_rows; ++row) {
            for (int idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx) {
                temp_rows[elem_num] =
                    row + 1; // +1 to adjust for 1 based indexing in mm-format
                temp_cols[elem_num] = col[idx] + 1;
                temp_values[elem_num] = val[idx];
                ++elem_num;
            }
        }

        std::string file_name = file_out_name + "_out_matrix.mtx";

        mm_write_mtx_crd(
            &file_name[0], n_rows, n_cols, nnz, &(temp_rows)[0],
            &(temp_cols)[0], &(temp_values)[0],
            const_cast<char *>("MCRG") // TODO: <- make more general
        );
    }

    void print(void) {
        std::cout << "NNZ: " << this->nnz << std::endl;
        std::cout << "N_rows: " << this->n_rows << std::endl;
        std::cout << "N_cols: " << this->n_cols << std::endl;

        std::cout << "Values: ";
        for (ULL i = 0; i < this->nnz; ++i)
            std::cout << this->val[i] << " ";

        std::cout << "\nCol Indices: ";
        for (ULL i = 0; i < this->nnz; ++i)
            std::cout << this->col[i] << " ";

        std::cout << "\nRow Ptr: ";
        for (ULL i = 0; i < this->n_rows + 1; ++i)
            std::cout << this->row_ptr[i] << " ";

        std::cout << std::endl;
        std::cout << std::endl;
    }

    void convert_coo_to_crs(COOMatrix *coo_mat) {
        this->n_rows = coo_mat->n_rows;
        this->n_cols = coo_mat->n_cols;
        this->nnz = coo_mat->nnz;

        this->row_ptr = new IT[this->n_rows + 1];
        ULL *tmp = new ULL[this->n_rows + 1];
        ULL *nnz_per_row = new ULL[this->n_rows];

        this->col = new IT[this->nnz];
        this->val = new VT[this->nnz];

#pragma omp parallel
        {
#pragma omp for schedule(static)
            for (ULL idx = 0; idx < this->nnz; ++idx) {
                this->col[idx] = coo_mat->J[idx];
                this->val[idx] = coo_mat->val[idx];
            }

#pragma omp for schedule(static)
            for (ULL i = 0; i < this->n_rows; ++i) {
                nnz_per_row[i] = 0;
            }
        }

        // count nnz per row
        for (ULL i = 0; i < this->nnz; ++i) {
            ++nnz_per_row[coo_mat->I[i]];
        }

        tmp[0] = 0;
        for (ULL i = 0; i < this->n_rows; ++i) {
            tmp[i + 1] = tmp[i] + nnz_per_row[i];
        }

#pragma omp parallel for schedule(static)
        for (ULL i = 0; i < this->n_rows + 1; ++i) {
            this->row_ptr[i] = tmp[i];
        }

        if (static_cast<ULL>(this->row_ptr[this->n_rows]) != this->nnz) {
            printf("ERROR: expected nnz: %lld does not match: %lld in "
                   "convert_coo_to_crs.\n",
                   static_cast<ULL>(this->row_ptr[this->n_rows]), this->nnz);
            exit(1);
        }

        delete[] nnz_per_row;
        delete[] tmp;
    }
};

template <typename IT, typename VT> struct BCRSMatrix {
    ULL n_blocks;
    ULL n_rows;
    ULL n_cols;
    ULL b_height;
    ULL b_width;
    ULL b_h_pad;
    ULL b_w_pad;
    VT *val;
    IT *col;
    IT *row_ptr;

    BCRSMatrix() {
        n_blocks = 0;
        n_rows = 0;
        n_cols = 0;
        b_height = 0;
        b_width = 0;
        b_h_pad = 0;
        b_w_pad = 0;
        val = nullptr;
        col = nullptr;
        row_ptr = nullptr;
    }

    BCRSMatrix(ULL _n_rows, ULL _n_cols, ULL _n_blocks, ULL _b_height,
               ULL _b_width, ULL _b_h_pad, ULL _b_w_pad) {
        n_rows = _n_rows;
        n_cols = _n_cols;
        n_blocks = _n_blocks;
        b_height = _b_height;
        b_width = _b_width;
        b_h_pad = _b_h_pad;
        b_w_pad = _b_w_pad;
        val = new VT[n_blocks * b_h_pad * b_w_pad];
        col = new IT[n_blocks];
        row_ptr = new IT[n_rows + 1];
    }

    ~BCRSMatrix() {
        delete[] val;
        delete[] col;
        delete[] row_ptr;
    }

    void print(void) {
        std::cout << "N_blocks: " << this->n_blocks << std::endl;
        std::cout << "N_rows: " << this->n_rows << std::endl;
        std::cout << "N_cols: " << this->n_cols << std::endl;
        std::cout << "b_height: " << this->b_height << std::endl;
        std::cout << "b_width: " << this->b_width << std::endl;
        std::cout << "b_h_pad: " << this->b_h_pad << std::endl;
        std::cout << "b_w_pad: " << this->b_w_pad << std::endl;

        std::cout << "Values: ";
        for (ULL i = 0; i < this->n_blocks; ++i) {
            for (ULL j = 0; j < this->b_h_pad; ++j) {
                for (ULL k = 0; k < this->b_w_pad; ++k) {
                    std::cout << this->val[i] << " ";
                }
            }
            std::cout << std::endl;
        }

        std::cout << "\nCol Indices: ";
        for (ULL i = 0; i < this->n_blocks; ++i)
            std::cout << this->col[i] << " ";

        std::cout << "\nRow Ptr: ";
        for (ULL i = 0; i < this->n_rows + 1; ++i)
            std::cout << this->row_ptr[i] << " ";

        std::cout << std::endl;
        std::cout << std::endl;
    }
};

template <typename IT, typename VT> struct SCSMatrix {
    ULL C;
    ULL sigma;
    ULL n_rows;
    ULL n_rows_padded;
    ULL n_cols;
    ULL n_chunks;
    ULL n_elements;
    ULL nnz;
    IT *chunk_ptr;
    IT *chunk_lengths;
    IT *col;
    VT *val;
    IT *perm;

    SCSMatrix() {
        C = 0;
        sigma = 0;
        n_rows = 0;
        n_rows_padded = 0;
        n_cols = 0;
        n_chunks = 0;
        n_elements = 0;
        nnz = 0;
        chunk_ptr = nullptr;
        chunk_lengths = nullptr;
        col = nullptr;
        val = nullptr;
        perm = nullptr;
    }

    SCSMatrix(ULL _C, ULL _sigma) {
        C = _C;
        sigma = _sigma;
        n_rows = 0;
        n_rows_padded = 0;
        n_cols = 0;
        n_chunks = 0;
        n_elements = 0;
        nnz = 0;
        chunk_ptr = nullptr;
        chunk_lengths = nullptr;
        col = nullptr;
        val = nullptr;
        perm = nullptr;
    }

    ~SCSMatrix() {
        delete[] chunk_ptr;
        delete[] chunk_lengths;
        delete[] col;
        delete[] val;
        delete[] perm;
    }

    // TODO: Adapt to SCS
    // void write_to_mtx_file(std::string file_out_name) {
    //     // Convert csr back to coo for mtx format printing
    //     std::vector<int> temp_rows(nnz);
    //     std::vector<int> temp_cols(nnz);
    //     std::vector<double> temp_values(nnz);

    //     ULL elem_num = 0;
    //     for (ULL row = 0; row < n_rows; ++row) {
    //         for (int idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx) {
    //             temp_rows[elem_num] =
    //                 row + 1; // +1 to adjust for 1 based indexing in
    //             mm - format temp_cols[elem_num] = col[idx] + 1;
    //             temp_values[elem_num] = val[idx];
    //             ++elem_num;
    //         }
    //     }

    //     std::string file_name = file_out_name + "_out_matrix.mtx";

    //     mm_write_mtx_crd(
    //         &file_name[0], n_rows, n_cols, nnz, &(temp_rows)[0],
    //         &(temp_cols)[0], &(temp_values)[0],
    //         const_cast<char *>("MCRG") // TODO: <- make more general
    //     );
    // }

    void print(void) {
        std::cout << "C: " << this->C << std::endl;
        std::cout << "sigma: " << this->sigma << std::endl;
        std::cout << "NNZ: " << this->nnz << std::endl;
        std::cout << "N_elements: " << this->n_elements << std::endl;
        std::cout << "N_rows: " << this->n_rows << std::endl;
        std::cout << "N_rows_padded: " << this->n_rows_padded << std::endl;
        std::cout << "N_cols: " << this->n_cols << std::endl;
        std::cout << "N_chunks: " << this->n_chunks << std::endl;

        std::cout << "Chunk_ptr: ";
        for (ULL i = 0; i < this->n_chunks + 1; ++i)
            std::cout << this->chunk_ptr[i] << " ";

        std::cout << "Chunk_lengths: ";
        for (ULL i = 0; i < this->n_chunks; ++i)
            std::cout << this->chunk_lengths[i] << " ";

        std::cout << "\nCol Indices: ";
        for (ULL i = 0; i < this->nnz; ++i)
            std::cout << this->col[i] << " ";

        std::cout << "\nVal: ";
        for (ULL i = 0; i < this->n_elements; ++i)
            std::cout << this->val[i] << " ";

        std::cout << "\nPerm: ";
        for (ULL i = 0; i < this->n_rows; ++i)
            std::cout << this->perm[i] << " ";

        std::cout << std::endl;
        std::cout << std::endl;
    }
};

template <typename VT> struct DenseMatrix {
    ULL n_rows;
    ULL n_cols;
    VT *val;

    DenseMatrix() {
        this->n_rows = 0;
        this->n_cols = 0;
        val = nullptr;
    }

    DenseMatrix(ULL n_rows, ULL n_cols, VT _val) {
        this->n_rows = n_rows;
        this->n_cols = n_cols;

        val = new VT[n_rows * n_cols];

// Initialize all elements to val
#pragma omp parallel for
        for (ULL i = 0; i < n_rows * n_cols; ++i) {
            val[i] = _val;
        }
    }

    ~DenseMatrix() { delete[] val; }

    DenseMatrix &operator-=(const DenseMatrix &mat) {
        for (ULL i = 0; i < n_cols * n_rows; i++) {
            val[i] = val[i] - mat.val[i];
        }
        return *this;
    }

    void print() {
        std::cout << "N_rows: " << this->n_rows << std::endl;
        std::cout << "N_cols: " << this->n_cols << std::endl;

        std::cout << "Values: ";
        for (ULL i = 0; i < this->n_rows * this->n_cols; ++i)
            std::cout << this->val[i] << " ";

        std::cout << std::endl;
        std::cout << std::endl;
    }
};