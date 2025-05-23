#pragma once

#include "mmio.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <numeric>
#include <set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#define PRINT_WIDTH 18

#ifdef _OPENMP
#define GET_THREAD_COUNT int n_threads = omp_get_max_threads();
#define GET_THREAD_ID int tid = omp_get_thread_num();
#else
#define GET_THREAD_COUNT int n_threads = 1;
#define GET_THREAD_ID int tid = 0;
#endif

#define CHECK_MKL_STATUS(status, message)                                      \
    if ((status) != SPARSE_STATUS_SUCCESS) {                                   \
        fprintf(stderr, "ERROR: %s\n", (message));                             \
        exit(EXIT_FAILURE);                                                    \
    }

#ifdef DEBUG_MODE
#define DIFF_STATUS_MACRO(relative_diff, working_file)                         \
    do {                                                                       \
        if ((std::abs(relative_diff) > 0.01) || std::isinf(relative_diff)) {   \
            working_file << std::left << std::setw(PRINT_WIDTH) << "ERROR";    \
        } else if (std::abs(relative_diff) > 0.0001) {                         \
            working_file << std::left << std::setw(PRINT_WIDTH) << "WARNING";  \
        }                                                                      \
    } while (0)
#else
#define DIFF_STATUS_MACRO(relative_diff, working_file)                         \
    do {                                                                       \
        if (std::abs(relative_diff) > 0.0001)                                  \
            working_file << std::left << std::setw(PRINT_WIDTH) << "WARNING";  \
    } while (0)
#endif

#define UPDATE_MAX_DIFFS(i, y_MKL, y_SMAX, relative_diff, absolute_diff)       \
    do {                                                                       \
        if ((relative_diff) > max_relative_diff) {                             \
            max_relative_diff = (relative_diff);                               \
            max_relative_diff_elem_SMAX = (y_SMAX)[(i)];                       \
            max_relative_diff_elem_MKL = (y_MKL)[(i)];                         \
        }                                                                      \
        if ((absolute_diff) > max_absolute_diff) {                             \
            max_absolute_diff = (absolute_diff);                               \
            max_absolute_diff_elem_SMAX = (y_SMAX)[(i)];                       \
            max_absolute_diff_elem_MKL = (y_MKL)[(i)];                         \
        }                                                                      \
    } while (0)

#ifdef DEBUG_MODE
#define CHECK_MAX_DIFFS_AND_PRINT_ERROR_WARNING(                               \
    max_relative_diff, max_absolute_diff, working_file)                        \
    do {                                                                       \
        if ((((std::abs(max_relative_diff) > 0.01) ||                          \
              std::isnan(max_relative_diff) ||                                 \
              std::isinf(max_relative_diff)) ||                                \
             (std::isnan(max_absolute_diff) ||                                 \
              std::isinf(max_absolute_diff)))) {                               \
            working_file << std::left << std::setw(PRINT_WIDTH) << "ERROR";    \
        } else if (std::abs(max_relative_diff) > 0.0001) {                     \
            working_file << std::left << std::setw(PRINT_WIDTH) << "WARNING";  \
        }                                                                      \
    } while (0)
#else
#define CHECK_MAX_DIFFS_AND_PRINT_ERROR_WARNING(                               \
    max_relative_diff, max_absolute_diff, working_file)                        \
    do {                                                                       \
        if (std::abs(max_relative_diff) > 0.0001)                              \
            working_file << std::left << std::setw(PRINT_WIDTH) << "WARNING";  \
    } while (0)
#endif

inline void sort_perm(int *arr, int *perm, int len, bool rev = false) {
    if (rev == false) {
        std::stable_sort(perm + 0, perm + len, [&](const int &a, const int &b) {
            return (arr[a] < arr[b]);
        });
    } else {
        std::stable_sort(perm + 0, perm + len, [&](const int &a, const int &b) {
            return (arr[a] > arr[b]);
        });
    }
}

double compute_euclid_dist(const int n_rows, const double *y_SMAX,
                           const double *y_MKL) {
    double tmp = 0.0;

#pragma omp parallel for reduction(+ : tmp)
    for (int i = 0; i < n_rows; ++i) {
        tmp += (y_SMAX[i] - y_MKL[i]) * (y_SMAX[i] - y_MKL[i]);
    }

    return std::sqrt(tmp);
}

// DL 4.5.25 TODO: Probably a better idea to leave args_ private and add an
// accessor
class CliParser {
  public:
    struct CliArgs {
        std::string matrix_file_name;
        virtual ~CliArgs() = default;
    };

  protected:
    CliArgs *args_;

  public:
    CliParser() : args_(nullptr) {}
    virtual ~CliParser() { delete args_; }

    virtual CliArgs *parse(int argc, char *argv[]) {
        // supress warnings
        (void)argc;
        (void)argv;

        delete args_;
        args_ = new CliArgs();
        return args_;
    }

    CliArgs *args() const { return args_; }
};

struct COOMatrix {
    long n_rows{};
    long n_cols{};
    long nnz{};

    bool is_sorted{};
    bool is_symmetric{};

    std::vector<int> I;
    std::vector<int> J;
    std::vector<double> val;

    void write_to_mtx(int my_rank, std::string file_out_name) {
        std::string file_name =
            file_out_name + "_rank_" + std::to_string(my_rank) + ".mtx";

        for (int nz_idx = 0; nz_idx < nnz; ++nz_idx) {
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
            nnz = static_cast<int>(val_data.size());
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
    }

    void print(void) {
        std::cout << "is_sorted = " << this->is_sorted << std::endl;
        std::cout << "is_symmetric = " << this->is_symmetric << std::endl;
        std::cout << "NNZ: " << this->nnz << std::endl;
        std::cout << "N_rows: " << this->n_rows << std::endl;
        std::cout << "N_cols: " << this->n_cols << std::endl;

        std::cout << "Values: ";
        for (int i = 0; i < this->nnz; ++i)
            std::cout << this->val[i] << " ";

        std::cout << "\nCol: ";
        for (int i = 0; i < this->nnz; ++i)
            std::cout << this->J[i] << " ";

        std::cout << "\nRow: ";
        for (int i = 0; i < this->nnz; ++i)
            std::cout << this->I[i] << " ";

        std::cout << std::endl;
        std::cout << std::endl;
    }
};

struct CRSMatrix {
    int nnz;
    int n_rows;
    int n_cols;
    double *val;
    int *col;
    int *row_ptr;

    CRSMatrix() {
        this->n_rows = 0;
        this->nnz = 0;
        this->n_cols = 0;

        val = nullptr;
        col = nullptr;
        row_ptr = nullptr;
    }

    CRSMatrix(int n_rows, int n_cols, int nnz) {
        this->n_rows = n_rows;
        this->nnz = nnz;
        this->n_cols = n_cols;

        val = new double[nnz];
        col = new int[nnz];
        row_ptr = new int[n_rows + 1];

        row_ptr[0] = 0;
    }

    ~CRSMatrix() {
        delete[] val;
        delete[] col;
        delete[] row_ptr;
    }

    void write_to_mtx_file(std::string file_out_name) {
        // Convert csr back to coo for mtx format printing
        std::vector<int> temp_rows(nnz);
        std::vector<int> temp_cols(nnz);
        std::vector<double> temp_values(nnz);

        int elem_num = 0;
        for (int row = 0; row < n_rows; ++row) {
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
        for (int i = 0; i < this->nnz; ++i)
            std::cout << this->val[i] << " ";

        std::cout << "\nCol Indices: ";
        for (int i = 0; i < this->nnz; ++i)
            std::cout << this->col[i] << " ";

        std::cout << "\nRow Ptr: ";
        for (int i = 0; i < this->n_rows + 1; ++i)
            std::cout << this->row_ptr[i] << " ";

        std::cout << std::endl;
        std::cout << std::endl;
    }

    void convert_coo_to_crs(COOMatrix *coo_mat) {
        this->n_rows = coo_mat->n_rows;
        this->n_cols = coo_mat->n_cols;
        this->nnz = coo_mat->nnz;

        this->row_ptr = new int[this->n_rows + 1];
        int *nnz_per_row = new int[this->n_rows];

        this->col = new int[this->nnz];
        this->val = new double[this->nnz];

        for (int idx = 0; idx < this->nnz; ++idx) {
            this->col[idx] = coo_mat->J[idx];
            this->val[idx] = coo_mat->val[idx];
        }

        for (int i = 0; i < this->n_rows; ++i) {
            nnz_per_row[i] = 0;
        }

        // count nnz per row
        for (int i = 0; i < this->nnz; ++i) {
            ++nnz_per_row[coo_mat->I[i]];
        }

        this->row_ptr[0] = 0;
        for (int i = 0; i < this->n_rows; ++i) {
            this->row_ptr[i + 1] = this->row_ptr[i] + nnz_per_row[i];
        }

        if (this->row_ptr[this->n_rows] != this->nnz) {
            printf("ERROR: converting to CRS.\n");
            exit(1);
        }

        delete[] nnz_per_row;
    }
};

struct DenseMatrix {
    int n_rows;
    int n_cols;
    double *val;

    DenseMatrix() {
        this->n_rows = 0;
        this->n_cols = 0;
        val = nullptr;
    }

    DenseMatrix(int n_rows, int n_cols, double _val) {
        this->n_rows = n_rows;
        this->n_cols = n_cols;

        val = new double[n_rows * n_cols];

        // Initialize all elements to val
        for (int i = 0; i < n_rows * n_cols; ++i) {
            val[i] = _val;
        }
    }

    ~DenseMatrix() { delete[] val; }

    void print() {
        std::cout << "N_rows: " << this->n_rows << std::endl;
        std::cout << "N_cols: " << this->n_cols << std::endl;

        std::cout << "Values: ";
        for (int i = 0; i < this->n_rows * this->n_cols; ++i)
            std::cout << this->val[i] << " ";

        std::cout << std::endl;
        std::cout << std::endl;
    }
};

void extract_D_L_U(const CRSMatrix &A, CRSMatrix &D_plus_L, CRSMatrix &U) {
    // Count nnz
    for (int i = 0; i < A.n_rows; ++i) {
        int row_start = A.row_ptr[i];
        int row_end = A.row_ptr[i + 1];

        // Loop over each non-zero entry in the current row
        for (int idx = row_start; idx < row_end; ++idx) {
            int col = A.col[idx];

            if (col <= i) {
                ++D_plus_L.nnz;
            } else {
                ++U.nnz;
            }
        }
    }

    // Allocate heap space and assign known metadata
    D_plus_L.val = new double[D_plus_L.nnz];
    D_plus_L.col = new int[D_plus_L.nnz];
    D_plus_L.row_ptr = new int[A.n_rows + 1];
    D_plus_L.row_ptr[0] = 0;
    D_plus_L.n_rows = A.n_rows;
    D_plus_L.n_cols = A.n_cols;

    U.val = new double[U.nnz];
    U.col = new int[U.nnz];
    U.row_ptr = new int[A.n_rows + 1];
    U.row_ptr[0] = 0;
    U.n_rows = A.n_rows;
    U.n_cols = A.n_cols;

    // Assign nonzeros
    int D_plus_L_count = 0;
    int U_count = 0;
    for (int i = 0; i < A.n_rows; ++i) {
        int row_start = A.row_ptr[i];
        int row_end = A.row_ptr[i + 1];

        // Loop over each non-zero entry in the current row
        for (int idx = row_start; idx < row_end; ++idx) {
            int col = A.col[idx];
            double val = A.val[idx];

            if (col <= i) {
                // Diagonal or lower triangular part (D + L)
                D_plus_L.val[D_plus_L_count] = val;
                D_plus_L.col[D_plus_L_count++] = col;
            } else {
                // Strictly upper triangular part (U)
                U.val[U_count] = val;
                U.col[U_count++] = col;
            }
        }

        // Update row pointers
        D_plus_L.row_ptr[i + 1] = D_plus_L_count;
        U.row_ptr[i + 1] = U_count;
    }
}

void extract_D_L_U_arrays(int A_n_rows, int A_n_cols, int A_nnz, int *A_row_ptr,
                          int *A_col, double *A_val, int &D_plus_L_n_rows,
                          int &D_plus_L_n_cols, int &D_plus_L_nnz,
                          int *&D_plus_L_row_ptr, int *&D_plus_L_col,
                          double *&D_plus_L_val, int &U_n_rows, int &U_n_cols,
                          int &U_nnz, int *&U_row_ptr, int *&U_col,
                          double *&U_val) {

    // supress warnings
    (void)A_nnz;

    // Count nnz
    for (int i = 0; i < A_n_rows; ++i) {
        int row_start = A_row_ptr[i];
        int row_end = A_row_ptr[i + 1];

        // Loop over each non-zero entry in the current row
        for (int idx = row_start; idx < row_end; ++idx) {
            int col = A_col[idx];

            if (col <= i) {
                ++D_plus_L_nnz;
            } else {
                ++U_nnz;
            }
        }
    }

    // Allocate heap space and assign known metadata
    D_plus_L_val = new double[D_plus_L_nnz];
    D_plus_L_col = new int[D_plus_L_nnz];
    D_plus_L_row_ptr = new int[A_n_rows + 1];
    D_plus_L_row_ptr[0] = 0;
    D_plus_L_n_rows = A_n_rows;
    D_plus_L_n_cols = A_n_cols;

    U_val = new double[U_nnz];
    U_col = new int[U_nnz];
    U_row_ptr = new int[A_n_rows + 1];
    U_row_ptr[0] = 0;
    U_n_rows = A_n_rows;
    U_n_cols = A_n_cols;

    // Assign nonzeros
    int D_plus_L_count = 0;
    int U_count = 0;
    for (int i = 0; i < A_n_rows; ++i) {
        int row_start = A_row_ptr[i];
        int row_end = A_row_ptr[i + 1];

        // Loop over each non-zero entry in the current row
        for (int idx = row_start; idx < row_end; ++idx) {
            int col = A_col[idx];
            double val = A_val[idx];

            if (col <= i) {
                // Diagonal or lower triangular part (D + L)
                D_plus_L_val[D_plus_L_count] = val;
                D_plus_L_col[D_plus_L_count++] = col;
            } else {
                // Strictly upper triangular part (U)
                U_val[U_count] = val;
                U_col[U_count++] = col;
            }
        }

        // Update row pointers
        D_plus_L_row_ptr[i + 1] = D_plus_L_count;
        U_row_ptr[i + 1] = U_count;
    }
}

template <typename VT> void print_vector(VT *vec, int n_rows) {
    printf("Vector: [");
    for (int i = 0; i < n_rows; ++i) {
        std::cout << vec[i] << ", ";
    }
    printf("]\n\n");
}

template <typename IT, typename VT>
void print_matrix(int n_rows, int n_cols, int nnz, IT *col, IT *row_ptr,
                  VT *val, bool symbolic = false) {
    printf("n_rows = %i\n", n_rows);
    printf("n_cols = %i\n", n_cols);
    printf("nnz = %i\n", nnz);
    printf("col = [");
    for (int i = 0; i < nnz; ++i) {
        std::cout << col[i] << ", ";
    }
    printf("]\n");

    printf("row_ptr = [");
    for (int i = 0; i <= n_rows; ++i) {
        std::cout << row_ptr[i] << ", ";
    }
    printf("]\n");

    if (!symbolic) {
        printf("val = [");
        for (int i = 0; i < nnz; ++i) {
            std::cout << val[i] << ", ";
        }
        printf("]\n");
    }
    printf("\n");
}