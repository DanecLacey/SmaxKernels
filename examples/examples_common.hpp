#ifndef EXAMPLES_COMMON_HPP
#define EXAMPLES_COMMON_HPP

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

#define GET_THREAD_COUNT int num_threads = omp_get_max_threads();
#define GET_THREAD_ID int tid = omp_get_thread_num();

#else

#define GET_THREAD_COUNT int num_threads = 1;
#define GET_THREAD_ID int tid = 0;

#endif

#define REGISTER_SPMV_KERNEL(kernel_name, mat, X, Y)                           \
    smax->register_kernel(kernel_name, SMAX::SPMV, SMAX::CPU);                 \
    smax->kernels[kernel_name]->register_A(mat->n_rows, mat->n_cols, mat->nnz, \
                                           &mat->col, &mat->row_ptr,           \
                                           &mat->values);                      \
    smax->kernels[kernel_name]->register_B(mat->n_cols, &X->values);           \
    smax->kernels[kernel_name]->register_C(mat->n_rows, &Y->values);

#define REGISTER_SPMM_KERNEL(kernel_name, mat, n_vectors, X, Y)                \
    smax->register_kernel(kernel_name, SMAX::SPMM, SMAX::CPU);                 \
    smax->kernels[kernel_name]->register_A(mat->n_rows, mat->n_cols, mat->nnz, \
                                           &mat->col, &mat->row_ptr,           \
                                           &mat->values);                      \
    smax->kernels[kernel_name]->register_B(mat->n_cols, n_vectors,             \
                                           &X->values);                        \
    smax->kernels[kernel_name]->register_C(mat->n_rows, n_vectors, &Y->values);

#define REGISTER_SPTRSV_KERNEL(kernel_name, mat, X, Y)                         \
    smax->register_kernel(kernel_name, SMAX::SPTRSV, SMAX::CPU);               \
    smax->kernels[kernel_name]->register_A(mat->n_rows, mat->n_cols, mat->nnz, \
                                           &mat->col, &mat->row_ptr,           \
                                           &mat->values);                      \
    smax->kernels[kernel_name]->register_B(mat->n_cols, &X->values);           \
    smax->kernels[kernel_name]->register_C(mat->n_rows, &Y->values);

#define DIFF_STATUS_MACRO(relative_diff, working_file)                         \
    do {                                                                       \
        if ((std::abs(relative_diff) > 0.01) || std::isinf(relative_diff)) {   \
            working_file << std::left << std::setw(PRINT_WIDTH) << "ERROR";    \
        } else if (std::abs(relative_diff) > 0.0001) {                         \
            working_file << std::left << std::setw(PRINT_WIDTH) << "WARNING";  \
        }                                                                      \
    } while (0)

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
    std::vector<double> values;

    void write_to_mtx(int my_rank, std::string file_out_name) {
        std::string file_name =
            file_out_name + "_rank_" + std::to_string(my_rank) + ".mtx";

        for (int nz_idx = 0; nz_idx < nnz; ++nz_idx) {
            ++I[nz_idx];
            ++J[nz_idx];
        }

        char arg_str[] = "MCRG";

        mm_write_mtx_crd(&file_name[0], n_rows, n_cols, nnz, &(I)[0], &(J)[0],
                         &(values)[0],
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
        this->values.resize(nnz);

        for (int i = 0; i < nnz; ++i) {
            this->I[i] = row_data[perm[i]];
            this->J[i] = col_data[perm[i]];
            this->values[i] = val_data[perm[i]];
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
            std::cout << this->values[i] << " ";

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
    double *values;
    int *col;
    int *row_ptr;

    CRSMatrix() {
        this->n_rows = 0;
        this->nnz = 0;
        this->n_cols = 0;

        values = nullptr;
        col = nullptr;
        row_ptr = nullptr;
    }

    CRSMatrix(int n_rows, int n_cols, int nnz) {
        this->n_rows = n_rows;
        this->nnz = nnz;
        this->n_cols = n_cols;

        values = new double[nnz];
        col = new int[nnz];
        row_ptr = new int[n_rows + 1];

        row_ptr[0] = 0;
    }

    ~CRSMatrix() {
        delete[] values;
        delete[] col;
        delete[] row_ptr;
    }

    void print(void) {
        std::cout << "NNZ: " << this->nnz << std::endl;
        std::cout << "N_rows: " << this->n_rows << std::endl;
        std::cout << "N_cols: " << this->n_cols << std::endl;

        std::cout << "Values: ";
        for (int i = 0; i < this->nnz; ++i)
            std::cout << this->values[i] << " ";

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
        this->values = new double[this->nnz];

        for (int idx = 0; idx < this->nnz; ++idx) {
            this->col[idx] = coo_mat->J[idx];
            this->values[idx] = coo_mat->values[idx];
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
    double *values;

    DenseMatrix() {
        this->n_rows = 0;
        this->n_cols = 0;
        values = nullptr;
    }

    DenseMatrix(int n_rows, int n_cols, double val) {
        this->n_rows = n_rows;
        this->n_cols = n_cols;

        values = new double[n_rows * n_cols];

        // Initialize all elements to val
        for (int i = 0; i < n_rows * n_cols; ++i) {
            values[i] = val;
        }
    }

    ~DenseMatrix() { delete[] values; }

    void print() {
        std::cout << "N_rows: " << this->n_rows << std::endl;
        std::cout << "N_cols: " << this->n_cols << std::endl;

        std::cout << "Values: ";
        for (int i = 0; i < this->n_rows * this->n_cols; ++i)
            std::cout << this->values[i] << " ";

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
    D_plus_L.values = new double[D_plus_L.nnz];
    D_plus_L.col = new int[D_plus_L.nnz];
    D_plus_L.row_ptr = new int[A.n_rows + 1];
    D_plus_L.row_ptr[0] = 0;
    D_plus_L.n_rows = A.n_rows;
    D_plus_L.n_cols = A.n_cols;

    U.values = new double[U.nnz];
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
            double val = A.values[idx];

            if (col <= i) {
                // Diagonal or lower triangular part (D + L)
                D_plus_L.values[D_plus_L_count] = val;
                D_plus_L.col[D_plus_L_count++] = col;
            } else {
                // Strictly upper triangular part (U)
                U.values[U_count] = val;
                U.col[U_count++] = col;
            }
        }

        // Update row pointers
        D_plus_L.row_ptr[i + 1] = D_plus_L_count;
        U.row_ptr[i + 1] = U_count;
    }
}

#endif // EXAMPLES_COMMON_HPP