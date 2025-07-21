#pragma once

#include "SmaxKernels/interface.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <numeric>
#include <set>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_FAST_MMIO
#include "mmio.hpp"
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <fstream>
namespace fmm = fast_matrix_market;
#else
#include "mmio.hpp"
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

using ULL = unsigned long long int;

#define PRINT_WIDTH 18
#define CACHE_LINE_ALIGNMENT 64

#ifdef _OPENMP
#define GET_THREAD_COUNT ULL n_threads = omp_get_max_threads();
#define GET_THREAD_ID ULL tid = omp_get_thread_num();
#else
#define GET_THREAD_COUNT ULL n_threads = 1;
#define GET_THREAD_ID ULL tid = 0;
#endif

#define CHECK_MKL_STATUS(status, message)                                      \
    if ((status) != SPARSE_STATUS_SUCCESS) {                                   \
        fprintf(stderr, "ERROR: %s\n", (message));                             \
        exit(EXIT_FAILURE);                                                    \
    }

#define CUDA_CHECK(cmd)                                                        \
    do {                                                                       \
        cudaError_t e = cmd;                                                   \
        if (e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(e));                                    \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

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

void *aligned_malloc(size_t bytesize) {
    int errorCode;
    void *ptr;

    errorCode = posix_memalign(&ptr, CACHE_LINE_ALIGNMENT, bytesize);

    if (errorCode) {
        if (errorCode == EINVAL) {
            fprintf(stderr,
                    "Error: Alignment parameter is not a power of two\n");
            exit(EXIT_FAILURE);
        }
        if (errorCode == ENOMEM) {
            fprintf(stderr,
                    "Error: Insufficient memory to fulfill the request\n");
            exit(EXIT_FAILURE);
        }
    }

    if (ptr == NULL) {
        fprintf(stderr, "Error: posix_memalign failed!\n");
        exit(EXIT_FAILURE);
    }

    return ptr;
}

// overload new and delete for alignement
void *operator new(size_t bytesize) {
    // printf("Overloading new operator with size: %lu\n", bytesize);
    int errorCode;
    void *ptr;
    errorCode = posix_memalign(&ptr, CACHE_LINE_ALIGNMENT, bytesize);

    if (errorCode) {
        if (errorCode == EINVAL) {
            fprintf(stderr,
                    "Error: Alignment parameter is not a power of two\n");
            exit(EXIT_FAILURE);
        }
        if (errorCode == ENOMEM) {
            fprintf(stderr, "Error: Insufficient memory to fulfill the request "
                            "for space\n");
            exit(EXIT_FAILURE);
        }
    }

    if (ptr == NULL) {
        fprintf(stderr, "Error: posix_memalign failed!\n");
        exit(EXIT_FAILURE);
    }

    return ptr;
}

void operator delete(void *p) {
    // printf("Overloading delete operator\n");
    free(p);
}

/*
Only type supported by this helper function at this time:
- int, long int, long long int, unsigned long long int
- float, double
*/
// TODO: Clean the mess up!
template <typename IT, typename VT>
void register_kernel(SMAX::Interface *smax, std::string kernel_name,
                     SMAX::KernelType KernelType,
                     SMAX::PlatformType PlatformType) {
    if constexpr (std::is_same_v<IT, int>) {
        if constexpr (std::is_same_v<VT, float>) {
            smax->register_kernel(kernel_name.c_str(), KernelType, PlatformType,
                                  SMAX::IntType::INT32,
                                  SMAX::FloatType::FLOAT32);
        } else if constexpr (std::is_same_v<VT, double>) {
            smax->register_kernel(kernel_name.c_str(), KernelType, PlatformType,
                                  SMAX::IntType::INT32,
                                  SMAX::FloatType::FLOAT64);
        } else {
            std::cout << "VT not recognized" << std::endl;
        }
    } else if constexpr (std::is_same_v<IT, long>) {
        if constexpr (std::is_same_v<VT, float>) {
            smax->register_kernel(kernel_name.c_str(), KernelType, PlatformType,
                                  SMAX::IntType::INT64,
                                  SMAX::FloatType::FLOAT32);
        } else if constexpr (std::is_same_v<VT, double>) {
            smax->register_kernel(kernel_name.c_str(), KernelType, PlatformType,
                                  SMAX::IntType::INT64,
                                  SMAX::FloatType::FLOAT64);
        } else {
            std::cout << "VT not recognized" << std::endl;
        }
    } else if constexpr (std::is_same_v<IT, unsigned int>) {
        if constexpr (std::is_same_v<VT, float>) {
            smax->register_kernel(kernel_name.c_str(), KernelType, PlatformType,
                                  SMAX::IntType::UINT32,
                                  SMAX::FloatType::FLOAT32);
        } else if constexpr (std::is_same_v<VT, double>) {
            smax->register_kernel(kernel_name.c_str(), KernelType, PlatformType,
                                  SMAX::IntType::UINT32,
                                  SMAX::FloatType::FLOAT64);
        } else {
            std::cout << "VT not recognized" << std::endl;
        }
    } else if constexpr (std::is_same_v<IT, long long>) {
        if constexpr (std::is_same_v<VT, float>) {
            smax->register_kernel(kernel_name.c_str(), KernelType, PlatformType,
                                  SMAX::IntType::INT64,
                                  SMAX::FloatType::FLOAT32);
        } else if constexpr (std::is_same_v<VT, double>) {
            smax->register_kernel(kernel_name.c_str(), KernelType, PlatformType,
                                  SMAX::IntType::INT64,
                                  SMAX::FloatType::FLOAT64);
        } else {
            std::cout << "VT not recognized" << std::endl;
        }
    } else if constexpr (std::is_same_v<IT, unsigned long long>) {
        if constexpr (std::is_same_v<VT, float>) {
            smax->register_kernel(kernel_name.c_str(), KernelType, PlatformType,
                                  SMAX::IntType::UINT64,
                                  SMAX::FloatType::FLOAT32);
        } else if constexpr (std::is_same_v<VT, double>) {
            smax->register_kernel(kernel_name.c_str(), KernelType, PlatformType,
                                  SMAX::IntType::UINT64,
                                  SMAX::FloatType::FLOAT64);
        } else {
            std::cout << "VT not recognized" << std::endl;
        }
    } else {
        std::cout << "IT not recognized" << std::endl;
    }
};

template <typename IT>
std::vector<IT> compute_sort_permutation(const std::vector<IT> &rows,
                                         const std::vector<IT> &cols) {
    std::vector<IT> perm(rows.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::stable_sort(perm.begin(), perm.end(),
                     [&](std::size_t i, std::size_t j) {
                         if (rows[i] != rows[j])
                             return rows[i] < rows[j];
                         if (cols[i] != cols[j])
                             return cols[i] < cols[j];
                         return false;
                     });
    return perm;
}

template <typename IT, typename VT>
std::vector<VT> apply_permutation(std::vector<IT> &perm,
                                  std::vector<VT> &original) {
    std::vector<VT> sorted;
    sorted.reserve(original.size());
    std::transform(perm.begin(), perm.end(), std::back_inserter(sorted),
                   [&](auto i) { return original[i]; });
    original = std::vector<VT>();
    return sorted;
}

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

double compute_euclid_dist(const ULL n_rows, const double *y_SMAX,
                           const double *y_MKL) {
    double tmp = 0.0;

#pragma omp parallel for reduction(+ : tmp)
    for (ULL i = 0; i < n_rows; ++i) {
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

template <typename IT, typename VT>
void extract_D_L_U(const CRSMatrix<IT, VT> &A, CRSMatrix<IT, VT> &D_plus_L,
                   CRSMatrix<IT, VT> &U) {
    
    // Clear data from targets
    if (D_plus_L.row_ptr != nullptr)
        delete[] D_plus_L.row_ptr;
    if (D_plus_L.col != nullptr)
        delete[] D_plus_L.col;
    if (D_plus_L.val != nullptr)
        delete[] D_plus_L.val;
    if (U.row_ptr != nullptr)
        delete[] U.row_ptr;
    if (U.col != nullptr)
        delete[] U.col;
    if (U.val != nullptr)
        delete[] U.val;
    D_plus_L.nnz = 0;
    U.nnz = 0;

    // Count nnz
    for (ULL i = 0; i < A.n_rows; ++i) {
        IT row_start = A.row_ptr[i];
        IT row_end = A.row_ptr[i + 1];

        // Loop over each non-zero entry in the current row
        for (IT idx = row_start; idx < row_end; ++idx) {
            IT col = A.col[idx];

            if (static_cast<ULL>(col) <= i) {
                ++D_plus_L.nnz;
            } else {
                ++U.nnz;
            }
        }
    }

    // Allocate heap space and assign known metadata
    D_plus_L.val = new VT[D_plus_L.nnz];
    D_plus_L.col = new IT[D_plus_L.nnz];
    D_plus_L.row_ptr = new IT[A.n_rows + 1];
    D_plus_L.row_ptr[0] = 0;
    D_plus_L.n_rows = A.n_rows;
    D_plus_L.n_cols = A.n_cols;

    U.val = new VT[U.nnz];
    U.col = new IT[U.nnz];
    U.row_ptr = new IT[A.n_rows + 1];
    U.row_ptr[0] = 0;
    U.n_rows = A.n_rows;
    U.n_cols = A.n_cols;

    // Assign nonzeros
    ULL D_plus_L_count = 0;
    ULL U_count = 0;
    for (ULL i = 0; i < A.n_rows; ++i) {
        IT row_start = A.row_ptr[i];
        IT row_end = A.row_ptr[i + 1];

        // Loop over each non-zero entry in the current row
        for (IT idx = row_start; idx < row_end; ++idx) {
            IT col = A.col[idx];
            VT val = A.val[idx];

            if (static_cast<ULL>(col) <= i) {
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

template <typename VT> void print_vector(VT *vec, ULL n_rows) {
    printf("Vector: [");
    for (ULL i = 0; i < n_rows; ++i) {
        std::cout << vec[i] << ", ";
    }
    printf("]\n\n");
}

template <typename VT> void print_exact_vector(VT *vec, ULL n_rows) {
#include <iomanip>
#include <iostream>
#include <limits>

    printf("Vector: [");
    for (ULL i = 0; i < n_rows; ++i) {
        std::cout << std::fixed
                  << std::setprecision(
                         std::numeric_limits<double>::max_digits10)
                  << vec[i] << std::endl;
    }
    printf("]\n\n");
}

template <typename IT, typename VT>
void print_matrix(ULL n_rows, ULL n_cols, ULL nnz, IT *col, IT *row_ptr,
                  VT *val, bool symbolic = false) {

    std::cout << "n_rows = " << n_rows << std::endl;
    std::cout << "n_cols = " << n_cols << std::endl;
    std::cout << "nnz = " << nnz << std::endl;

    printf("col = [");
    for (ULL i = 0; i < nnz; ++i) {
        std::cout << col[i] << ", ";
    }
    printf("]\n");

    printf("row_ptr = [");
    for (ULL i = 0; i <= n_rows; ++i) {
        std::cout << row_ptr[i] << ", ";
    }
    printf("]\n");

    if (!symbolic) {
        printf("val = [");
        for (ULL i = 0; i < nnz; ++i) {
            std::cout << val[i] << ", ";
        }
        printf("]\n");
    }
    printf("\n");
}

// Just for unit tests. In practice, we leave nonzeros in a row unsorted
template <typename IT, typename ST>
void sort_csr_rows_by_col(IT *row_ptr, IT *col, ST n_rows, ST nnz) {
    for (ST i = 0; i < n_rows; ++i) {
        IT *row_start = col + row_ptr[i];
        IT *row_end = col + row_ptr[i + 1];

        std::sort(row_start, row_end);
    }
};
