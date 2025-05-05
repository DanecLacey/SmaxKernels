#include "../examples_common.hpp"
#include "validation_common.hpp"

void compare_spmm(const int n_rows, const int n_vectors, const double *y_SMAX,
                  const double *y_MKL, const std::string mtx_name) {

    std::fstream working_file;
    std::string output_filename = "compare_spmm.txt";
    working_file.open(output_filename,
                      std::fstream::in | std::fstream::out | std::fstream::app);

    GET_THREAD_COUNT(int, num_threads);

    working_file << mtx_name << " with " << num_threads << " thread(s)"
                 << std::endl;

    double relative_diff, max_relative_diff, max_relative_diff_elem_SMAX,
        max_relative_diff_elem_MKL;
    relative_diff = max_relative_diff = max_relative_diff_elem_SMAX =
        max_relative_diff_elem_MKL = 0.0;
    double absolute_diff, max_absolute_diff, max_absolute_diff_elem_SMAX,
        max_absolute_diff_elem_MKL;
    absolute_diff = max_absolute_diff = max_absolute_diff_elem_SMAX =
        max_absolute_diff_elem_MKL = 0.0;

// Print header
#if VERBOSITY == 0
    working_file << std::left << std::setw(PRINT_WIDTH)
                 << "MKL rel. elem:" << std::left << std::setw(PRINT_WIDTH)
                 << "SMAX rel. elem:" << std::left << std::setw(PRINT_WIDTH)
                 << "lrgst rel. (%):" << std::left << std::setw(PRINT_WIDTH)
                 << "MKL abs. elem:" << std::left << std::setw(PRINT_WIDTH)
                 << "SMAX abs. elem:" << std::left << std::setw(PRINT_WIDTH)
                 << "lrgst abs.:" << std::left << std::setw(PRINT_WIDTH)
                 << "||MKL - SMAX||" << std::left << std::setw(PRINT_WIDTH)
                 << std::endl;

    working_file << std::left << std::setw(PRINT_WIDTH) << "-------------"
                 << std::left << std::setw(PRINT_WIDTH) << "---------------"
                 << std::left << std::setw(PRINT_WIDTH) << "-------------"
                 << std::left << std::setw(PRINT_WIDTH) << "-------------"
                 << std::left << std::setw(PRINT_WIDTH) << "---------------"
                 << std::left << std::setw(PRINT_WIDTH) << "-----------"
                 << std::left << std::setw(PRINT_WIDTH) << "---------------"
                 << std::endl;
#elif VERBOSITY == 1
    int n_result_digits = n_rows > 0 ? (int)log10((double)n_rows) + 1 : 1;

    working_file << std::left << std::setw(n_result_digits + 8)
                 << "vec idx:" << std::left << std::setw(n_result_digits + 8)
                 << "row idx:" << std::left << std::setw(PRINT_WIDTH)
                 << "MKL results:" << std::left << std::setw(PRINT_WIDTH)
                 << "SMAX results:" << std::left << std::setw(PRINT_WIDTH)
                 << "rel. diff(%):" << std::left << std::setw(PRINT_WIDTH)
                 << "abs. diff:" << std::endl;

    working_file << std::left << std::setw(n_result_digits + 8) << "--------"
                 << std::left << std::setw(n_result_digits + 8) << "--------"
                 << std::left << std::setw(PRINT_WIDTH) << "-----------"
                 << std::left << std::setw(PRINT_WIDTH) << "------------"
                 << std::left << std::setw(PRINT_WIDTH) << "------------"
                 << std::left << std::setw(PRINT_WIDTH) << "---------"
                 << std::endl;
#endif

    // Print comparison
    int vec_count = 0;
    for (int i = 0; i < n_rows * n_vectors; ++i) {

        relative_diff = std::abs(y_MKL[i] - y_SMAX[i]) / y_MKL[i];
        absolute_diff = std::abs(y_MKL[i] - y_SMAX[i]);

#if VERBOSITY == 1
        // Protect against printing 'inf's
        if (std::abs(y_MKL[i]) < 1e-25) {
            relative_diff = y_SMAX[i];
        }

        working_file << std::left << std::setw(n_result_digits + 8) << vec_count
                     << std::left << std::setw(n_result_digits + 8)
                     << (i - (n_rows * vec_count)) << std::left
                     << std::setprecision(8) << std::scientific
                     << std::setw(PRINT_WIDTH) << y_MKL[i] << std::left
                     << std::setw(PRINT_WIDTH) << y_SMAX[i] << std::left
                     << std::setw(PRINT_WIDTH) << 100 * relative_diff
                     << std::left << std::setw(PRINT_WIDTH) << absolute_diff;

        if ((std::abs(relative_diff) > .01) || std::isinf(relative_diff)) {
            working_file << std::left << std::setw(PRINT_WIDTH) << "ERROR";
        } else if (std::abs(relative_diff) > .0001) {
            working_file << std::left << std::setw(PRINT_WIDTH) << "WARNING";
        }

        working_file << std::endl;

#elif VERBOSITY == 0

        if (relative_diff > max_relative_diff) {
            max_relative_diff = relative_diff;
            max_relative_diff_elem_SMAX = y_SMAX[i];
            max_relative_diff_elem_MKL = y_MKL[i];
        }
        if (absolute_diff > max_absolute_diff) {
            max_absolute_diff = absolute_diff;
            max_absolute_diff_elem_SMAX = y_SMAX[i];
            max_absolute_diff_elem_MKL = y_MKL[i];
        }
#endif
        // increments RHS vector counting for block x_vector
        if ((i + 1) % n_rows == 0 && i > 0)
            ++vec_count;
    }

#if VERBOSITY == 0
    working_file << std::scientific << std::left << std::setw(PRINT_WIDTH)
                 << max_relative_diff_elem_MKL << std::left
                 << std::setw(PRINT_WIDTH) << max_relative_diff_elem_SMAX
                 << std::left << std::setw(PRINT_WIDTH)
                 << 100 * max_relative_diff << std::left
                 << std::setw(PRINT_WIDTH) << max_absolute_diff_elem_MKL
                 << std::left << std::setw(PRINT_WIDTH)
                 << max_absolute_diff_elem_SMAX << std::left
                 << std::setw(PRINT_WIDTH) << max_absolute_diff << std::left
                 << std::setw(PRINT_WIDTH + 6)
                 << compute_euclid_dist(n_rows, y_SMAX, y_MKL);

    if (((std::abs(max_relative_diff) > .01) || std::isnan(max_relative_diff) ||
         std::isinf(max_relative_diff)) ||
        (std::isnan(max_absolute_diff) || std::isinf(max_absolute_diff))) {
        working_file << std::left << std::setw(PRINT_WIDTH) << "ERROR";
    } else if (std::abs(max_relative_diff) > .0001) {
        working_file << std::left << std::setw(PRINT_WIDTH) << "WARNING";
    }

    working_file << std::endl;
#endif
    working_file << "\n";
    working_file.close();
}

int main(int argc, char *argv[]) {
    INIT_MTX;

    int n_vectors = cli_args->block_vec_width;

    DenseMatrix *X = new DenseMatrix(crs_mat->n_cols, n_vectors, 1.0);
    DenseMatrix *Y_smax = new DenseMatrix(crs_mat->n_cols, n_vectors, 0.0);
    DenseMatrix *Y_mkl = new DenseMatrix(crs_mat->n_cols, n_vectors, 0.0);

    // Smax SpMM
    SMAX::Interface *smax = new SMAX::Interface();

    smax->register_kernel("spmm", SMAX::SPMM, SMAX::CPU);
    smax->kernels["spmm"]->register_A(crs_mat->n_rows, crs_mat->n_cols,
                                      crs_mat->nnz, &crs_mat->col,
                                      &crs_mat->row_ptr, &crs_mat->values);
    smax->kernels["spmm"]->register_B(crs_mat->n_cols, n_vectors, &X->values);
    smax->kernels["spmm"]->register_C(crs_mat->n_rows, n_vectors,
                                      &Y_smax->values);

    smax->kernels["spmm"]->run();

    // MKL SpMM
    sparse_matrix_t A;
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    // Create the matrix handle from CSR data
    sparse_status_t status = mkl_sparse_d_create_csr(
        &A, SPARSE_INDEX_BASE_ZERO, crs_mat->n_rows, crs_mat->n_cols,
        crs_mat->row_ptr, crs_mat->row_ptr + 1, crs_mat->col, crs_mat->values);

    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "Failed to create MKL sparse matrix.\n";
        return 1;
    }

    // Optimize the matrix
    status = mkl_sparse_optimize(A);
    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "Failed to optimize MKL sparse matrix.\n";
        return 1;
    }

    status = mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                             1.0, // alpha
                             A, descr, SPARSE_LAYOUT_COLUMN_MAJOR, X->values,
                             n_vectors,
                             crs_mat->n_cols, // leading dimension of X
                             0.0,             // beta
                             Y_mkl->values,
                             crs_mat->n_rows // leading dimension of Y
    );

    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "MKL sparse matrix-matrix multiply failed.\n";
        return 1;
    }

    // Compare
    compare_spmm(crs_mat->n_rows, n_vectors, Y_smax->values, Y_mkl->values,
                 cli_args->matrix_file_name);

    delete X;
    delete Y_smax;
    delete Y_mkl;
    mkl_sparse_destroy(A);
    DESTROY_MTX;
}