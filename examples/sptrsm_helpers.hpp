#pragma once

#include "examples_common.hpp"

#define SPTRSM_OUTPUT_FILENAME "compare_sptrsm.txt"
#define SPTRSM_FLOPS_PER_NZ 2
#define SPTRSM_FLOPS_PER_ROW 2

#define INIT_SPTRSM                                                            \
    SpTRSMParser *parser = new SpTRSMParser;                                   \
    SpTRSMParser::SpTRSMArgs *cli_args = parser->parse(argc, argv);            \
    COOMatrix *coo_mat = new COOMatrix;                                        \
    coo_mat->read_from_mtx(cli_args->matrix_file_name);                        \
    CRSMatrix *crs_mat = new CRSMatrix;                                        \
    crs_mat->convert_coo_to_crs(coo_mat);                                      \
    CRSMatrix *crs_mat_D_plus_L = new CRSMatrix;                               \
    CRSMatrix *crs_mat_U = new CRSMatrix;                                      \
    extract_D_L_U(*crs_mat, *crs_mat_D_plus_L, *crs_mat_U);                    \
    int n_vectors = cli_args->block_vector_size;

#define FINALIZE_SPTRSM                                                        \
    delete cli_args;                                                           \
    delete coo_mat;                                                            \
    delete crs_mat;                                                            \
    delete crs_mat_U;                                                          \
    delete crs_mat_D_plus_L;

#define REGISTER_SPTRSM_DATA(kernel_name, mat, X, B)                           \
    smax->kernel(kernel_name)                                                  \
        ->register_A(mat->n_rows, mat->n_cols, mat->nnz, &mat->col,            \
                     &mat->row_ptr, &mat->val);                                \
    smax->kernel(kernel_name)->register_B(mat->n_cols, n_vectors, &X->val);    \
    smax->kernel(kernel_name)->register_C(mat->n_rows, n_vectors, &B->val);

// TODO
#define PRINT_SPTRSM_BENCH                                                     \
    std::cout << "----------------" << std::endl;                              \
    std::cout << "--" << bench_name << " Bench--" << std::endl;                \
    std::cout << cli_args->matrix_file_name << " with " << n_threads           \
              << " thread(s)" << std::endl;                                    \
    std::cout << "Runtime: " << runtime << std::endl;                          \
    std::cout << "Iterations: " << n_iter << std::endl;                        \
                                                                               \
    long flops_per_iter =                                                      \
        n_vectors * (crs_mat_D_plus_L->nnz * SPTRSM_FLOPS_PER_NZ +             \
                     crs_mat_D_plus_L->n_rows * SPTRSM_FLOPS_PER_ROW);         \
    long iter_per_second = static_cast<long>(n_iter / runtime);                \
                                                                               \
    std::cout << "Performance: " << flops_per_iter * iter_per_second * F_TO_GF \
              << " [GF/s]" << std::endl;                                       \
    std::cout << "----------------" << std::endl;

class SpTRSMParser : public CliParser {
  public:
    struct SpTRSMArgs : public CliArgs {
        int block_vector_size = 1;
    };

    SpTRSMArgs *parse(int argc, char *argv[]) override {
        if (argc != 3) {
            std::cerr << "Usage: " << argv[0]
                      << " <matrix_file.mtx> <block_vector_size>[int]\n";
            std::exit(EXIT_FAILURE);
        }

        delete args_;
        auto *sptrsm_args = new SpTRSMArgs();
        sptrsm_args->matrix_file_name = argv[1];
        sptrsm_args->block_vector_size = atoi(argv[2]);
        args_ = sptrsm_args;
        return sptrsm_args;
    }

    SpTRSMArgs *args() const { return static_cast<SpTRSMArgs *>(args_); }
};

void compare_sptrsm(const int n_rows, const int n_vectors, const double *y_SMAX,
                    const double *y_MKL, const std::string mtx_name) {

    std::fstream working_file;
    working_file.open(SPTRSM_OUTPUT_FILENAME,
                      std::fstream::in | std::fstream::out | std::fstream::app);

    double relative_diff, max_relative_diff, max_relative_diff_elem_SMAX,
        max_relative_diff_elem_MKL;
    relative_diff = max_relative_diff = max_relative_diff_elem_SMAX =
        max_relative_diff_elem_MKL = 0.0;
    double absolute_diff, max_absolute_diff, max_absolute_diff_elem_SMAX,
        max_absolute_diff_elem_MKL;
    absolute_diff = max_absolute_diff = max_absolute_diff_elem_SMAX =
        max_absolute_diff_elem_MKL = 0.0;

    // Print header
    GET_THREAD_COUNT;
    working_file << mtx_name << " with " << n_threads << " thread(s)"
                 << std::endl;
#if VERBOSITY == 0
    working_file << std::left << std::setw(PRINT_WIDTH)
                 << "MKL rel. elem:" << std::left << std::setw(PRINT_WIDTH)
                 << "SMAX rel. elem:" << std::left << std::setw(PRINT_WIDTH)
                 << "lrgst rel. (%):" << std::left << std::setw(PRINT_WIDTH)
                 << "MKL abs. elem:" << std::left << std::setw(PRINT_WIDTH)
                 << "SMAX abs. elem:" << std::left << std::setw(PRINT_WIDTH)
                 << "lrgst abs.:" << std::left << std::setw(PRINT_WIDTH)
                 << "||MKL - SMAX||" << std::endl;
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

        DIFF_STATUS_MACRO(relative_diff, working_file);

        working_file << std::endl;

#elif VERBOSITY == 0
        UPDATE_MAX_DIFFS(i, y_MKL, y_SMAX, relative_diff, absolute_diff);
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

    CHECK_MAX_DIFFS_AND_PRINT_ERROR_WARNING(max_relative_diff,
                                            max_absolute_diff, working_file);

    working_file << std::endl;
#endif
    working_file << "\n";
    working_file.close();
}
