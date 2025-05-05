#ifndef SMAX_SPTRSV_HELPERS
#define SMAX_SPTRSV_HELPERS

#include "examples_common.hpp"

#define OUTPUT_FILENAME "compare_sptrsv.txt"
#define SPLTSV_FLOPS_PER_NZ 2
#define SPLTSV_FLOPS_PER_ROW 2

#define INIT_SPTRSV                                                            \
    SpTRSVParser *parser = new SpTRSVParser;                                   \
    SpTRSVParser::SpTRSVArgs *cli_args = parser->parse(argc, argv);            \
    COOMatrix *coo_mat = new COOMatrix;                                        \
    coo_mat->read_from_mtx(cli_args->matrix_file_name);                        \
    CRSMatrix *crs_mat = new CRSMatrix;                                        \
    crs_mat->convert_coo_to_crs(coo_mat);                                      \
    CRSMatrix *crs_mat_D_plus_L = new CRSMatrix;                               \
    CRSMatrix *crs_mat_U = new CRSMatrix;                                      \
    extract_D_L_U(*crs_mat, *crs_mat_D_plus_L, *crs_mat_U);

#define FINALIZE_SPTRSV                                                        \
    delete cli_args;                                                           \
    delete coo_mat;                                                            \
    delete crs_mat;                                                            \
    delete crs_mat_U;                                                          \
    delete crs_mat_D_plus_L;

class SpTRSVParser : public CliParser {
  public:
    struct SpTRSVArgs : public CliArgs {
        // No extra fields
    };

    SpTRSVArgs *parse(int argc, char *argv[]) override {
        if (argc != 2) {
            std::cerr << "Usage: " << argv[0] << " <matrix_file.mtx>\n";
            std::exit(EXIT_FAILURE);
        }

        delete args_;
        auto *sptrsv_args = new SpTRSVArgs();
        sptrsv_args->matrix_file_name = argv[1];
        args_ = sptrsv_args;
        return sptrsv_args;
    }

    SpTRSVArgs *args() const { return static_cast<SpTRSVArgs *>(args_); }
};

#define PRINT_SPTRSV_BENCH(bench_name, cli_args, n_threads, runtime, n_iter,   \
                           crs_mat_L)                                          \
    std::cout << "----------------" << std::endl;                              \
    std::cout << "--" << bench_name << " Bench--" << std::endl;                \
    std::cout << (cli_args)->matrix_file_name << " with " << (n_threads)       \
              << " thread(s)" << std::endl;                                    \
    std::cout << "Runtime: " << (runtime) << std::endl;                        \
    std::cout << "Iterations: " << (n_iter) << std::endl;                      \
                                                                               \
    long flops_per_iter = ((crs_mat_L)->nnz * SPLTSV_FLOPS_PER_NZ +            \
                           (crs_mat_L)->n_rows * SPLTSV_FLOPS_PER_ROW);        \
    long iter_per_second = static_cast<long>((n_iter) / (runtime));            \
                                                                               \
    std::cout << "Performance: " << flops_per_iter * iter_per_second * F_TO_GF \
              << " [GF/s]" << std::endl;                                       \
    std::cout << "----------------" << std::endl;

void compare_sptrsv(const int n_rows, const double *y_SMAX, const double *y_MKL,
                    const std::string mtx_name) {

    std::fstream working_file;
    working_file.open(OUTPUT_FILENAME,
                      std::fstream::in | std::fstream::out | std::fstream::app);

    GET_THREAD_COUNT;

    double relative_diff, max_relative_diff, max_relative_diff_elem_SMAX,
        max_relative_diff_elem_MKL;
    relative_diff = max_relative_diff = max_relative_diff_elem_SMAX =
        max_relative_diff_elem_MKL = 0.0;
    double absolute_diff, max_absolute_diff, max_absolute_diff_elem_SMAX,
        max_absolute_diff_elem_MKL;
    absolute_diff = max_absolute_diff = max_absolute_diff_elem_SMAX =
        max_absolute_diff_elem_MKL = 0.0;

    // Print header
    working_file << mtx_name << " with " << num_threads << " thread(s)"
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
                 << "row idx:" << std::left << std::setw(PRINT_WIDTH)
                 << "MKL results:" << std::left << std::setw(PRINT_WIDTH)
                 << "SMAX results:" << std::left << std::setw(PRINT_WIDTH)
                 << "rel. diff(%):" << std::left << std::setw(PRINT_WIDTH)
                 << "abs. diff:" << std::endl;

    working_file << std::left << std::setw(n_result_digits + 8) << "--------"
                 << std::left << std::setw(PRINT_WIDTH) << "-----------"
                 << std::left << std::setw(PRINT_WIDTH) << "------------"
                 << std::left << std::setw(PRINT_WIDTH) << "------------"
                 << std::left << std::setw(PRINT_WIDTH) << "---------"
                 << std::endl;
#endif

    // Print comparison
    for (int i = 0; i < n_rows; ++i) {

        relative_diff = std::abs(y_MKL[i] - y_SMAX[i]) / y_MKL[i];
        absolute_diff = std::abs(y_MKL[i] - y_SMAX[i]);

#if VERBOSITY == 1
        // Protect against printing 'inf's
        if (std::abs(y_MKL[i]) < 1e-25) {
            relative_diff = y_SMAX[i];
        }

        working_file << std::setw(n_result_digits + 8) << i << std::left
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

#endif // SMAX_SPTRSV_HELPERS
