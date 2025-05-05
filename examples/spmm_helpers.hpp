#ifndef SMAX_SPMM_HELPERS
#define SMAX_SPMM_HELPERS

#include "examples_common.hpp"

#define OUTPUT_FILENAME "compare_spmm.txt"

#define INIT_SPMM                                                              \
    SpMMParser *parser = new SpMMParser;                                       \
    SpMMParser::SpMMArgs *cli_args = parser->parse(argc, argv);                \
    COOMatrix *coo_mat = new COOMatrix;                                        \
    coo_mat->read_from_mtx(cli_args->matrix_file_name);                        \
    CRSMatrix *crs_mat = new CRSMatrix;                                        \
    crs_mat->convert_coo_to_crs(coo_mat);                                      \
    int n_vectors = cli_args->block_vector_size;

#define FINALIZE_SPMM                                                          \
    delete parser;                                                             \
    delete coo_mat;                                                            \
    delete crs_mat;

class SpMMParser : public CliParser {
  public:
    struct SpMMArgs : public CliArgs {
        int block_vector_size = 1;
    };

    SpMMArgs *parse(int argc, char *argv[]) override {
        if (argc != 3) {
            std::cerr << "Usage: " << argv[0]
                      << " <matrix_file.mtx> <block_vector_size>[int]\n";
            std::exit(EXIT_FAILURE);
        }

        delete args_;
        auto *spmm_args = new SpMMArgs();
        spmm_args->matrix_file_name = argv[1];
        spmm_args->block_vector_size = atoi(argv[2]);
        args_ = spmm_args;
        return spmm_args;
    }

    SpMMArgs *args() const { return static_cast<SpMMArgs *>(args_); }
};

void compare_spmm(const int n_rows, const int n_vectors, const double *y_SMAX,
                  const double *y_MKL, const std::string mtx_name) {

    std::fstream working_file;
    working_file.open(OUTPUT_FILENAME,
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

#endif // SMAX_SPMM_HELPERS