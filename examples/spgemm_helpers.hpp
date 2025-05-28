#pragma once

#include "examples_common.hpp"

#define SPGEMM_OUTPUT_FILENAME "compare_spgemm.txt"
#define RELATIVE_VAL_ERROR_TOL 1e-14
#define ERROR_CUT_OFF 100
#define SPGEMM_FLOPS_PER_NZ 2

#define INIT_SPGEMM                                                            \
    SpGEMMParser *parser = new SpGEMMParser;                                   \
    SpGEMMParser::SpGEMMArgs *cli_args = parser->parse(argc, argv);            \
    bool compute_AA = false;                                                   \
    if (cli_args->matrix_file_name_A == cli_args->matrix_file_name_B) {        \
        compute_AA = true;                                                     \
    }                                                                          \
    COOMatrix *coo_mat_A = new COOMatrix;                                      \
    coo_mat_A->read_from_mtx(cli_args->matrix_file_name_A);                    \
    CRSMatrix *crs_mat_A = new CRSMatrix;                                      \
    crs_mat_A->convert_coo_to_crs(coo_mat_A);                                  \
    COOMatrix *coo_mat_B = new COOMatrix;                                      \
    if (!compute_AA) {                                                         \
        coo_mat_B->read_from_mtx(cli_args->matrix_file_name_B);                \
    }                                                                          \
    CRSMatrix *crs_mat_B = new CRSMatrix;                                      \
    crs_mat_B->convert_coo_to_crs(coo_mat_B);

#define FINALIZE_SPGEMM                                                        \
    delete parser;                                                             \
    delete coo_mat_A;                                                          \
    delete crs_mat_A;                                                          \
    delete coo_mat_B;                                                          \
    delete crs_mat_B;

#define REGISTER_SPGEMM_DATA(kernel_name, matA, matB, matC)                    \
    smax->kernel(kernel_name)                                                  \
        ->register_A(matA->n_rows, matA->n_cols, matA->nnz, matA->col,         \
                     matA->row_ptr, matA->val);                                \
    smax->kernel(kernel_name)                                                  \
        ->register_B(matB->n_rows, matB->n_cols, matB->nnz, matB->col,         \
                     matB->row_ptr, matB->val);                                \
    smax->kernel(kernel_name)                                                  \
        ->register_C(&matC->n_rows, &matC->n_cols, &matC->nnz, &matC->col,     \
                     &matC->row_ptr, &matC->val);

#define PRINT_SPGEMM_BENCH(result_nnz)                                         \
    std::cout << "----------------" << std::endl;                              \
    std::cout << "--" << bench_name << " Bench--" << std::endl;                \
    std::cout << cli_args->matrix_file_name_A << " * "                         \
              << cli_args->matrix_file_name_B << " with " << n_threads         \
              << " thread(s)" << std::endl;                                    \
    std::cout << "Runtime: " << runtime << std::endl;                          \
    std::cout << "Iterations: " << n_iter << std::endl;                        \
                                                                               \
    long flops_per_iter = (result_nnz) * SPGEMM_FLOPS_PER_NZ;                  \
    long iter_per_second = static_cast<long>(n_iter / runtime);                \
                                                                               \
    std::cout << "Performance: " << flops_per_iter * iter_per_second * F_TO_GF \
              << " [GF/s]" << std::endl;                                       \
    std::cout << "----------------" << std::endl;

class SpGEMMParser : public CliParser {
  public:
    struct SpGEMMArgs : public CliArgs {
        std::string matrix_file_name_A;
        std::string matrix_file_name_B;
    };

    SpGEMMArgs *parse(int argc, char *argv[]) override {
        if (argc != 3) {
            std::cerr << "Usage: " << argv[0]
                      << " <matrix_file1.mtx> <matrix_file2.mtx>\n";
            std::exit(EXIT_FAILURE);
        }

        delete args_;
        auto *SpGEMM_args = new SpGEMMArgs();
        SpGEMM_args->matrix_file_name_A = argv[1];
        SpGEMM_args->matrix_file_name_B = argv[2];
        args_ = SpGEMM_args;
        return SpGEMM_args;
    }

    SpGEMMArgs *args() const { return static_cast<SpGEMMArgs *>(args_); }
};

void compare_spgemm(CRSMatrix *C_smax, CRSMatrix *C_mkl,
                    const std::string mtx_name_A,
                    const std::string mtx_name_B) {

    std::fstream working_file;
    working_file.open(SPGEMM_OUTPUT_FILENAME,
                      std::fstream::in | std::fstream::out | std::fstream::app);

    GET_THREAD_COUNT;

    working_file << mtx_name_A << " * " << mtx_name_B << " with " << n_threads
                 << " thread(s)\n";

    // Basic sanity checks
    if (C_smax->n_rows != C_mkl->n_rows) {
        fprintf(stderr,
                "ERROR: Number of rows mismatch between C_smax (%i) and C_mkl "
                "(%i).\n",
                C_smax->n_rows, C_mkl->n_rows);
        exit(EXIT_FAILURE);
    }
    if (C_smax->n_cols != C_mkl->n_cols) {
        fprintf(stderr,
                "ERROR: Number of cols mismatch between C_smax (%i) and C_mkl "
                "(%i).\n",
                C_smax->n_cols, C_mkl->n_cols);
        exit(EXIT_FAILURE);
    }
    if (C_smax->nnz != C_mkl->nnz) {
        fprintf(
            stderr,
            "ERROR: Number of non zeros mismatch between C_smax (%i) and C_mkl "
            "(%i).\n",
            C_smax->nnz, C_mkl->nnz);
        exit(EXIT_FAILURE);
    }

    int col_diff = 0;
    double val_diff = 0.0;
    double relative_val_diff = 0.0;
    int row_ptr_diff = 0;
    int bad_idxs = 0;

    int nnz_digits = C_smax->nnz > 0 ? (int)log10((double)C_smax->nnz) + 1 : 2;

    // Check if there are any errors at all
    for (int i = 0; i < C_smax->nnz; ++i) {
        col_diff = C_smax->col[i] - C_mkl->col[i];
        val_diff = C_smax->val[i] - C_mkl->val[i];
        relative_val_diff = val_diff / C_mkl->val[i];

        if (std::abs(col_diff) > 0) {
            ++bad_idxs;
        }
        if (std::abs(relative_val_diff) > RELATIVE_VAL_ERROR_TOL) {
            ++bad_idxs;
        }
        if (i <= C_smax->n_rows) {
            row_ptr_diff = C_smax->row_ptr[i] - C_mkl->row_ptr[i];
            if (std::abs(row_ptr_diff) > 0) {
                ++bad_idxs;
            }
        }
    }

    // If so, print out all errors
    if (bad_idxs > 0) {
        working_file << "Possible errors detected:" << std::endl;
        working_file << std::left << std::setw(8) << "Idx"
                     << std::setw(PRINT_WIDTH + 4) << "C_smax.col-C_mkl.col"
                     << std::setw(PRINT_WIDTH) << "C_smax.val"
                     << std::setw(PRINT_WIDTH) << "C_mkl.val"
                     << std::setw(PRINT_WIDTH) << "val_diff"
                     << std::setw(PRINT_WIDTH) << "rel_diff (%)"
                     << std::setw(PRINT_WIDTH) << "row_ptr_diff"
                     << "\n";

        working_file << std::left << std::setw(8) << "----"
                     << std::setw(PRINT_WIDTH + 4) << "------------------"
                     << std::setw(PRINT_WIDTH) << "----------"
                     << std::setw(PRINT_WIDTH) << "---------"
                     << std::setw(PRINT_WIDTH) << "--------"
                     << std::setw(PRINT_WIDTH) << "------------"
                     << std::setw(PRINT_WIDTH) << "------------"
                     << "\n";

        int error_cut_off_counter = 0;
        for (int i = 0; i < C_smax->nnz; ++i) {
            col_diff = C_smax->col[i] - C_mkl->col[i];
            val_diff = C_smax->val[i] - C_mkl->val[i];
            relative_val_diff = val_diff / C_mkl->val[i];

            bool is_row_ptr_valid = (i <= C_smax->n_rows);
            if (is_row_ptr_valid) {
                row_ptr_diff = C_smax->row_ptr[i] - C_mkl->row_ptr[i];
            }

            if (std::abs(col_diff) > 0 ||
                std::abs(relative_val_diff) > RELATIVE_VAL_ERROR_TOL ||
                (is_row_ptr_valid && std::abs(row_ptr_diff) > 0)) {

                working_file
                    << std::left << std::setw(nnz_digits + 8) << i << std::left
                    << std::setw(PRINT_WIDTH) << col_diff << std::left
                    << std::setw(PRINT_WIDTH) << std::setprecision(8)
                    << std::scientific << C_smax->val[i] << std::left
                    << std::setw(PRINT_WIDTH) << std::setprecision(8)
                    << std::scientific << C_mkl->val[i] << std::left
                    << std::setw(PRINT_WIDTH) << std::setprecision(8)
                    << std::scientific << val_diff << std::left
                    << std::setw(PRINT_WIDTH + 8) << std::setprecision(8)
                    << std::scientific << (100.0 * relative_val_diff) << "%"
                    << std::left << std::setw(PRINT_WIDTH + 4)
                    << (is_row_ptr_valid ? std::to_string(row_ptr_diff) : "n/a")
                    << std::endl;

                if (++error_cut_off_counter == ERROR_CUT_OFF) {
                    working_file
                        << "... (cutoff reached: " << (bad_idxs - ERROR_CUT_OFF)
                        << " more possible errors)" << std::endl;
                    break;
                }
            }
        }
    } else {
        working_file << "Done! No errors detected." << std::endl;
    }
    working_file << std::endl;
}
