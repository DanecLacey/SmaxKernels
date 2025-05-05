#ifndef SMAX_SPGEMM_HELPERS
#define SMAX_SPGEMM_HELPERS

#include "examples_common.hpp"

#define OUTPUT_FILENAME "compare_spgemm.txt"

#define INIT_SPGEMM                                                            \
    SpGEMMParser *parser = new SpGEMMParser;                                   \
    SpGEMMParser::SpGEMMArgs *cli_args = parser->parse(argc, argv);            \
    COOMatrix *coo_mat_A = new COOMatrix;                                      \
    coo_mat_A->read_from_mtx(cli_args->matrix_file_name_A);                    \
    CRSMatrix *crs_mat_A = new CRSMatrix;                                      \
    crs_mat_A->convert_coo_to_crs(coo_mat_A);                                  \
    COOMatrix *coo_mat_B = new COOMatrix;                                      \
    coo_mat_B->read_from_mtx(cli_args->matrix_file_name_B);                    \
    CRSMatrix *crs_mat_B = new CRSMatrix;                                      \
    crs_mat_B->convert_coo_to_crs(coo_mat_B);

#define FINALIZE_SPGEMM                                                        \
    delete parser;                                                             \
    delete coo_mat_A;                                                          \
    delete crs_mat_A;                                                          \
    delete coo_mat_B;                                                          \
    delete crs_mat_B;

#endif // SMAX_SPGEMM_HELPERS