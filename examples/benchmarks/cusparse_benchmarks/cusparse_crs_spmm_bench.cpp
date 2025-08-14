#include "../../examples_common.hpp"
#include "../../spmm_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "cusparse_benchmarks_common.hpp"

// Set datatypes
// NOTE: cuSPARSE only takes signed indices
using IT = int;
using VT = float;

int main(int argc, char *argv[]) {
    DEFINE_CUSPARSE_TYPES(IT, VT)

    init_pin(); // Just takes pinning overhead away from timers

    // Setup data structures
    INIT_SPMM(IT, VT);
    auto dX =
        std::make_unique<DenseMatrix<VT>>(crs_mat->n_cols, n_vectors, 1.0);
    auto dY =
        std::make_unique<DenseMatrix<VT>>(crs_mat->n_rows, n_vectors, 0.0);

    int A_num_rows = crs_mat->n_rows;
    int A_num_cols = crs_mat->n_cols;
    int A_nnz = crs_mat->nnz;
    int B_num_rows = A_num_cols;
    int B_num_cols = n_vectors;
    int C_num_rows = A_num_rows;
    int C_num_cols = n_vectors;
    int ldb = B_num_rows;
    int ldc = A_num_rows;
    int B_size = B_num_rows * B_num_cols;
    int C_size = C_num_rows * C_num_cols;
    IT *hA_crsOffsets = crs_mat->row_ptr;
    IT *hA_columns = crs_mat->col;
    VT *hA_values = crs_mat->val;
    VT *hB = dX->val;
    VT *hC = dY->val;

    VT alpha = 1.0;
    VT beta = 0.0;

    //--------------------------------------------------------------------------
    // Device memory management
    IT *dA_crsOffsets, *dA_columns;
    VT *dA_values, *dB, *dC;
    CHECK_CUDA(
        cudaMalloc((void **)&dA_crsOffsets, (A_num_rows + 1) * sizeof(IT)));
    CHECK_CUDA(cudaMalloc((void **)&dA_columns, A_nnz * sizeof(IT)));
    CHECK_CUDA(cudaMalloc((void **)&dA_values, A_nnz * sizeof(VT)));
    CHECK_CUDA(cudaMalloc((void **)&dB, B_size * sizeof(VT)));
    CHECK_CUDA(cudaMalloc((void **)&dC, C_size * sizeof(VT)));

    CHECK_CUDA(cudaMemcpy(dA_crsOffsets, hA_crsOffsets,
                          (A_num_rows + 1) * sizeof(IT),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(IT),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(VT),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, B_size * sizeof(VT), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC, hC, C_size * sizeof(VT), cudaMemcpyHostToDevice));
    //--------------------------------------------------------------------------

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA, A_num_rows, A_num_cols, A_nnz, dA_crsOffsets, dA_columns,
        dA_values, CUSPARSE_INDEX_TYPE, CUSPARSE_INDEX_TYPE,
        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_FLOAT_TYPE));
    // Create dense matrix B
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                       CUSPARSE_FLOAT_TYPE,
                                       CUSPARSE_ORDER_COL));
    // Create dense matrix C
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                       CUSPARSE_FLOAT_TYPE,
                                       CUSPARSE_ORDER_COL));

    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
        CUSPARSE_FLOAT_TYPE, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // execute preprocess (optional)
    CHECK_CUSPARSE(cusparseSpMM_preprocess(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
        CUSPARSE_FLOAT_TYPE, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));

    std::string bench_name = "cusparse_crs_cuda_spmm";
    SETUP_BENCH;

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::function<void()> lambda = [&]() {
        CHECK_CUSPARSE(cusparseSpMM(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
            CUSPARSE_FLOAT_TYPE, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));
    };
    //--------------------------------------------------------------------------
    CHECK_CUDA(cudaDeviceSynchronize());

    // Execute benchmark and print results
    RUN_BENCH;
    PRINT_SPMM_BENCH;

    // device result check
    // CHECK_CUDA(
    //     cudaMemcpy(dY->val, dC, B_size * sizeof(VT),
    //     cudaMemcpyDeviceToHost));
    // for (int i = 0; i < A_num_rows; i++) {
    //     for (int j = 0; j < B_num_cols; j++) {
    //         printf("%f ", hC[i + j * ldc]);
    //     }
    //     printf("\n");
    // }
    //--------------------------------------------------------------------------

    // Clean up
    FINALIZE_SPMM;
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    // device memory deallocation
    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUDA(cudaFree(dA_crsOffsets));
    CHECK_CUDA(cudaFree(dA_columns));
    CHECK_CUDA(cudaFree(dA_values));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
}