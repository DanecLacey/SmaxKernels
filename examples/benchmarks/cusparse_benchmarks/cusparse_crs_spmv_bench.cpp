#include "../../examples_common.hpp"
#include "../../spmv_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "cusparse_benchmarks_common.hpp"

// Set datatypes
// NOTE: cuSPARSE only takes signed indices
using IT = int;
using VT = double;

int main(int argc, char *argv[]) {
    DEFINE_CUSPARSE_TYPES(IT, VT)

    init_pin(); // Just takes pinning overhead away from timers

    // Setup data structures
    INIT_SPMV(IT, VT);
    auto hX = std::make_unique<DenseMatrix<VT>>(crs_mat->n_cols, 1, 1.0);
    auto hY = std::make_unique<DenseMatrix<VT>>(crs_mat->n_rows, 1, 0.0);

    const unsigned long long A_num_rows = crs_mat->n_rows;
    const unsigned long long A_num_cols = crs_mat->n_cols;
    const unsigned long long A_nnz = crs_mat->nnz;
    IT *hA_crsOffsets = crs_mat->row_ptr;
    IT *hA_columns = crs_mat->col;
    VT *hA_values = crs_mat->val;

    VT alpha = 1.0;
    VT beta = 0.0;

    //--------------------------------------------------------------------------
    // Device memory management
    IT *dA_crsOffsets, *dA_columns;
    VT *dA_values, *dX, *dY;
    CHECK_CUDA(
        cudaMalloc((void **)&dA_crsOffsets, (A_num_rows + 1) * sizeof(IT)));
    CHECK_CUDA(cudaMalloc((void **)&dA_columns, A_nnz * sizeof(IT)));
    CHECK_CUDA(cudaMalloc((void **)&dA_values, A_nnz * sizeof(VT)));
    CHECK_CUDA(cudaMalloc((void **)&dX, A_num_cols * sizeof(VT)));
    CHECK_CUDA(cudaMalloc((void **)&dY, A_num_rows * sizeof(VT)));

    CHECK_CUDA(cudaMemcpy(dA_crsOffsets, hA_crsOffsets,
                          (A_num_rows + 1) * sizeof(IT),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(IT),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(VT),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dX, hX->val, A_num_cols * sizeof(VT),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dY, hY->val, A_num_rows * sizeof(VT),
                          cudaMemcpyHostToDevice));
    //--------------------------------------------------------------------------

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA, A_num_rows, A_num_cols, A_nnz, dA_crsOffsets, dA_columns,
        dA_values, CUSPARSE_INDEX_TYPE, CUSPARSE_INDEX_TYPE,
        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_FLOAT_TYPE));
    // Create dense vector X
    CHECK_CUSPARSE(
        cusparseCreateDnVec(&vecX, A_num_cols, dX, CUSPARSE_FLOAT_TYPE));
    // Create dense vector y
    CHECK_CUSPARSE(
        cusparseCreateDnVec(&vecY, A_num_rows, dY, CUSPARSE_FLOAT_TYPE));
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta,
        vecY, CUSPARSE_FLOAT_TYPE, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // execute preprocess (optional)
    CHECK_CUSPARSE(cusparseSpMV_preprocess(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta,
        vecY, CUSPARSE_FLOAT_TYPE, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

    std::string bench_name = "cusparse_crs_cuda_spmv";
    SETUP_BENCH;

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::function<void()> lambda = [handle, matA, vecX, vecY, dBuffer, beta,
                                    alpha]() {
        CHECK_CUSPARSE(cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta,
            vecY, CUSPARSE_FLOAT_TYPE, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
    };
    //--------------------------------------------------------------------------
    CHECK_CUDA(cudaDeviceSynchronize());

    // Execute benchmark and print results
    RUN_BENCH;
    PRINT_SPMV_BENCH;

    // device result check
    // CHECK_CUDA(cudaMemcpy(hY->val, dY, A_num_rows * sizeof(VT),
    //                       cudaMemcpyDeviceToHost));
    // for (int i = 0; i < A_num_rows; i++) {
    //     printf("%f\n", hY->val[i]);
    // }
    //--------------------------------------------------------------------------

    // Clean up
    FINALIZE_SPMV;
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    // device memory deallocation
    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUDA(cudaFree(dA_crsOffsets));
    CHECK_CUDA(cudaFree(dA_columns));
    CHECK_CUDA(cudaFree(dA_values));
    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dY));
}