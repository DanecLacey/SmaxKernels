#include "../../examples_common.hpp"
#include "../../spmv_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "cusparse_benchmarks_common.hpp"

using IT = int;
using VT = float;

int main(int argc, char *argv[]) {

    // Just to take overhead of pinning away from timers
    init_pin();

    INIT_SPMV(IT, VT);

    DenseMatrix<VT> *hX = new DenseMatrix<VT>(crs_mat->n_cols, 1, 1.0);
    DenseMatrix<VT> *hY = new DenseMatrix<VT>(crs_mat->n_rows, 1, 0.0);

    const int A_num_rows = crs_mat->n_rows;
    const int A_num_cols = crs_mat->n_cols;
    const int A_nnz = crs_mat->nnz;
    int *hA_csrOffsets = crs_mat->row_ptr;
    int *hA_columns = crs_mat->col;
    VT *hA_values = crs_mat->val;

    float alpha = 1.0f;
    float beta = 0.0f;

    //--------------------------------------------------------------------------
    // Device memory management
    int *dA_csrOffsets, *dA_columns;
    VT *dA_values, *dX, *dY;
    CHECK_CUDA(
        cudaMalloc((void **)&dA_csrOffsets, (A_num_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&dA_columns, A_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&dA_values, A_nnz * sizeof(VT)));
    CHECK_CUDA(cudaMalloc((void **)&dX, A_num_cols * sizeof(VT)));
    CHECK_CUDA(cudaMalloc((void **)&dY, A_num_rows * sizeof(VT)));

    CHECK_CUDA(cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                          (A_num_rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
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
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                     dA_csrOffsets, dA_columns, dA_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    // Create dense vector X
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F));
    // Create dense vector y
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_32F));
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta,
        vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // execute preprocess (optional)
    CHECK_CUSPARSE(cusparseSpMV_preprocess(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta,
        vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

    std::string bench_name = "cusparse_csr_cuda_spmv";
    float runtime = 0.0;
    int n_iter = MIN_NUM_ITERS;

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::function<void(bool)> lambda = [handle, matA, vecX, vecY, dBuffer, beta,
                                        alpha](bool warmup) {
        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
    };
    //--------------------------------------------------------------------------
    CHECK_CUDA(cudaDeviceSynchronize());

    RUN_BENCH;
    PRINT_SPMV_BENCH;
    FINALIZE_SPMV;

    // device result check
    // CHECK_CUDA(cudaMemcpy(hY->val, dY, A_num_rows * sizeof(VT),
    //                       cudaMemcpyDeviceToHost));
    // for (int i = 0; i < A_num_rows; i++) {
    //     printf("%f\n", hY->val[i]);
    // }
    //--------------------------------------------------------------------------

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    // device memory deallocation
    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUDA(cudaFree(dA_csrOffsets));
    CHECK_CUDA(cudaFree(dA_columns));
    CHECK_CUDA(cudaFree(dA_values));
    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dY));
    delete hX;
    delete hY;
    return 0;
}