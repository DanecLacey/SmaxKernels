#include "../../examples_common.hpp"
#include "../../spmv_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "cusparse_benchmarks_common.hpp"

// Set datatypes
using IT = int;
using VT = float;

int main(int argc, char *argv[]) {

    auto smax = std::make_unique<SMAX::Interface>();

    init_pin(); // Just takes pinning overhead away from timers

    // Setup data structures
    INIT_SPMV(IT, VT);
    int A_scs_C = _C;
    int A_scs_sigma = 1; // Fixed for cudsparse
    int A_scs_n_rows = 0;
    int A_scs_n_rows_padded = 0;
    int A_scs_n_cols = 0;
    int A_scs_n_chunks = 0;
    int A_scs_n_elements = 0;
    int A_scs_nnz = 0;
    IT *A_scs_chunk_ptr = nullptr;
    IT *A_scs_chunk_lengths = nullptr;
    IT *A_scs_col = nullptr;
    VT *A_scs_val = nullptr;
    IT *A_scs_perm = nullptr;
    smax->utils->convert_crs_to_scs<IT, VT, int>(
        crs_mat->n_rows, crs_mat->n_cols, crs_mat->nnz, crs_mat->col,
        crs_mat->row_ptr, crs_mat->val, A_scs_C, A_scs_sigma, A_scs_n_rows,
        A_scs_n_rows_padded, A_scs_n_cols, A_scs_n_chunks, A_scs_n_elements,
        A_scs_nnz, A_scs_chunk_ptr, A_scs_chunk_lengths, A_scs_col, A_scs_val,
        A_scs_perm);

    auto X = std::make_unique<DenseMatrix<VT>>(crs_mat->n_cols, 1, 1.0);
    auto Y = std::make_unique<DenseMatrix<VT>>(crs_mat->n_rows, 1, 0.0);

    const int A_num_rows = crs_mat->n_rows;
    const int A_num_cols = crs_mat->n_cols;
    const int A_nnz = crs_mat->nnz;
    const int A_slice_size = A_scs_C;
    const int A_values_size = A_scs_n_elements;
    int A_num_slices = A_scs_n_chunks; // 2

    int *hA_sliceOffsets = A_scs_chunk_ptr;
    int *hA_columns = A_scs_col;
    float *hA_values = A_scs_val;
    float *hX = X->val;
    float *hY = Y->val;
    float alpha = 1.0f;
    float beta = 0.0f;

    //--------------------------------------------------------------------------
    // Device memory management
    int *dA_sliceOffsets, *dA_columns;
    float *dA_values, *dX, *dY;
    CHECK_CUDA(cudaMalloc((void **)&dA_sliceOffsets,
                          (A_num_slices + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&dA_columns, A_values_size * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&dA_values, A_values_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&dX, A_num_cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&dY, A_num_rows * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA_sliceOffsets, hA_sliceOffsets,
                          (A_num_slices + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns, A_values_size * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_values, hA_values, A_values_size * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(dX, hX, A_num_cols * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(dY, hY, A_num_rows * sizeof(float), cudaMemcpyHostToDevice));
    //--------------------------------------------------------------------------

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    // Create sparse matrix A in SELL format
    CHECK_CUSPARSE(cusparseCreateSlicedEll(
        &matA, A_num_rows, A_num_cols, A_nnz, A_values_size, A_slice_size,
        dA_sliceOffsets, dA_columns, dA_values, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    // Create dense vector X
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F));
    // Create dense vector y
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_32F));
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta,
        vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    std::string bench_name = "cusparse_SELL_cuda_spmv";
    SETUP_BENCH;

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::function<void()> lambda = [&]() {
        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
    };
    //--------------------------------------------------------------------------
    CHECK_CUDA(cudaDeviceSynchronize());

    // Execute benchmark and print results
    RUN_BENCH;
    PRINT_SPMV_BENCH;

    // device result check
    // CHECK_CUDA(
    //     cudaMemcpy(hY, dY, A_num_rows * sizeof(VT), cudaMemcpyDeviceToHost));
    // for (int i = 0; i < A_num_rows; i++) {
    //     printf("%f\n", hY[i]);
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
    CHECK_CUDA(cudaFree(dA_sliceOffsets));
    CHECK_CUDA(cudaFree(dA_columns));
    CHECK_CUDA(cudaFree(dA_values));
    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dY));
}