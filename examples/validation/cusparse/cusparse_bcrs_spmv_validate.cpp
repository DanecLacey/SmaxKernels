#include "../../examples_common.hpp"
#include "../../spmv_helpers.hpp"
#include "../validation_common.hpp"
#include "cusparse_validation_common.hpp"

// Set datatypes
// NOTE: cuSPARSE only takes signed indices
using IT = int;
using VT = double;

int main(int argc, char *argv[]) {
    DEFINE_CUSPARSE_TYPES(IT, VT)

    INIT_SPMV(IT, VT);
    IT hpad = cli_args->_hpad;
    IT wpad = cli_args->_wpad;
    bool use_cm = cli_args->_use_cm;
    std::string custom_kernel = cli_args->_ck;

    DenseMatrix<VT> *x = new DenseMatrix<VT>(crs_mat->n_cols, 1, 1.0);
    DenseMatrix<VT> *y_smax = new DenseMatrix<VT>(crs_mat->n_rows, 1, 0.0);
    DenseMatrix<VT> *y_cusparse = new DenseMatrix<VT>(crs_mat->n_rows, 1, 0.0);

    // Smax SpMV
    SMAX::Interface *smax = new SMAX::Interface();

    register_kernel<IT, VT>(smax, std::string("my_bcrs_spmv"),
                            SMAX::KernelType::SPMV, SMAX::PlatformType::CUDA);

    // Declare bcrs operand
    BCRSMatrix<IT, VT> *A_bcrs = new BCRSMatrix<IT, VT>();

    // Convert CRS matrix to BCRS
    smax->utils->convert_crs_to_bcrs<IT, VT, ULL>(
        crs_mat->n_rows, crs_mat->n_cols, crs_mat->nnz, crs_mat->col,
        crs_mat->row_ptr, crs_mat->val, A_bcrs->n_rows, A_bcrs->n_cols,
        A_bcrs->n_blocks, A_bcrs->b_height, A_bcrs->b_width, A_bcrs->b_h_pad,
        A_bcrs->b_w_pad, A_bcrs->col, A_bcrs->row_ptr, A_bcrs->val, hpad, wpad,
        hpad, wpad, use_cm);

    smax->kernel("my_bcrs_spmv")->set_mat_bcrs(true);
    smax->kernel("my_bcrs_spmv")->set_block_column_major(use_cm);

    SMAX::SpMVType custom_kernel_type = SMAX::SpMVType::naive_thread_per_row;

    if (custom_kernel == "tpr") {
        custom_kernel_type = SMAX::SpMVType::naive_thread_per_row;
    }
    else if (custom_kernel == "nws") {
        custom_kernel_type = SMAX::SpMVType::naive_warp_shuffle;
    }
    else if (custom_kernel == "nwg") {
        custom_kernel_type = SMAX::SpMVType::naive_warp_group;
    }

    smax->kernel("my_bcrs_spmv")
        ->set_kernel_implementation(custom_kernel_type);

    // A is assumed to be in BCRS format
    smax->kernel("my_bcrs_spmv")
        ->register_A(A_bcrs->n_rows, A_bcrs->n_cols, A_bcrs->n_blocks,
                     A_bcrs->b_height, A_bcrs->b_width, A_bcrs->b_h_pad,
                     A_bcrs->b_w_pad, A_bcrs->col, A_bcrs->row_ptr,
                     A_bcrs->val);
    // x and y are dense matrices
    smax->kernel("my_bcrs_spmv")->register_B(A_bcrs->n_cols, x->val);
    smax->kernel("my_bcrs_spmv")->register_C(A_bcrs->n_rows, y_smax->val);

    smax->kernel("my_bcrs_spmv")->run();

    // cuSPARSE CRS SpMV
    IT *hA_csrOffsets = crs_mat->row_ptr;
    IT *hA_columns = crs_mat->col;
    VT *hA_values = crs_mat->val;

    VT alpha = 1.0;
    VT beta = 0.0;

    //--------------------------------------------------------------------------
    // Device memory management
    IT *dA_csrOffsets, *dA_columns;
    VT *dA_values, *dX, *dY;
    CHECK_CUDA(cudaMalloc((void **)&dA_csrOffsets,
                          (crs_mat->n_rows + 1) * sizeof(IT)));
    CHECK_CUDA(cudaMalloc((void **)&dA_columns, crs_mat->nnz * sizeof(IT)));
    CHECK_CUDA(cudaMalloc((void **)&dA_values, crs_mat->nnz * sizeof(VT)));
    CHECK_CUDA(cudaMalloc((void **)&dX, crs_mat->n_cols * sizeof(VT)));
    CHECK_CUDA(cudaMalloc((void **)&dY, crs_mat->n_rows * sizeof(VT)));

    CHECK_CUDA(cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                          (crs_mat->n_rows + 1) * sizeof(IT),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns, crs_mat->nnz * sizeof(IT),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_values, hA_values, crs_mat->nnz * sizeof(VT),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dX, x->val, crs_mat->n_cols * sizeof(VT),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dY, y_cusparse->val, crs_mat->n_rows * sizeof(VT),
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
        &matA, crs_mat->n_rows, crs_mat->n_cols, crs_mat->nnz, dA_csrOffsets,
        dA_columns, dA_values, CUSPARSE_INDEX_TYPE, CUSPARSE_INDEX_TYPE,
        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_FLOAT_TYPE));
    // Create dense vector X
    CHECK_CUSPARSE(
        cusparseCreateDnVec(&vecX, crs_mat->n_cols, dX, CUSPARSE_FLOAT_TYPE));

    // Create dense vector y
    CHECK_CUSPARSE(
        cusparseCreateDnVec(&vecY, crs_mat->n_rows, dY, CUSPARSE_FLOAT_TYPE));

    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta,
        vecY, CUSPARSE_FLOAT_TYPE, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    CHECK_CUSPARSE(cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta,
        vecY, CUSPARSE_FLOAT_TYPE, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(y_cusparse->val, dY, sizeof(VT) * crs_mat->n_rows,
                          cudaMemcpyDeviceToHost));

    // Compare
    compare_spmv<VT>(crs_mat->n_rows, y_smax->val, y_cusparse->val,
                     cli_args->matrix_file_name);

    delete x;
    delete y_smax;
    delete y_cusparse;
    delete A_bcrs;

    FINALIZE_SPMV;

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
}