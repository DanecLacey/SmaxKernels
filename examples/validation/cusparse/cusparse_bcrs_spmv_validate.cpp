#include "../../examples_common.hpp"
#include "../../spmv_helpers.hpp"
#include "../validation_common.hpp"
#include "cusparse_validation_common.hpp"

// Set datatypes
// NOTE: cuSPARSE only takes signed indices
using IT = int;
using VT = double;

// Since we need compile-time dispatching
template <typename IT, typename VT>
void bsr_mv(cusparseHandle_t handle, cusparseMatDescr_t desc,
            cusparseDirection_t direction, ULL mb, ULL nb, ULL nnzb,
            const VT *d_val, const IT *d_row, const IT *d_col, ULL block_dim,
            const VT *d_x, VT *d_y, VT alpha, VT beta) {
    if constexpr (std::is_same_v<VT, double>) {
        CHECK_CUSPARSE(cusparseDbsrmv(
            handle, direction, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, nnzb,
            &alpha, desc, d_val, d_row, d_col, block_dim, d_x, &beta, d_y));
    } else { // float
        CHECK_CUSPARSE(cusparseSbsrmv(
            handle, direction, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, nnzb,
            &alpha, desc, d_val, d_row, d_col, block_dim, d_x, &beta, d_y));
    }
}

int main(int argc, char *argv[]) {
    DEFINE_CUSPARSE_TYPES(IT, VT)

    INIT_SPMV(IT, VT);
    IT hpad = cli_args->_hpad;
    IT wpad = cli_args->_wpad;
    IT block_dim = cli_args->_hpad;
    printf("Warning, cuSPARSE accepts only square blocks. Fixing "
           "block_dim=hpad=%d\n",
           block_dim);
    bool use_cm = cli_args->_use_cm;
    std::string custom_kernel = cli_args->_ck;

    DenseMatrix<VT> *x = new DenseMatrix<VT>(crs_mat->n_cols, 1, 1.0);
    DenseMatrix<VT> *y_smax = new DenseMatrix<VT>(crs_mat->n_rows, 1, 0.0);
    DenseMatrix<VT> *y_cusparse_crs =
        new DenseMatrix<VT>(crs_mat->n_rows, 1, 0.0);
    DenseMatrix<VT> *y_cusparse_bcrs =
        new DenseMatrix<VT>(crs_mat->n_rows, 1, 0.0);

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
    } else if (custom_kernel == "nws") {
        custom_kernel_type = SMAX::SpMVType::naive_warp_shuffle;
    } else if (custom_kernel == "nwg") {
        custom_kernel_type = SMAX::SpMVType::naive_warp_group;
    }

    smax->kernel("my_bcrs_spmv")->set_kernel_implementation(custom_kernel_type);

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

    VT alpha = 1.0;
    VT beta = 0.0;
    // clang-format off
    //--------------------------------------------------------------------------
    // cuSPARSE CRS SpMV
    IT *hA_rowptr_crs = crs_mat->row_ptr;
    IT *hA_col_crs = crs_mat->col;
    VT *hA_val_crs = crs_mat->val;

    // Device memory management
    IT *dA_rowptr_crs, *dA_col_crs;
    VT *dA_val_crs, *dx_crs, *dy_crs;
    CHECK_CUDA(cudaMalloc((void **)&dA_rowptr_crs, (crs_mat->n_rows + 1) * sizeof(IT)));
    CHECK_CUDA(cudaMalloc((void **)&dA_col_crs, crs_mat->nnz * sizeof(IT)));
    CHECK_CUDA(cudaMalloc((void **)&dA_val_crs, crs_mat->nnz * sizeof(VT)));
    CHECK_CUDA(cudaMalloc((void **)&dx_crs, crs_mat->n_cols * sizeof(VT)));
    CHECK_CUDA(cudaMalloc((void **)&dy_crs, crs_mat->n_rows * sizeof(VT)));

    CHECK_CUDA(cudaMemcpy(dA_rowptr_crs, hA_rowptr_crs, (crs_mat->n_rows + 1) * sizeof(IT), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_col_crs, hA_col_crs, crs_mat->nnz * sizeof(IT), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_val_crs, hA_val_crs, crs_mat->nnz * sizeof(VT), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dx_crs, x->val, crs_mat->n_cols * sizeof(VT), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dy_crs, y_cusparse_crs->val, crs_mat->n_rows * sizeof(VT), cudaMemcpyHostToDevice));

    cusparseHandle_t handle_crs = NULL;
    cusparseSpMatDescr_t matA_crs;
    cusparseDnVecDescr_t vecX_crs, vecY_crs;
    void *dBuffer_crs = NULL;
    size_t bufferSize_crs = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle_crs));
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA_crs, crs_mat->n_rows, crs_mat->n_cols, crs_mat->nnz, dA_rowptr_crs,
        dA_col_crs, dA_val_crs, CUSPARSE_INDEX_TYPE, CUSPARSE_INDEX_TYPE,
        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_FLOAT_TYPE));
    // Create dense vector X
    CHECK_CUSPARSE(
        cusparseCreateDnVec(&vecX_crs, crs_mat->n_cols, dx_crs, CUSPARSE_FLOAT_TYPE));

    // Create dense vector y
    CHECK_CUSPARSE(
        cusparseCreateDnVec(&vecY_crs, crs_mat->n_rows, dy_crs, CUSPARSE_FLOAT_TYPE));

    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle_crs, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA_crs, vecX_crs, &beta,
        vecY_crs, CUSPARSE_FLOAT_TYPE, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize_crs));
    CHECK_CUDA(cudaMalloc(&dBuffer_crs, bufferSize_crs));

    CHECK_CUSPARSE(cusparseSpMV(
        handle_crs, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA_crs, vecX_crs, &beta,
        vecY_crs, CUSPARSE_FLOAT_TYPE, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer_crs));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(y_cusparse_crs->val, dy_crs, sizeof(VT) * crs_mat->n_rows, cudaMemcpyDeviceToHost));
    //--------------------------------------------------------------------------
    
    BCRSMatrix<IT, VT> *A_bcrs_square = new BCRSMatrix<IT, VT>();

    // Convert CRS matrix to BCRS
    smax->utils->convert_crs_to_bcrs<IT, VT, ULL>(
        crs_mat->n_rows, crs_mat->n_cols, crs_mat->nnz, crs_mat->col,
        crs_mat->row_ptr, crs_mat->val, A_bcrs_square->n_rows, A_bcrs_square->n_cols,
        A_bcrs_square->n_blocks, A_bcrs_square->b_height, A_bcrs_square->b_width, A_bcrs_square->b_h_pad,
        A_bcrs_square->b_w_pad, A_bcrs_square->col, A_bcrs_square->row_ptr, A_bcrs_square->val, block_dim, block_dim,
        block_dim, block_dim, use_cm);
    
    // cuSPARSE BCRS SpMV
    IT *dA_rowptr_bcrs, *dA_col_bcrs;
    VT *dA_val_bcrs, *dx_bcrs, *dy_bcrs;

    CHECK_CUDA(cudaMalloc((void **)&dA_rowptr_bcrs, (A_bcrs_square->n_rows + 1) * sizeof(IT)));
    CHECK_CUDA(cudaMalloc((void **)&dA_col_bcrs, A_bcrs_square->n_blocks * sizeof(IT)));
    CHECK_CUDA(cudaMalloc((void **)&dA_val_bcrs, A_bcrs_square->n_blocks * block_dim * block_dim * sizeof(VT)));
    CHECK_CUDA(cudaMalloc((void **)&dx_bcrs, A_bcrs_square->n_cols * block_dim * sizeof(VT)));
    CHECK_CUDA(cudaMalloc((void **)&dy_bcrs, A_bcrs_square->n_rows * block_dim * sizeof(VT)));

    CHECK_CUDA(cudaMemcpy(dA_rowptr_bcrs, A_bcrs_square->row_ptr, (A_bcrs_square->n_rows + 1) * sizeof(IT),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_col_bcrs, A_bcrs_square->col, A_bcrs_square->n_blocks * sizeof(IT), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_val_bcrs, A_bcrs_square->val, A_bcrs_square->n_blocks * block_dim * block_dim * sizeof(VT), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dx_bcrs, x->val, A_bcrs_square->n_cols * block_dim * sizeof(VT), cudaMemcpyHostToDevice));

    cusparseHandle_t handle_bcrs = NULL;
    cusparseSpMatDescr_t matA_bcrs;
    cusparseDnVecDescr_t vecX_bcrs, vecY_bcrs;
    cusparseMatDescr_t desc;
    cusparseCreateMatDescr(&desc);
    cusparseSetMatType(desc, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(desc, CUSPARSE_INDEX_BASE_ZERO);

    CHECK_CUSPARSE(cusparseCreate(&handle_bcrs));

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    bsr_mv<IT, VT>(
        handle_bcrs,
        desc,
        (use_cm) ? CUSPARSE_DIRECTION_COLUMN : CUSPARSE_DIRECTION_ROW,
        A_bcrs_square->n_rows, 
        A_bcrs_square->n_cols, 
        A_bcrs_square->n_blocks, 
        dA_val_bcrs, 
        dA_rowptr_bcrs, 
        dA_col_bcrs,
        block_dim, 
        dx_bcrs, 
        dy_bcrs,
        alpha,
        beta
    );

    CUDA_CHECK(cudaMemcpy(y_cusparse_bcrs->val, dy_bcrs, sizeof(VT) * crs_mat->n_rows, cudaMemcpyDeviceToHost));

    //--------------------------------------------------------------------------
    // clang-format on

    // Compare
    compare_spmv<VT>(crs_mat->n_rows, y_smax->val, y_cusparse_crs->val,
                     cli_args->matrix_file_name);
    compare_spmv<VT>(crs_mat->n_rows, y_smax->val, y_cusparse_bcrs->val,
                     cli_args->matrix_file_name);

    delete x;
    delete y_smax;
    delete y_cusparse_crs;
    delete y_cusparse_bcrs;
    delete A_bcrs;

    FINALIZE_SPMV;

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA_crs));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX_crs));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY_crs));
    CHECK_CUSPARSE(cusparseDestroy(handle_crs));
    CHECK_CUSPARSE(cusparseDestroy(handle_bcrs));
    // device memory deallocation
    CHECK_CUDA(cudaFree(dBuffer_crs));
    CHECK_CUDA(cudaFree(dA_rowptr_crs));
    CHECK_CUDA(cudaFree(dA_col_crs));
    CHECK_CUDA(cudaFree(dA_val_crs));
    CHECK_CUDA(cudaFree(dx_crs));
    CHECK_CUDA(cudaFree(dy_crs));
    CHECK_CUDA(cudaFree(dA_rowptr_bcrs));
    CHECK_CUDA(cudaFree(dA_col_bcrs));
    CHECK_CUDA(cudaFree(dA_val_bcrs));
    CHECK_CUDA(cudaFree(dx_bcrs));
    CHECK_CUDA(cudaFree(dy_bcrs));
}