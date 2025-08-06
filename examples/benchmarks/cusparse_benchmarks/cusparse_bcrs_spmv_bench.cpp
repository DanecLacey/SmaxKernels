#include "../../examples_common.hpp"
#include "../../spmv_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "cusparse_benchmarks_common.hpp"

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

    auto smax = std::make_unique<SMAX::Interface>();

    // Setup data structures
    INIT_SPMV(IT, VT);
    IT block_dim = cli_args->_hpad;
    printf("Warning, cuSPARSE accepts only square blocks. Fixing "
           "block_dim=hpad=%d\n",
           block_dim);
    bool use_cm = cli_args->_use_cm;
    std::string custom_kernel = cli_args->_ck;

    // Declare bcrs operand
    BCRSMatrix<IT, VT> *A_bcrs = new BCRSMatrix<IT, VT>();

    // Convert CRS matrix to BCRS
    smax->utils->convert_crs_to_bcrs<IT, VT, ULL>(
        crs_mat->n_rows, crs_mat->n_cols, crs_mat->nnz, crs_mat->col,
        crs_mat->row_ptr, crs_mat->val, A_bcrs->n_rows, A_bcrs->n_cols,
        A_bcrs->n_blocks, A_bcrs->b_height, A_bcrs->b_width, A_bcrs->b_h_pad,
        A_bcrs->b_w_pad, A_bcrs->col, A_bcrs->row_ptr, A_bcrs->val, block_dim,
        block_dim, block_dim, block_dim, use_cm);

    auto h_x = std::make_unique<DenseMatrix<VT>>(crs_mat->n_cols, 1, 1.0);
    auto h_y = std::make_unique<DenseMatrix<VT>>(crs_mat->n_rows, 1, 0.0);

    // clang-format off
    //--------------------------------------------------------------------------
    // Device memory management
    IT *d_bsrRowPtr, *d_bsrColInd;
    VT *d_bsrVal, *d_x, *d_y;

    CHECK_CUDA(cudaMalloc((void**)&d_bsrRowPtr, (A_bcrs->n_rows+1) * sizeof(IT)));
    CHECK_CUDA(cudaMalloc((void**)&d_bsrColInd, A_bcrs->n_blocks * sizeof(IT)));
    CHECK_CUDA(cudaMalloc((void**)&d_bsrVal, A_bcrs->n_blocks * block_dim * block_dim * sizeof(VT)));
    CHECK_CUDA(cudaMalloc((void**)&d_x, A_bcrs->n_cols * block_dim * sizeof(VT)));
    CHECK_CUDA(cudaMalloc((void**)&d_y, A_bcrs->n_rows * block_dim * sizeof(VT)));

    CHECK_CUDA(cudaMemcpy(d_bsrRowPtr, A_bcrs->row_ptr, (A_bcrs->n_rows+1)*sizeof(IT), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bsrColInd, A_bcrs->col, A_bcrs->n_blocks*sizeof(IT), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bsrVal, A_bcrs->val, A_bcrs->n_blocks * block_dim * block_dim * sizeof(VT), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x->val, A_bcrs->n_cols * block_dim * sizeof(VT), cudaMemcpyHostToDevice));
    //--------------------------------------------------------------------------
    // clang-format on

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    cusparseMatDescr_t desc;
    cusparseCreateMatDescr(&desc);
    cusparseSetMatType(desc, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(desc, CUSPARSE_INDEX_BASE_ZERO);
    VT alpha = 1.0;
    VT beta = 0.0;

    CHECK_CUSPARSE(cusparseCreate(&handle));

    std::string bench_name = "cusparse_bcrs_cuda_spmv";
    SETUP_BENCH;

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // clang-format off
    std::function<void()> lambda = [&]() {
        bsr_mv<IT, VT>(
            handle,
            desc, 
            (use_cm) ? CUSPARSE_DIRECTION_COLUMN : CUSPARSE_DIRECTION_ROW,
            A_bcrs->n_rows, 
            A_bcrs->n_cols, 
            A_bcrs->n_blocks, 
            d_bsrVal, 
            d_bsrRowPtr, 
            d_bsrColInd,
            block_dim, 
            d_x, 
            d_y,
            alpha,
            beta
        );
    };
    // clang-format on
    //--------------------------------------------------------------------------
    CHECK_CUDA(cudaDeviceSynchronize());

    // Execute benchmark and print results
    RUN_BENCH;
    PRINT_SPMV_BENCH;

    // Clean up
    FINALIZE_SPMV;
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroy(handle));
    // device memory deallocation
    CHECK_CUDA(cudaFree(d_bsrRowPtr));
    CHECK_CUDA(cudaFree(d_bsrColInd));
    CHECK_CUDA(cudaFree(d_bsrVal));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
}