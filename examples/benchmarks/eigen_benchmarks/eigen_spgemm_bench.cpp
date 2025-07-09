#include "../../examples_common.hpp"
#include "../../spgemm_helpers.hpp"
#include "../benchmarks_common.hpp"
#include "eigen_benchmarks_common.hpp"

// Specialize for SpGEMM
#undef MIN_NUM_ITERS
#define MIN_NUM_ITERS 10

int main(int argc, char *argv[]) {

    // Set datatypes
    using IT = int;
    using VT = double;

    // Just takes pinning overhead away from timers
    init_pin();

    // Setup data structures
    INIT_SPGEMM(IT, VT);
    CRSMatrix<IT, VT> *crs_mat_C = new CRSMatrix<IT, VT>();

    Eigen::SparseMatrix<VT> eigen_A(crs_mat_A->n_rows, crs_mat_A->n_cols);

    int B_n_rows, B_n_cols;
    if (compute_AA) {
        B_n_rows = crs_mat_A->n_rows;
        B_n_cols = crs_mat_A->n_cols;
    } else {
        B_n_rows = crs_mat_B->n_rows;
        B_n_cols = crs_mat_B->n_cols;
    }
    Eigen::SparseMatrix<VT> eigen_B(B_n_rows, B_n_cols);
    {
        std::vector<Eigen::Triplet<VT>> triA;
        triA.reserve(crs_mat_A->nnz);
        for (IT i = 0; i < crs_mat_A->n_rows; ++i)
            for (IT k = crs_mat_A->row_ptr[i]; k < crs_mat_A->row_ptr[i + 1];
                 ++k)
                triA.emplace_back(i, crs_mat_A->col[k], crs_mat_A->val[k]);
        eigen_A.setFromTriplets(triA.begin(), triA.end());
    }
    {
        std::vector<Eigen::Triplet<VT>> triB;
        if (compute_AA) {
            triB.reserve(crs_mat_A->nnz);
            for (IT i = 0; i < crs_mat_A->n_rows; ++i)
                for (IT k = crs_mat_A->row_ptr[i];
                     k < crs_mat_A->row_ptr[i + 1]; ++k)
                    triB.emplace_back(i, crs_mat_A->col[k], crs_mat_A->val[k]);
            eigen_B.setFromTriplets(triB.begin(), triB.end());
        } else {
            std::vector<Eigen::Triplet<VT>> triB;
            triB.reserve(crs_mat_B->nnz);
            for (IT i = 0; i < crs_mat_B->n_rows; ++i)
                for (IT k = crs_mat_B->row_ptr[i];
                     k < crs_mat_B->row_ptr[i + 1]; ++k)
                    triB.emplace_back(i, crs_mat_B->col[k], crs_mat_B->val[k]);
            eigen_B.setFromTriplets(triB.begin(), triB.end());
        }
    }

    Eigen::SparseMatrix<VT> eigen_C(crs_mat_A->n_rows, crs_mat_B->n_cols);

    // Setup benchmark harness
    std::string bench_name = "eigen_spgemm";
    SETUP_BENCH(bench_name);

    std::function<void()> lambda = [bench_name, &eigen_A, &eigen_B,
                                    &eigen_C]() {
        eigen_C = eigen_A * eigen_B;
    };

    // Execute benchmark and print results
    RUN_BENCH;
    PRINT_SPGEMM_BENCH(eigen_C.nonZeros());

    // Clean up
    FINALIZE_SPGEMM;

#ifdef USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif
}