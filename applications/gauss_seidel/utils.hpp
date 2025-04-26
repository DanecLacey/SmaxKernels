#include <vector>
#include <iostream>

struct CRSMatrix {
    int nnz;
    int n_rows;
    int n_cols;
    std::vector<double> values;      // Non-zero values
    std::vector<int> col;    // Column indices of non-zero values
    std::vector<int> row_ptr;        // Row pointers
};

struct DenseMatrix {
    int n_rows;
    int n_cols;
    std::vector<double> values;
};

DenseMatrix createDenseMatrix(int n_rows, double val){
    DenseMatrix X;
    X.n_rows = n_rows;
    X.n_cols = 1;

    for(int i = 0; i < n_rows; ++i){
        X.values.push_back(val);
    }

    return X;
};

CRSMatrix create1DPoissonMatrixCRS(int n) {
    CRSMatrix A;

    int N = n - 2; // internal nodes (excluding Dirichlet boundaries)

    A.row_ptr.push_back(0);
    
    for (int i = 0; i < N; ++i) {
        // Diagonal entry
        A.values.push_back(2.0);
        A.col.push_back(i);

        // Left neighbor (if not first row)
        if (i > 0) {
            A.values.push_back(-1.0);
            A.col.push_back(i - 1);
        }

        // Right neighbor (if not last row)
        if (i < N - 1) {
            A.values.push_back(-1.0);
            A.col.push_back(i + 1);
        }

        A.row_ptr.push_back(static_cast<int>(A.values.size()));
    }

    A.n_rows = A.row_ptr.back();
    A.n_cols = A.n_rows;
    A.nnz = A.values.size();

    return A;
};

void printCRSMatrix(const CRSMatrix& A) {
    std::cout << "NNZ: " << A.nnz << std::endl;
    std::cout << "N_rows: " << A.n_rows << std::endl;
    std::cout << "N_cols: " << A.n_cols << std::endl;
    std::cout << "Values: ";
    for (double val : A.values)
        std::cout << val << " ";
    std::cout << "\nCol Indices: ";
    for (int col : A.col)
        std::cout << col << " ";
    std::cout << "\nRow Ptr: ";
    for (int ptr : A.row_ptr)
        std::cout << ptr << " ";
    std::cout << std::endl;
};

void subtract_vectors(
    double *result_vec,
    const double *vec1,
    const double *vec2,
    const int N,
    const double scale = 1.0
){
    #pragma omp parallel for
    for (int i = 0; i < N; ++i){
        result_vec[i] = vec1[i] - scale*vec2[i];
    }
};

double infty_vec_norm(
    DenseMatrix X
){
    double max_abs = 0.0;
    double curr_abs;
    for (int i = 0; i < X.n_rows; ++i){
        curr_abs = (X.values[i] >= 0) ? X.values[i]  : -1*X.values[i];
        if ( curr_abs > max_abs){
            max_abs = curr_abs; 
        }
    }

    return max_abs;
};

void extract_D_L_U(const CRSMatrix& A, CRSMatrix& D_plus_L, CRSMatrix& U) {
    int n_rows = A.row_ptr.size() - 1;

    D_plus_L.row_ptr.push_back(0);
    U.row_ptr.push_back(0);

    for (int i = 0; i < n_rows; ++i) {
        int row_start = A.row_ptr[i];
        int row_end = A.row_ptr[i + 1];

        for (int idx = row_start; idx < row_end; ++idx) {
            int col = A.col[idx];
            double val = A.values[idx];

            if (col <= i) {
                // Diagonal or lower triangular
                D_plus_L.values.push_back(val);
                D_plus_L.col.push_back(col);
                ++D_plus_L.nnz;
            } else {
                // Strictly upper triangular
                U.values.push_back(val);
                U.col.push_back(col);
                ++U.nnz;
            }
        }

        D_plus_L.row_ptr.push_back(static_cast<int>(D_plus_L.values.size()));
        U.row_ptr.push_back(static_cast<int>(U.values.size()));
    }
    D_plus_L.n_rows = A.n_rows;
    D_plus_L.n_cols = A.n_rows;
    U.n_rows = A.n_rows;
    U.n_cols = A.n_rows;
};

void spltsv(const CRSMatrix& L, const std::vector<double>& b, std::vector<double>& x) {
    int n = b.size();
    x.resize(n);

    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        double diag = 0.0;

        int row_start = L.row_ptr[i];
        int row_end = L.row_ptr[i + 1];

        for (int idx = row_start; idx < row_end; ++idx) {
            int col = L.col[idx];
            double val = L.values[idx];

            std::cout << "col = " << col << std::endl;
            std::cout << "val = " << val << std::endl;

            if (col < i) {
                sum += val * x[col];  // Known values
            } else if (col == i) {
                diag = val;
            } else {
                // Should not happen for lower-triangular matrix
                continue;
            }
        }

        // if (diag == 0.0) {
        //     throw std::runtime_error("Zero diagonal encountered in lower triangular solve.");
        // }

        x[i] = (b[i] - sum) / diag;
    }
};

void spmv(const CRSMatrix& A, const std::vector<double>& x, std::vector<double>& y) {
    int n_rows = A.row_ptr.size() - 1;
    y.assign(n_rows, 0.0);  // Initialize output vector

    for (int i = 0; i < n_rows; ++i) {
        double sum = 0.0;
        int row_start = A.row_ptr[i];
        int row_end = A.row_ptr[i + 1];

        for (int idx = row_start; idx < row_end; ++idx) {
            int col = A.col[idx];
            sum += A.values[idx] * x[col];
        }

        y[i] = sum;
    }
}