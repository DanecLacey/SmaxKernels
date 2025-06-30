#ifndef SPARSE_MATRIX_BASE_HPP
#define SPARSE_MATRIX_BASE_HPP
#pragma once

#include <iomanip>
#include <iostream>

template <typename IT> class sparsematrix {
  public:
    sparsematrix(IT n_rows = 0, IT n_cols = 0, IT n_nz = 0, bool is_sorted = false,
                 bool is_symmetric = false)
        : nrows_(n_rows), ncols_(n_cols), nnz_(n_nz), is_sorted_(is_sorted),
          is_symmetric_(is_symmetric) {}
    IT rows() const { return nrows_; }
    IT cols() const { return ncols_; }
    IT nnz() const { return nnz_; }
    bool is_symmetric() const { return is_symmetric_; }
    bool is_sorted() const { return is_sorted_; }
    virtual void print(void) {
        std::cout << std::boolalpha;
        std::cout << "is_sorted = " << this->is_sorted_ << "\n";
        std::cout << "is_symmetric = " << this->is_symmetric_ << "\n";
        std::cout << std::noboolalpha;
        std::cout << "NNZ: " << this->nnz_ << "\n";
        std::cout << "N_rows: " << this->nrows_ << "\n";
        std::cout << "N_cols: " << this->ncols_ << std::endl;
    }
    virtual ~sparsematrix() = default;

  protected:
    IT nrows_;
    IT ncols_;
    IT nnz_;
    bool is_sorted_;
    bool is_symmetric_;
};

#endif