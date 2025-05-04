#ifndef VALIDATION_COMMON_HPP
#define VALIDATION_COMMON_HPP

#include "SmaxKernels/interface.hpp"
#include <mkl.h>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <iomanip>
#include <math.h>

#define PRINT_WIDTH 18

double compute_euclid_dist(const int n_rows, const double *y_SMAX,
                           const double *y_MKL) {
    double tmp = 0.0;

#pragma omp parallel for reduction(+ : tmp)
    for (int i = 0; i < n_rows; ++i) {
        tmp += (y_SMAX[i] - y_MKL[i]) * (y_SMAX[i] - y_MKL[i]);
    }

    return std::sqrt(tmp);
}

void compare_spmv(const int n_rows, const double *y_SMAX, const double *y_MKL,
                  const std::string mtx_name) {

    std::fstream working_file;
    std::string output_filename = "compare_spmv.txt";
    working_file.open(output_filename,
                      std::fstream::in | std::fstream::out | std::fstream::app);

    int n_threads = 1;
#ifdef _OPENMP
#pragma omp parallel
    {
        n_threads = omp_get_num_threads();
    }
#endif

    working_file << mtx_name << " with " << n_threads << " thread(s)"
                 << std::endl;

    double relative_diff, max_relative_diff, max_relative_diff_elem_SMAX,
        max_relative_diff_elem_MKL;
    relative_diff = max_relative_diff = max_relative_diff_elem_SMAX =
        max_relative_diff_elem_MKL = 0.0;
    double absolute_diff, max_absolute_diff, max_absolute_diff_elem_SMAX,
        max_absolute_diff_elem_MKL;
    absolute_diff = max_absolute_diff = max_absolute_diff_elem_SMAX =
        max_absolute_diff_elem_MKL = 0.0;

    // Print header
#if VERBOSITY == 0
    working_file << std::left << std::setw(PRINT_WIDTH)
                 << "MKL rel. elem:" << std::left << std::setw(PRINT_WIDTH)
                 << "SMAX rel. elem:" << std::left << std::setw(PRINT_WIDTH)
                 << "lrgst rel. (%):" << std::left << std::setw(PRINT_WIDTH)
                 << "MKL abs. elem:" << std::left << std::setw(PRINT_WIDTH)
                 << "SMAX abs. elem:" << std::left << std::setw(PRINT_WIDTH)
                 << "lrgst abs.:" << std::left << std::setw(PRINT_WIDTH)
                 << "||MKL - SMAX||" << std::left << std::setw(PRINT_WIDTH)
                 << std::endl;

    working_file << std::left << std::setw(PRINT_WIDTH) << "-------------"
                 << std::left << std::setw(PRINT_WIDTH) << "---------------"
                 << std::left << std::setw(PRINT_WIDTH) << "-------------"
                 << std::left << std::setw(PRINT_WIDTH) << "-------------"
                 << std::left << std::setw(PRINT_WIDTH) << "---------------"
                 << std::left << std::setw(PRINT_WIDTH) << "-----------"
                 << std::left << std::setw(PRINT_WIDTH) << "---------------"
                 << std::endl;
#elif VERBOSITY == 1
    int n_result_digits = n_rows > 0 ? (int)log10((double)n_rows) + 1 : 1;

    working_file << std::left << std::setw(n_result_digits + 8)
                 << "vec idx:" << std::left << std::setw(n_result_digits + 8)
                 << "row idx:" << std::left << std::setw(PRINT_WIDTH)
                 << "MKL results:" << std::left << std::setw(PRINT_WIDTH)
                 << "SMAX results:" << std::left << std::setw(PRINT_WIDTH)
                 << "rel. diff(%):" << std::left << std::setw(PRINT_WIDTH)
                 << "abs. diff:" << std::endl;

    working_file << std::left << std::setw(n_result_digits + 8) << "--------"
                 << std::left << std::setw(n_result_digits + 8) << "--------"
                 << std::left << std::setw(PRINT_WIDTH) << "-----------"
                 << std::left << std::setw(PRINT_WIDTH) << "------------"
                 << std::left << std::setw(PRINT_WIDTH) << "------------"
                 << std::left << std::setw(PRINT_WIDTH) << "---------"
                 << std::endl;
#endif

    // Print comparison
    int vec_count = 0;
    for (int i = 0; i < n_rows; ++i) {

        relative_diff = std::abs(y_MKL[i] - y_SMAX[i]) / y_MKL[i];
        absolute_diff = std::abs(y_MKL[i] - y_SMAX[i]);

#if VERBOSITY == 1
        // Protect against printing 'inf's
        if (std::abs(y_MKL[i]) < 1e-25) {
            relative_diff = y_SMAX[i];
        }

        working_file << std::left << std::setw(n_result_digits + 8) << vec_count
                     << std::left << std::setw(n_result_digits + 8)
                     << (i - (n_rows * vec_count)) << std::left
                     << std::setprecision(8) << std::scientific
                     << std::setw(PRINT_WIDTH) << y_MKL[i] << std::left
                     << std::setw(PRINT_WIDTH) << y_SMAX[i] << std::left
                     << std::setw(PRINT_WIDTH) << 100 * relative_diff
                     << std::left << std::setw(PRINT_WIDTH) << absolute_diff;

        if ((std::abs(relative_diff) > .01) || std::isinf(relative_diff)) {
            working_file << std::left << std::setw(PRINT_WIDTH) << "ERROR";
        } else if (std::abs(relative_diff) > .0001) {
            working_file << std::left << std::setw(PRINT_WIDTH) << "WARNING";
        }

        working_file << std::endl;

#elif VERBOSITY == 0

        if (relative_diff > max_relative_diff) {
            max_relative_diff = relative_diff;
            max_relative_diff_elem_SMAX = y_SMAX[i];
            max_relative_diff_elem_MKL = y_MKL[i];
        }
        if (absolute_diff > max_absolute_diff) {
            max_absolute_diff = absolute_diff;
            max_absolute_diff_elem_SMAX = y_SMAX[i];
            max_absolute_diff_elem_MKL = y_MKL[i];
        }
#endif
        // increments RHS vector counting for block x_vector
        if ((i + 1) % n_rows == 0 && i > 0)
            ++vec_count;
    }

#if VERBOSITY == 0
    working_file << std::scientific << std::left << std::setw(PRINT_WIDTH)
                 << max_relative_diff_elem_MKL << std::left
                 << std::setw(PRINT_WIDTH) << max_relative_diff_elem_SMAX
                 << std::left << std::setw(PRINT_WIDTH)
                 << 100 * max_relative_diff << std::left
                 << std::setw(PRINT_WIDTH) << max_absolute_diff_elem_MKL
                 << std::left << std::setw(PRINT_WIDTH)
                 << max_absolute_diff_elem_SMAX << std::left
                 << std::setw(PRINT_WIDTH) << max_absolute_diff << std::left
                 << std::setw(PRINT_WIDTH + 6)
                 << compute_euclid_dist(n_rows, y_SMAX, y_MKL);

    if (((std::abs(max_relative_diff) > .01) || std::isnan(max_relative_diff) ||
         std::isinf(max_relative_diff)) ||
        (std::isnan(max_absolute_diff) || std::isinf(max_absolute_diff))) {
        working_file << std::left << std::setw(PRINT_WIDTH) << "ERROR";
    } else if (std::abs(max_relative_diff) > .0001) {
        working_file << std::left << std::setw(PRINT_WIDTH) << "WARNING";
    }

    working_file << std::endl;
#endif
    working_file << "\n";
    working_file.close();
}

#endif // VALIDATION_COMMON_HPP