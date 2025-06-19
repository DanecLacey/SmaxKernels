#pragma once

#include "common.hpp"

namespace SMAX {

class Kernel {
  private:
    // Type erasure helpers
    template <typename T> Variant wrap_arg(T **ptr) {
        return reinterpret_cast<void **>(ptr);
    }

    template <typename T> Variant wrap_arg(T *ptr) {
        return static_cast<void *>(ptr);
    }

    Variant wrap_arg(int val) { return val; }

  public:
    std::unique_ptr<KernelContext> k_ctx;
    Timers *timers;

    Kernel(std::unique_ptr<KernelContext> _k_ctx) : k_ctx(std::move(_k_ctx)) {
        this->timers = new Timers;
#ifdef USE_TIMERS
        CREATE_SMAX_STOPWATCH(initialize)
        CREATE_SMAX_STOPWATCH(apply)
        CREATE_SMAX_STOPWATCH(finalize)
#endif
    }
    virtual ~Kernel() { delete this->timers; };

    // Methods to override
    virtual int initialize(ULL A_offset, ULL B_offset, ULL C_offset) = 0;
    virtual int apply(ULL A_offset, ULL B_offset, ULL C_offset) = 0;
    virtual int finalize(ULL A_offset, ULL B_offset, ULL C_offset) = 0;
    virtual int _register_A(const std::vector<Variant> &args) = 0;
    virtual int _register_B(const std::vector<Variant> &args) = 0;
    virtual int _register_C(const std::vector<Variant> &args) = 0;

    // Flag setters to override
    virtual int set_mat_perm(bool) {
        std::cerr << "set_mat_perm not supported for this kernel.\n";
        return 1;
    }
    virtual int set_mat_upper_triang(bool) {
        std::cerr << "set_mat_upper_triang not supported for this kernel.\n";
        return 1;
    }
    virtual int set_mat_lower_triang(bool) {
        std::cerr << "set_mat_lower_triang not supported for this kernel.\n";
        return 1;
    }
    virtual int set_vec_row_major(bool) {
        std::cerr << "set_vec_row_major not supported for this kernel.\n";
        return 1;
    }
    virtual int set_mat_scs(bool) {
        std::cerr << "set_mat_scs not supported for this kernel.\n";
        return 1;
    }

    // Swapping utility to override
    virtual int swap_operands(void) {
        std::cerr << "swap_operands not supported for this kernel.\n";
        return 1;
    }

    // User-facing helpers
    int run(ULL A_offset = 0, ULL B_offset = 0, ULL C_offset = 0) {
        // DL 09.05.2025 NOTE: Since only a single memory address is
        // registered, the offsets are to access other parts of memory at
        // runtime (e.g. register A to kernel, but A[i*n_rows] needed)
        // DL 18.06.2025 NOTE: an int can be automatically promoted to unsigned
        // long long int when passed by value, so this is okay
        CHECK_ERROR(initialize(A_offset, B_offset, C_offset), "initialize");
        CHECK_ERROR(apply(A_offset, B_offset, C_offset), "apply");
        CHECK_ERROR(finalize(A_offset, B_offset, C_offset), "finalize");

        return 0;
    }

    template <typename... Args> int register_A(Args &&...args) {
        std::vector<Variant> packed_args{wrap_arg(std::forward<Args>(args))...};
        return _register_A(packed_args);
    }

    template <typename... Args> int register_B(Args &&...args) {
        std::vector<Variant> packed_args{wrap_arg(std::forward<Args>(args))...};
        return _register_B(packed_args);
    }

    template <typename... Args> int register_C(Args &&...args) {
        std::vector<Variant> packed_args{wrap_arg(std::forward<Args>(args))...};
        return _register_C(packed_args);
    }
};

} // namespace SMAX
