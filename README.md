# SmaxKernels

**S**parse **MA**trix mpi+**X** **Kernels** is a lightweight, portable C++ library providing high-performance implementations of sparse matrix kernels of the form `C = A op B`.

## Features ##
* Clean, minimalist library interface 
* Efficient implementations of Sparse:
    * Matrix-Vector Multiplication -- **SpMV**
    * Matrix-Multiple Vector Multiplication -- **SpMM**
    * Matrix-Sparse Vector Multiplication -- **SpGEMV**
    * Matrix-Sparse Matrix Multiplication -- **SpGEMM**
    * Triangular Solve -- **SpTRSV**
    * Batched Triangular Solve -- **SpTRSM**
* Stacked timers around key regions
* Supports multiple integer and floating-point types

## Building SmaxKernels ##
```bash
git clone https://github.com/DanecLacey/SmaxKernels.git
cd SmaxKernels
export INSTALL_PATH=<install path>
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH
make install
```

## Usage Examples ##
Basic usage examples are provided in the `/examples` directory.
```bash
cd ../examples
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$INSTALL_PATH
make
```
* Very basic examples are provided in `/examples/tests`.
* More realistic examples can be found in `/examples/applications`.
* There are pre-made benchmarks of the kernels provided by SmaxKernels, as well as optional third party [[Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page), [MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html), [PETSc](https://petsc.org/release/)] kernels to compare against, which are all found in `/examples/benchmarks`.
* Validation of Smax kernels against MKL kernels are found in `/examples/validation` (requires MKL).

## Notice ##
This project is very much still in development, and many features may be unfinished, broken, or subject to change.
* As of 2025.05.05, Only CPU-OpenMP implementations of SpMV/M, SpGEMV/M, and SpTRSV/M are publicly available
<!-- * MPK, SpADD, SpTRSP kernels are in progress, as well as GPU and MPI functionality -->
* It is assumed that all optional third party libraries are installed in `$INSTALL_PATH`
* The PETSc library is found via. the PkgConfig module. So if benchmarking PETSc kernels, you should configure PETSc with `--with-pkg-config=1` when building.

## Contributing ##
Pull requests and issues are welcome. Please open an issue first to propose changes or feature additions.