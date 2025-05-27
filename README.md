# SmaxKernels

**S**parse **MA**trix mpi+**X** **Kernels** is a lightweight, portable C++ library providing high-performance implementations of popular sparse matrix kernels of the form `C = A op B`.

## Supported Kernels, Parallel Frameworks, and Formats ## 
| Sparse Kernel                                     | OpenMP    | CUDA     | MPI       | Matrix Formats |
|--------------                                     |---------- |----------|---------- | ----------|
| Matrix-Vector Multiplication -- **SpMV**          | ✅       | ✅       | ❌       | CRS |
| Matrix-Multiple Vector Multiplication -- **SpMM** | ✅       | ❌       | ❌       | CRS |
| Matrix-Sparse Matrix Multiplication -- **SpGEMM** | ✅       | ❌       | ❌       | CRS |
| Triangular Solve -- **SpTRSV**                    | ✅       | ❌       | ❌       | CRS |
| Batched Triangular Solve -- **SpTRSM**            | ✅       | ❌       | ❌       | CRS |

## Features ## 
* Clean, minimalist library interface
* Stacked timers around key regions
* Supports signed and unsigned **16-, 32-, and 64-bit integers**, as well as **32- and 64-bit floating-point types**

## Building SmaxKernels ##
```bash
git clone https://github.com/DanecLacey/SmaxKernels.git
cd SmaxKernels
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=<INSTALL_PATH>
make install -j
```

## Usage Examples ##
Basic usage examples are provided in the `/examples` directory.
```bash
cd ../examples
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=<INSTALL_PATH> 
make -j
```
* Very basic demonstrations of the API are provided in `/examples/demos`.
* More realistic examples can be found in `/examples/applications`.
* There are pre-made benchmarks of the kernels provided by SmaxKernels, as well as optional third party [[Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page), [MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html), [PETSc](https://petsc.org/release/)] kernels to compare against, which are all found in `/examples/benchmarks`.
* Validation of Smax kernels against MKL kernels are found in `/examples/validation` (requires MKL).
* Unit tests are found in `/examples/tests`

## Notice ##
This project is very much still in development, and many features may be unfinished, broken, or subject to change.
* This project requires C++17 features
* It is assumed that all optional third party libraries are installed in `$INSTALL_PATH`
* The PETSc library is found via. the PkgConfig module. So if benchmarking PETSc kernels, you should configure PETSc with `--with-pkg-config=1` when building.

## Contributing ##
Pull requests and issues are welcome. Please open an issue first to propose changes or feature additions.