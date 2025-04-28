# SmaxKernels

**S**parse **MA**trix mpi+**X** **Kernels** is a lightweight, portable C++ library providing high-performance implementations of sparse matrix kernels.

## Features ##
* Clean, minimalist library interface 
* Efficient implementations of Sparse:
    * Matrix-(Multiple) Vector Multiplication (SpMV)
    * Matrix-Matrix Multiplication (SpGEMM)
    * Triangular Solve (SpTSV)
* Stacked timers around key regions
* Supports multiple integer and floating-point types

## Building SmaxKernels ##
```bash
git clone https://github.com/DanecLacey/SmaxKernels.git
cd SmaxKernels
export INSTALL_PATH=<install path>
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH
make && make install
```

## Usage Examples ##
```bash
cd examples
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$INSTALL_PATH
make
```
A few basic usage examples are provided in the `/tests` directory. There are pre-made benchmarks of `Smax` kernels, as well as optional third party kernels, which all in `/benchmarks`. More realistic examples can be found in `/applications`.

## Notice ##
This project is very much still in development, and many features may be unfinished or broken.
* As of 2025.04.28 Only CPU-OpenMP implementations of SpMV, SpGEMM, and SpTSV are publicly available
* MPK, SpADD, SpTR kernels are in progress, as well as MPI functionality

## Contributing ##
Pull requests and issues are welcome. Please open an issue first to propose changes or feature additions.