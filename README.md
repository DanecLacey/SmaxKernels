# SmaxKernels

**S**parse **MA**trix mpi+**X** **Kernels** is a lightweight C++ library providing high-performance kernel implementations across multiple platforms.

### Features ###
* Clean, minimalist library interface 
* Efficient implementations of:
    * Sparse Matrix-(Multiple)Vector Multiplication (SpMV)
    * Sparse Matrix-Matrix Multiplication (SpGEMM)
* Stacked timers around key regions
* Supports multiple integer and floating-point types
* Designed for integration into larger numerical frameworks

### Building SmaxKernels ###
```bash
git clone https://github.com/DanecLacey/SmaxKernels.git
cd SmaxKernels
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=<install path>
make install
```

### Usage Examples ###
A few usage examples are provided in the `/tests` directory.
To build the tests:
```bash
cd SmaxKernels/tests
make INSTALL_PATH=<install path>
```

### Contributing ###
Pull requests and issues are welcome. Please open an issue first to propose changes or feature additions.