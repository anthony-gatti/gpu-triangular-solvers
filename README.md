# GPU-Accelerated Lower and Upper Triangular Solvers

## Overview
This project implements lower and upper triangular solvers using CUDA, leveraging GPU acceleration. The objective is to compare the performance of direct methods (forward and backward substitution) and iterative methods (Jacobi) and optimize performance by considering cache fetch forward policies.

## Objectives
- Implement lower and upper triangular solvers using CUDA.
- Compare the performance of direct and iterative methods.
- Optimize performance by considering cache fetch forward policies.
- Perform timing analysis to determine the fastest method.

## Implementation Details
The project involves implementing both direct and iterative methods for solving lower and upper triangular systems on the GPU. The steps include:
1. **Reading in a Matrix:** The matrix is read into CSR format.
2. **Allocating Variables:** Necessary variables for the kernel are allocated.
3. **Copying Data:** Relevant data is copied to the device.
4. **Timing:** The execution time is measured for performance analysis.
5. **Triangular Solve:** The triangular solve is performed.
6. **Copying Results:** Data is copied back to the host for analysis.

## Code Structure
- **kernels.cu:** Contains CUDA kernels for forward and backward substitution (direct methods) and Jacobi method (iterative method).
- **read_mtx.cu:** Handles reading matrices from the SuiteSparse Matrix Collection.
- **main.cu:** Drives the triangular solve computation, performs timing analysis, and experiments with cache fetch forward policies.

## Performance Analysis
### Test Setup
- **Matrix Used for Benchmarking:** nlpkkt80
  - Dimensions: 1,062,400 x 1,062,400
  - Non-zero elements: 28,192,672
- **GPU Architecture:** NVIDIA A5000
  - Memory Bandwidth: 768 GB/s

### Results
#### Direct Methods
- **Forward Substitution Execution Time:** 1.5005 ms
- **Backward Substitution Execution Time:** 0.358901 ms

#### Iterative Methods
- **Jacobi Lower Execution Time:** 0.645809 ms
- **Jacobi Upper Execution Time:** 0.646661 ms

### Bandwidth Analysis
- **50% Bandwidth:** 315.807 µs
- **Peak Bandwidth:** 157.9035 µs

### FLOP Analysis
- **50% Bandwidth:** 89.27184 Gflops
- **Peak Bandwidth:** 178.54368 Gflops

### Bandwidth Analysis of Results
#### Direct Methods
- **Forward Substitution Achieved Bandwidth:** 14.47 GB/s
- **Backward Substitution Achieved Bandwidth:** 60.57 GB/s

#### Iterative Methods
- **Jacobi Lower Achieved Bandwidth:** 33.65 GB/s
- **Jacobi Upper Achieved Bandwidth:** 33.62 GB/s

### FLOP Analysis of Results
#### Direct Methods
- **Forward Substitution Achieved FLOPs:** 1.80 GFLOPs
- **Backward Substitution Achieved FLOPs:** 7.53 GFLOPs

#### Iterative Methods
- **Jacobi Lower Achieved FLOPs:** 3.15 GFLOPs
- **Jacobi Upper Achieved FLOPs:** 3.15 GFLOPs

## Cache Fetch Forward Policies
Optimizing direct solvers by accounting for cache fetch forward policies can significantly improve performance by:
- Enhancing temporal locality (reuse of data within a short period).
- Enhancing spatial locality (accessing data elements close to each other in memory).

## How to Run the Code
1. **Download a Testing Matrix**
   Go to [SuiteSparse Matrix Collection](https://sparse.tamu.edu/) and download a matrix to use in the code.
2. **Clone the Repository:**
   ```bash
   git clone <gpu-triangular-solvers>
   cd <gpu-triangular-solvers>
   ```
3. **Compile the Code**
   ```bash
   nvcc -o tri-solve main.cu kernels.cu read_mtx.cu
   ```
4. **Run the Program**
   ```bash
   ./tri-solve
   ```