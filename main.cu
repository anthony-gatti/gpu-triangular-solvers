#include <cuda_runtime.h>
#include <iostream>
#include "kernels.cuh"

// CUDA kernel declarations for direct solvers
__global__ void forward_substitution_kernel(int n, const int *row_ptr, const int *col_ind, const double *values, const double *b, double *x);
__global__ void backward_substitution_kernel(int n, const int *row_ptr, const int *col_ind, const double *values, const double *b, double *x);
__global__ void jacobi_lower_kernel(int n, const int *row_ptr, const int *col_ind, const double *val, const double *b, double *x, double *x_new);
__global__ void jacobi_upper_kernel(int n, const int *row_ptr, const int *col_ind, const double *val, const double *b, double *x, double *x_new);

// Utility functions
void read_matrix(const char* file_path, int*& row_ptr, int*& col_ind, double*& values, int& num_rows, int& num_cols);
void forward_substitution(int *row_ptr, int *col_ind, double *values, double *b, double *x, int num_rows, float& time);
void backward_substitution(int *row_ptr, int *col_ind, double *values, double *b, double *x, int num_rows, float& time);
void jacobi(int *row_ptr, int *col_ind, double *values, double *b, double *x, int num_rows, bool is_upper, float& time);

int main(int argc, char** argv) {
    cudaSetDevice(1);

    // Read matrix from file
    int *row_ptr, *col_ind, num_rows, num_cols;
    double *values, *b, *x;

    read_matrix("./nlpkkt80.mtx", row_ptr, col_ind, values, num_rows, num_cols);

    // Allocate memory for vectors b and x
    b = new double[num_rows];
    x = new double[num_rows];
    // Initialize vector b with some values
    for (int i = 0; i < num_rows; ++i) b[i] = 1.0;

    float time_forward, time_backward, time_jacobi_lower, time_jacobi_upper, time_forward_total, time_backward_total, time_jacobi_lower_total, time_jacobi_upper_total, time_forward_average, time_backward_average, time_jacobi_lower_average, time_jacobi_upper_average;

    // Perform forward substitution (lower triangular solver)
    time_forward_total = 0;
    for(int i = 0; i < 100; i++ ) {
        forward_substitution(row_ptr, col_ind, values, b, x, num_rows, time_forward);
        time_forward_total += time_forward;
    }

    // Perform backward substitution (upper triangular solver)
    time_backward_total = 0;
    for(int i = 0; i < 100; i++ ) {
        backward_substitution(row_ptr, col_ind, values, b, x, num_rows, time_backward);
        time_backward_total += time_backward;
    }

    // Perform Jacobi method for lower triangular solver
    time_jacobi_lower_total = 0;
    for (int i = 0; i < 100; i++) {
        jacobi(row_ptr, col_ind, values, b, x, num_rows, false, time_jacobi_lower);
        time_jacobi_lower_total += time_jacobi_lower;
    }

    // Perform Jacobi method for upper triangular solver
    time_jacobi_upper_total = 0;
    for (int i = 0; i < 100; i++) {
        jacobi(row_ptr, col_ind, values, b, x, num_rows, true, time_jacobi_upper);
        time_jacobi_upper_total += time_jacobi_upper;
    }

    // Calculate average times
    time_forward_average = time_forward_total / 100;
    time_backward_average = time_backward_total / 100;
    time_jacobi_lower_average = time_jacobi_lower_total / 100;
    time_jacobi_upper_average = time_jacobi_upper_total / 100;

    std::cout << "Average Forward Substitution Execution Time: " << time_forward_average << " ms" << std::endl;
    std::cout << "Average Backward Substitution Execution Time: " << time_backward_average << " ms" << std::endl;
    std::cout << "Average Jacobi Lower Execution Time: " << time_jacobi_lower_average << " ms" << std::endl;
    std::cout << "Average Jacobi Upper Execution Time: " << time_jacobi_upper_average << " ms" << std::endl;

    // Clean up
    delete[] row_ptr;
    delete[] col_ind;
    delete[] values;
    delete[] b;
    delete[] x;

    return 0;
}

void forward_substitution(int *row_ptr, int *col_ind, double *values, double *b, double *x, int num_rows, float& time) {
    int *d_row_ptr, *d_col_ind;
    double *d_values, *d_b, *d_x;

    cudaMalloc(&d_row_ptr, (num_rows + 1) * sizeof(int));
    cudaMalloc(&d_col_ind, row_ptr[num_rows] * sizeof(int));
    cudaMalloc(&d_values, row_ptr[num_rows] * sizeof(double));
    cudaMalloc(&d_b, num_rows * sizeof(double));
    cudaMalloc(&d_x, num_rows * sizeof(double));

    cudaMemcpy(d_row_ptr, row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, col_ind, row_ptr[num_rows] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, row_ptr[num_rows] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, num_rows * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, num_rows * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (num_rows + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    forward_substitution_kernel<<<numBlocks, blockSize>>>(num_rows, d_row_ptr, d_col_ind, d_values, d_b, d_x);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaMemcpy(x, d_x, num_rows * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_values);
    cudaFree(d_b);
    cudaFree(d_x);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void backward_substitution(int *row_ptr, int *col_ind, double *values, double *b, double *x, int num_rows, float& time) {
    int *d_row_ptr, *d_col_ind;
    double *d_values, *d_b, *d_x;

    cudaMalloc(&d_row_ptr, (num_rows + 1) * sizeof(int));
    cudaMalloc(&d_col_ind, row_ptr[num_rows] * sizeof(int));
    cudaMalloc(&d_values, row_ptr[num_rows] * sizeof(double));
    cudaMalloc(&d_b, num_rows * sizeof(double));
    cudaMalloc(&d_x, num_rows * sizeof(double));

    cudaMemcpy(d_row_ptr, row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, col_ind, row_ptr[num_rows] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, row_ptr[num_rows] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, num_rows * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, num_rows * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (num_rows + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    backward_substitution_kernel<<<numBlocks, blockSize>>>(num_rows, d_row_ptr, d_col_ind, d_values, d_b, d_x);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaMemcpy(x, d_x, num_rows * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_values);
    cudaFree(d_b);
    cudaFree(d_x);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void jacobi(int *row_ptr, int *col_ind, double *values, double *b, double *x, int num_rows, bool is_upper, float& time) {
    int *d_row_ptr, *d_col_ind;
    double *d_values, *d_b, *d_x, *d_x_new;

    cudaMalloc(&d_row_ptr, (num_rows + 1) * sizeof(int));
    cudaMalloc(&d_col_ind, row_ptr[num_rows] * sizeof(int));
    cudaMalloc(&d_values, row_ptr[num_rows] * sizeof(double));
    cudaMalloc(&d_b, num_rows * sizeof(double));
    cudaMalloc(&d_x, num_rows * sizeof(double));
    cudaMalloc(&d_x_new, num_rows * sizeof(double));

    cudaMemcpy(d_row_ptr, row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, col_ind, row_ptr[num_rows] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, row_ptr[num_rows] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, num_rows * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, num_rows * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (num_rows + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int max_iterations = 1000;
    for (int iter = 0; iter < max_iterations; iter++) {
        cudaEventRecord(start, 0);

        if (is_upper) {
            jacobi_upper_kernel<<<numBlocks, blockSize>>>(num_rows, d_row_ptr, d_col_ind, d_values, d_b, d_x, d_x_new);
        } else {
            jacobi_lower_kernel<<<numBlocks, blockSize>>>(num_rows, d_row_ptr, d_col_ind, d_values, d_b, d_x, d_x_new);
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaMemcpy(x, d_x_new, num_rows * sizeof(double), cudaMemcpyDeviceToHost);
        cudaEventElapsedTime(&time, start, stop);

        // Swap pointers for the next iteration
        double *temp = d_x;
        d_x = d_x_new;
        d_x_new = temp;
    }

    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_values);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_x_new);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}