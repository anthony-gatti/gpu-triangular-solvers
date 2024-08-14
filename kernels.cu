__global__ void forward_substitution_kernel(int n, const int *row_ptr, const int *col_ind, const double *val, const double *b, double *x) {
    __shared__ double shared_x[6144];
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    shared_x[threadIdx.x] = x[row];
    __syncthreads();

    double sum = 0.0;
    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    double diag_val = 0.0;

    for (int i = start; i < end; i++) {
        int col = col_ind[i];
        if (col < row) {
            sum += val[i] * shared_x[col % blockDim.x];
        } else if (col == row) {
            diag_val = val[i];
        }
    }

    // Ensure the diagonal value is non-zero to avoid division by zero
    if (diag_val != 0.0) {
        x[row] = (b[row] - sum) / diag_val;
        shared_x[threadIdx.x] = x[row];
    }
    
    __syncthreads();
}

__global__ void backward_substitution_kernel(int n, const int *row_ptr, const int *col_ind, const double *val, const double *b, double *x) {
    __shared__ double shared_x[6144];

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    shared_x[threadIdx.x] = x[row];
    __syncthreads();

    double sum = 0.0;
    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    double diag_val = 0.0;

    for (int i = start; i < end; i++) {
        int col = col_ind[i];
        if (col > row) {
            sum += val[i] * shared_x[col % blockDim.x];
        } else if (col == row) {
            diag_val = val[i];
        }
    }

    // Ensure the diagonal value is non-zero to avoid division by zero
    if (diag_val != 0.0) {
        x[row] = (b[row] - sum) / diag_val;
        shared_x[threadIdx.x] = x[row];
    }
    
    __syncthreads();
}

__global__ void jacobi_lower_kernel(int n, const int *row_ptr, const int *col_ind, const double *val, const double *b, double *x, double *x_new) {
    __shared__ double shared_x[6144];

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    shared_x[threadIdx.x] = x[row];
    __syncthreads();

    double sum = 0.0;
    double diag_val = 0.0;
    int start = row_ptr[row];
    int end = row_ptr[row + 1];

    for (int i = start; i < end; i++) {
        int col = col_ind[i];
        if (col != row) {
            sum += val[i] * shared_x[col % blockDim.x];
        } else {
            diag_val = val[i];
        }
    }

    if (diag_val != 0.0) {
        x_new[row] = (b[row] - sum) / diag_val;
        shared_x[threadIdx.x] = x_new[row];
    }
    
    __syncthreads();
}

__global__ void jacobi_upper_kernel(int n, const int *row_ptr, const int *col_ind, const double *val, const double *b, double *x, double *x_new) {
    __shared__ double shared_x[6144];

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    shared_x[threadIdx.x] = x[row];
    __syncthreads();

    double sum = 0.0;
    double diag_val = 0.0;
    int start = row_ptr[row];
    int end = row_ptr[row + 1];

    for (int i = start; i < end; i++) {
        int col = col_ind[i];
        if (col != row) {
            sum += val[i] * shared_x[col % blockDim.x];
        } else {
            diag_val = val[i];
        }
    }

    if (diag_val != 0.0) {
        x_new[row] = (b[row] - sum) / diag_val;
        shared_x[threadIdx.x] = x_new[row];
    }
    
    __syncthreads();
}