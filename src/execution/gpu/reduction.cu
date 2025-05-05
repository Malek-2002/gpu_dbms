#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <iostream>

// overload the kernel to handle long data type
__global__ void reduce_sum_kernel(const long* input, long* output, int N) {
    extern __shared__ long sdata_long[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory
    sdata_long[tid] = (i < N) ? input[i] : 0.0;
    __syncthreads();

    // In-place reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata_long[tid] += sdata_long[tid + s];
        __syncthreads();
    }

    // Write result of this block to output
    if (tid == 0)
        output[blockIdx.x] = sdata_long[0];
}

// overload the kernel to handle double data type
__global__ void reduce_sum_kernel(const double* input, double* output, int N) {
    extern __shared__ double sdata_double[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory
    sdata_double[tid] = (i < N) ? input[i] : 0.0;
    __syncthreads();

    // In-place reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata_double[tid] += sdata_double[tid + s];
        __syncthreads();
    }

    // Write result of this block to output
    if (tid == 0)
        output[blockIdx.x] = sdata_double[0];
}

// Wrapper for reduction (long)
extern long parallel_sum(const std::vector<long>& data) {
    int N = data.size();
    if (N == 0) return 0.0;

    long *d_input, *d_output;
    int blockSize = 256;
    int current_size = N;
    long result = 0.0;

    // Allocate input array on device
    cudaMalloc(&d_input, N * sizeof(long));
    cudaMemcpy(d_input, data.data(), N * sizeof(long), cudaMemcpyHostToDevice);

    // We will reduce until one value remains
    while (current_size > 1) {
        int gridSize = (current_size + blockSize - 1) / blockSize;
        cudaMalloc(&d_output, gridSize * sizeof(long));

        // Call reduction kernel
        reduce_sum_kernel<<<gridSize, blockSize, blockSize * sizeof(long)>>>(d_input, d_output, current_size);

        // Cleanup old input
        cudaFree(d_input);

        // Prepare for next iteration
        d_input = d_output;
        current_size = gridSize;
    }

    // Copy final result back
    cudaMemcpy(&result, d_input, sizeof(long), cudaMemcpyDeviceToHost);
    cudaFree(d_input);

    return result;
}

// Wrapper for reduction (double)
extern double parallel_sum(const std::vector<double>& data) {
    int N = data.size();
    if (N == 0) return 0.0;

    double *d_input, *d_output;
    int blockSize = 256;
    int current_size = N;
    double result = 0.0;

    // Allocate input array on device
    cudaMalloc(&d_input, N * sizeof(double));
    cudaMemcpy(d_input, data.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // We will reduce until one value remains
    while (current_size > 1) {
        int gridSize = (current_size + blockSize - 1) / blockSize;
        cudaMalloc(&d_output, gridSize * sizeof(double));

        // Call reduction kernel
        reduce_sum_kernel<<<gridSize, blockSize, blockSize * sizeof(double)>>>(d_input, d_output, current_size);

        // Cleanup old input
        cudaFree(d_input);

        // Prepare for next iteration
        d_input = d_output;
        current_size = gridSize;
    }

    // Copy final result back
    cudaMemcpy(&result, d_input, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_input);

    return result;
}
