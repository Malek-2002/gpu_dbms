#include <cuda_runtime.h>
#include <vector>

__global__ void copy_int_column(const long* src, long* dst, int num_rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_rows) {
        dst[idx] = src[idx];
    }
}

__global__ void copy_double_column(const double* src, double* dst, int num_rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_rows) {
        dst[idx] = src[idx];
    }
}

extern std::vector<long> copy_column(const std::vector<long>& input_data) {
    long* d_input;
    long* d_output;
    size_t size = input_data.size() * sizeof(long);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, input_data.data(), size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (input_data.size() + threads - 1) / threads;
    copy_int_column<<<blocks, threads>>>(d_input, d_output, input_data.size());

    std::vector<long> output_data(input_data.size());

    cudaMemcpy(output_data.data(), d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output_data;
}

extern std::vector<double> copy_column(const std::vector<double>& input_data) {
    double* d_input;
    double* d_output;
    size_t size = input_data.size() * sizeof(double);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, input_data.data(), size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (input_data.size() + threads - 1) / threads;
    copy_double_column<<<blocks, threads>>>(d_input, d_output, input_data.size());

    std::vector<double> output_data(input_data.size());

    cudaMemcpy(output_data.data(), d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output_data;
}