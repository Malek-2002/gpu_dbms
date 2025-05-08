#include <cuda_runtime.h>
#include <vector>

__global__ void copy_int_column(const long* src, long* dst, int num_rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_rows) {
        dst[idx] = src[idx];
    }
}

__global__ void copy_int_column_sorted(const long* src, const size_t* order, long* dst, int num_rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_rows) {
        dst[idx] = src[order[idx]];
    }
}

__global__ void copy_double_column(const double* src, double* dst, int num_rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_rows) {
        dst[idx] = src[idx];
    }
}

__global__ void copy_double_column_sorted(const double* src, const size_t* order, double* dst, int num_rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_rows) {
        dst[idx] = src[order[idx]];
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

extern std::vector<long> copy_column_sorted(const std::vector<long>& input_data, const std::vector<size_t>& order) {
    long* d_input;
    size_t* d_order;
    long* d_output;
    size_t size = input_data.size() * sizeof(long);
    size_t order_size = order.size() * sizeof(size_t);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_order, order_size);

    cudaMemcpy(d_input, input_data.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_order, order.data(), order_size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (input_data.size() + threads - 1) / threads;
    copy_int_column_sorted<<<blocks, threads>>>(d_input, d_order, d_output, input_data.size());

    std::vector<long> output_data(input_data.size());

    cudaMemcpy(output_data.data(), d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_order);
    cudaFree(d_output);

    return output_data;
}

extern std::vector<double> copy_column_sorted(const std::vector<double>& input_data, const std::vector<size_t>& order) {
    double* d_input;
    size_t* d_order;
    double* d_output;
    size_t size = input_data.size() * sizeof(double);
    size_t order_size = order.size() * sizeof(size_t);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_order, order_size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, input_data.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_order, order.data(), order_size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (input_data.size() + threads - 1) / threads;
    copy_double_column_sorted<<<blocks, threads>>>(d_input, d_order, d_output, input_data.size());

    std::vector<double> output_data(input_data.size());

    cudaMemcpy(output_data.data(), d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_order);
    cudaFree(d_output);

    return output_data;
}
