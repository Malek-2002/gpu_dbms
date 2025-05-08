#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <cfloat>
#include <iostream>

__global__ void join_and_or_kernel(
    const int* left_data, int n,
    const int* right_data, int m,
    int* result,
    int op_code
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < m) {
        int l = left_data[i * m + j];
        int r = right_data[i * m + j];

        int res = false;
        switch (op_code) {
            case 0: res = (l && r); break; // AND
            case 1: res = (l || r); break; // OR
            // case 2: res = (l ^ r); break;  // XOR
        }

        result[i * m + j] = res;
    }
}

__global__ void join_compare_kernel(
    const int* left_data, int n,
    const int* right_data, int m,
    int* result,
    int op_code // 0: EQ, 1: NEQ, 2: LT, 3: GT, 4: LEQ, 5: GEQ
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < m) {
        int l = left_data[i];
        int r = right_data[j];

        int res = false;
        switch (op_code) {
            case 0: res = (l == r); break;
            case 1: res = (l != r); break;
            case 2: res = (l <  r); break;
            case 3: res = (l >  r); break;
            case 4: res = (l <= r); break;
            case 5: res = (l >= r); break;
        }

        result[i * m + j] = res;
    }
}

__global__ void join_compare_kernel(
    const long* left_data, int n,
    const long* right_data, int m,
    int* result,
    int op_code // 0: EQ, 1: NEQ, 2: LT, 3: GT, 4: LEQ, 5: GEQ
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < m) {
        long l = left_data[i];
        long r = right_data[j];

        int res = false;
        switch (op_code) {
            case 0: res = (l == r); break;
            case 1: res = (l != r); break;
            case 2: res = (l <  r); break;
            case 3: res = (l >  r); break;
            case 4: res = (l <= r); break;
            case 5: res = (l >= r); break;
        }

        result[i * m + j] = res;
    }
}

__global__ void join_compare_kernel(
    const double* left_data, int n,
    const double* right_data, int m,
    int* result,
    int op_code // 0: EQ, 1: NEQ, 2: LT, 3: GT, 4: LEQ, 5: GEQ
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < m) {
        double l = left_data[i];
        double r = right_data[j];

        int res = false;
        switch (op_code) {
            case 0: res = (l == r); break;
            case 1: res = (l != r); break;
            case 2: res = (l <  r); break;
            case 3: res = (l >  r); break;
            case 4: res = (l <= r); break;
            case 5: res = (l >= r); break;
        }

        result[i * m + j] = res;
    }
}

extern std::vector<int> launch_join_compare(
    const std::vector<int>& left,
    const std::vector<int>& right,
    size_t n, size_t m,
    int op_code)
{
    size_t result_size = n * m;

    int *d_left, *d_right;
    int *d_result;

    cudaMalloc(&d_left, n * sizeof(int));
    cudaMalloc(&d_right, m * sizeof(int));
    cudaMalloc(&d_result, result_size * sizeof(int));

    cudaMemcpy(d_left, left.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, right.data(), m * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((m + 15) / 16, (n + 15) / 16);

    join_compare_kernel<<<gridDim, blockDim>>>(d_left, n, d_right, m, d_result, op_code);

    std::vector<int> result(result_size);
    cudaMemcpy(result.data(), d_result, result_size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_result);

    return result;
}

extern std::vector<int> launch_join_compare(
    const std::vector<long>& left,
    const std::vector<long>& right,
    size_t n, size_t m,
    int op_code)
{
    size_t result_size = n * m;

    long *d_left, *d_right;
    int *d_result;

    cudaMalloc(&d_left, n * sizeof(long));
    cudaMalloc(&d_right, m * sizeof(long));
    cudaMalloc(&d_result, result_size * sizeof(int));

    cudaMemcpy(d_left, left.data(), n * sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, right.data(), m * sizeof(long), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((m + 15) / 16, (n + 15) / 16);

    join_compare_kernel<<<gridDim, blockDim>>>(d_left, n, d_right, m, d_result, op_code);

    std::vector<int> result(result_size);
    cudaMemcpy(result.data(), d_result, result_size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_result);

    return result;
}

extern std::vector<int> launch_join_compare(
    const std::vector<double>& left,
    const std::vector<double>& right,
    size_t n, size_t m,
    int op_code)
{
    size_t result_size = n * m;

    double *d_left, *d_right;
    int *d_result;

    cudaMalloc(&d_left, n * sizeof(double));
    cudaMalloc(&d_right, m * sizeof(double));
    cudaMalloc(&d_result, result_size * sizeof(int));

    cudaMemcpy(d_left, left.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, right.data(), m * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((m + 15) / 16, (n + 15) / 16);

    join_compare_kernel<<<gridDim, blockDim>>>(d_left, n, d_right, m, d_result, op_code);

    std::vector<int> result(result_size);
    cudaMemcpy(result.data(), d_result, result_size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_result);

    return result;
}

extern std::vector<int> launch_join_and_or(
    const std::vector<int>& left,
    const std::vector<int>& right,
    size_t n, size_t m,
    int op_code)
{
    size_t result_size = n * m;

    int *d_left, *d_right;
    int *d_result;

    cudaMalloc(&d_left, n * m * sizeof(int));
    cudaMalloc(&d_right, m * n *sizeof(int));
    cudaMalloc(&d_result, result_size * sizeof(int));

    cudaMemcpy(d_left, left.data(), n * m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, right.data(), m * n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((m + 15) / 16, (n + 15) / 16);

    join_and_or_kernel<<<gridDim, blockDim>>>(d_left, n, d_right, m, d_result, op_code);

    std::vector<int> result(result_size);
    cudaMemcpy(result.data(), d_result, result_size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_result);

    return result;
}
