
#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <cfloat>
#include <iostream>
#include <algorithm>

#define THREADS_PER_BLOCK 256

__global__ void reduce_sum_kernel_streams(const int* input, int* output, int N) {
    extern __shared__ int sdata_sum_int[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata_sum_int[tid] = (i < N) ? input[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata_sum_int[tid] += sdata_sum_int[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata_sum_int[0];
}

extern int parallel_reduce_sum(const std::vector<int>& host_data) {
    int N = host_data.size();
    if (N == 0) return 0;

    int chunk_size = 1 << 20;  // 1M elements per chunk
    int num_chunks = (N + chunk_size - 1) / chunk_size;
    std::vector<int> partial_results(num_chunks);

    int* d_input = nullptr;
    int* d_partial = nullptr;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc(&d_input, chunk_size * sizeof(int));
    cudaMalloc(&d_partial, ((chunk_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) * sizeof(int));

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        int offset = chunk * chunk_size;
        int current_chunk_size = std::min(chunk_size, N - offset);
        int blocks = (current_chunk_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        cudaMemcpyAsync(d_input, host_data.data() + offset, current_chunk_size * sizeof(int), cudaMemcpyHostToDevice, stream);
        reduce_sum_kernel_streams<<<blocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int), stream>>>(d_input, d_partial, current_chunk_size);

        // Second pass reduction
        int* d_final = nullptr;
        cudaMalloc(&d_final, sizeof(int));
        reduce_sum_kernel_streams<<<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int), stream>>>(d_partial, d_final, blocks);
        cudaMemcpyAsync(&partial_results[chunk], d_final, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaFree(d_final);
        cudaStreamSynchronize(stream);
    }

    cudaFree(d_input);
    cudaFree(d_partial);
    cudaStreamDestroy(stream);

    return std::accumulate(partial_results.begin(), partial_results.end(), static_cast<int>(0));
}

__global__ void reduce_sum_kernel_streams(const long* input, long* output, int N) {
    extern __shared__ long sdata_sum_long[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata_sum_long[tid] = (i < N) ? input[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata_sum_long[tid] += sdata_sum_long[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata_sum_long[0];
}

extern long parallel_reduce_sum(const std::vector<long>& host_data) {
    int N = host_data.size();
    if (N == 0) return 0;

    int chunk_size = 1 << 20;  // 1M elements per chunk
    int num_chunks = (N + chunk_size - 1) / chunk_size;
    std::vector<long> partial_results(num_chunks);

    long* d_input = nullptr;
    long* d_partial = nullptr;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc(&d_input, chunk_size * sizeof(long));
    cudaMalloc(&d_partial, ((chunk_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) * sizeof(long));

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        int offset = chunk * chunk_size;
        int current_chunk_size = std::min(chunk_size, N - offset);
        int blocks = (current_chunk_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        cudaMemcpyAsync(d_input, host_data.data() + offset, current_chunk_size * sizeof(long), cudaMemcpyHostToDevice, stream);
        reduce_sum_kernel_streams<<<blocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(long), stream>>>(d_input, d_partial, current_chunk_size);

        // Second pass reduction
        long* d_final = nullptr;
        cudaMalloc(&d_final, sizeof(long));
        reduce_sum_kernel_streams<<<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(long), stream>>>(d_partial, d_final, blocks);
        cudaMemcpyAsync(&partial_results[chunk], d_final, sizeof(long), cudaMemcpyDeviceToHost, stream);
        cudaFree(d_final);
        cudaStreamSynchronize(stream);
    }

    cudaFree(d_input);
    cudaFree(d_partial);
    cudaStreamDestroy(stream);

    return std::accumulate(partial_results.begin(), partial_results.end(), static_cast<long>(0));
}

__global__ void reduce_sum_kernel_streams(const double* input, double* output, int N) {
    extern __shared__ double sdata_sum_double[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata_sum_double[tid] = (i < N) ? input[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata_sum_double[tid] += sdata_sum_double[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata_sum_double[0];
}

extern double parallel_reduce_sum(const std::vector<double>& host_data) {
    int N = host_data.size();
    if (N == 0) return 0;

    int chunk_size = 1 << 20;  // 1M elements per chunk
    int num_chunks = (N + chunk_size - 1) / chunk_size;
    std::vector<double> partial_results(num_chunks);

    double* d_input = nullptr;
    double* d_partial = nullptr;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc(&d_input, chunk_size * sizeof(double));
    cudaMalloc(&d_partial, ((chunk_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) * sizeof(double));

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        int offset = chunk * chunk_size;
        int current_chunk_size = std::min(chunk_size, N - offset);
        int blocks = (current_chunk_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        cudaMemcpyAsync(d_input, host_data.data() + offset, current_chunk_size * sizeof(double), cudaMemcpyHostToDevice, stream);
        reduce_sum_kernel_streams<<<blocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double), stream>>>(d_input, d_partial, current_chunk_size);

        // Second pass reduction
        double* d_final = nullptr;
        cudaMalloc(&d_final, sizeof(double));
        reduce_sum_kernel_streams<<<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double), stream>>>(d_partial, d_final, blocks);
        cudaMemcpyAsync(&partial_results[chunk], d_final, sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaFree(d_final);
        cudaStreamSynchronize(stream);
    }

    cudaFree(d_input);
    cudaFree(d_partial);
    cudaStreamDestroy(stream);

    return std::accumulate(partial_results.begin(), partial_results.end(), static_cast<double>(0));
}

__global__ void reduce_min_kernel_streams(const int* input, int* output, int N) {
    extern __shared__ int sdata_min_int[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata_min_int[tid] = (i < N) ? input[i] : DBL_MAX;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata_min_int[tid] = min(sdata_min_int[tid], sdata_min_int[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata_min_int[0];
}

extern int parallel_reduce_min(const std::vector<int>& host_data) {
    int N = host_data.size();
    if (N == 0) return 0;

    int chunk_size = 1 << 20;
    int num_chunks = (N + chunk_size - 1) / chunk_size;
    std::vector<int> partial_results(num_chunks);

    int* d_input = nullptr;
    int* d_partial = nullptr;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc(&d_input, chunk_size * sizeof(int));
    cudaMalloc(&d_partial, ((chunk_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) * sizeof(int));

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        int offset = chunk * chunk_size;
        int current_chunk_size = std::min(chunk_size, N - offset);
        int blocks = (current_chunk_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        cudaMemcpyAsync(d_input, host_data.data() + offset, current_chunk_size * sizeof(int), cudaMemcpyHostToDevice, stream);
        reduce_min_kernel_streams<<<blocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int), stream>>>(d_input, d_partial, current_chunk_size);

        int* d_final = nullptr;
        cudaMalloc(&d_final, sizeof(int));
        reduce_min_kernel_streams<<<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int), stream>>>(d_partial, d_final, blocks);
        cudaMemcpyAsync(&partial_results[chunk], d_final, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaFree(d_final);
        cudaStreamSynchronize(stream);
    }

    cudaFree(d_input);
    cudaFree(d_partial);
    cudaStreamDestroy(stream);

    return *std::min_element(partial_results.begin(), partial_results.end());
}

__global__ void reduce_min_kernel_streams(const long* input, long* output, int N) {
    extern __shared__ long sdata_min_long[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata_min_long[tid] = (i < N) ? input[i] : DBL_MAX;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata_min_long[tid] = min(sdata_min_long[tid], sdata_min_long[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata_min_long[0];
}

extern long parallel_reduce_min(const std::vector<long>& host_data) {
    int N = host_data.size();
    if (N == 0) return 0;

    int chunk_size = 1 << 20;
    int num_chunks = (N + chunk_size - 1) / chunk_size;
    std::vector<long> partial_results(num_chunks);

    long* d_input = nullptr;
    long* d_partial = nullptr;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc(&d_input, chunk_size * sizeof(long));
    cudaMalloc(&d_partial, ((chunk_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) * sizeof(long));

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        int offset = chunk * chunk_size;
        int current_chunk_size = std::min(chunk_size, N - offset);
        int blocks = (current_chunk_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        cudaMemcpyAsync(d_input, host_data.data() + offset, current_chunk_size * sizeof(long), cudaMemcpyHostToDevice, stream);
        reduce_min_kernel_streams<<<blocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(long), stream>>>(d_input, d_partial, current_chunk_size);

        long* d_final = nullptr;
        cudaMalloc(&d_final, sizeof(long));
        reduce_min_kernel_streams<<<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(long), stream>>>(d_partial, d_final, blocks);
        cudaMemcpyAsync(&partial_results[chunk], d_final, sizeof(long), cudaMemcpyDeviceToHost, stream);
        cudaFree(d_final);
        cudaStreamSynchronize(stream);
    }

    cudaFree(d_input);
    cudaFree(d_partial);
    cudaStreamDestroy(stream);

    return *std::min_element(partial_results.begin(), partial_results.end());
}

__global__ void reduce_min_kernel_streams(const double* input, double* output, int N) {
    extern __shared__ double sdata_min_double[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata_min_double[tid] = (i < N) ? input[i] : DBL_MAX;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata_min_double[tid] = min(sdata_min_double[tid], sdata_min_double[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata_min_double[0];
}

extern double parallel_reduce_min(const std::vector<double>& host_data) {
    int N = host_data.size();
    if (N == 0) return 0;

    int chunk_size = 1 << 20;
    int num_chunks = (N + chunk_size - 1) / chunk_size;
    std::vector<double> partial_results(num_chunks);

    double* d_input = nullptr;
    double* d_partial = nullptr;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc(&d_input, chunk_size * sizeof(double));
    cudaMalloc(&d_partial, ((chunk_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) * sizeof(double));

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        int offset = chunk * chunk_size;
        int current_chunk_size = std::min(chunk_size, N - offset);
        int blocks = (current_chunk_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        cudaMemcpyAsync(d_input, host_data.data() + offset, current_chunk_size * sizeof(double), cudaMemcpyHostToDevice, stream);
        reduce_min_kernel_streams<<<blocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double), stream>>>(d_input, d_partial, current_chunk_size);

        double* d_final = nullptr;
        cudaMalloc(&d_final, sizeof(double));
        reduce_min_kernel_streams<<<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double), stream>>>(d_partial, d_final, blocks);
        cudaMemcpyAsync(&partial_results[chunk], d_final, sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaFree(d_final);
        cudaStreamSynchronize(stream);
    }

    cudaFree(d_input);
    cudaFree(d_partial);
    cudaStreamDestroy(stream);

    return *std::min_element(partial_results.begin(), partial_results.end());
}

__global__ void reduce_max_kernel_streams(const int* input, int* output, int N) {
    extern __shared__ int sdata_max_int[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata_max_int[tid] = (i < N) ? input[i] : INT_MIN;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata_max_int[tid] = max(sdata_max_int[tid], sdata_max_int[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata_max_int[0];
}

extern int parallel_reduce_max(const std::vector<int>& host_data) {
    int N = host_data.size();
    if (N == 0) return 0;

    int chunk_size = 1 << 20;
    int num_chunks = (N + chunk_size - 1) / chunk_size;
    std::vector<int> partial_results(num_chunks);

    int* d_input = nullptr;
    int* d_partial = nullptr;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc(&d_input, chunk_size * sizeof(int));
    cudaMalloc(&d_partial, ((chunk_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) * sizeof(int));

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        int offset = chunk * chunk_size;
        int current_chunk_size = std::min(chunk_size, N - offset);
        int blocks = (current_chunk_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        cudaMemcpyAsync(d_input, host_data.data() + offset, current_chunk_size * sizeof(int), cudaMemcpyHostToDevice, stream);
        reduce_max_kernel_streams<<<blocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int), stream>>>(d_input, d_partial, current_chunk_size);

        int* d_final = nullptr;
        cudaMalloc(&d_final, sizeof(int));
        reduce_max_kernel_streams<<<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int), stream>>>(d_partial, d_final, blocks);
        cudaMemcpyAsync(&partial_results[chunk], d_final, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaFree(d_final);
        cudaStreamSynchronize(stream);
    }

    cudaFree(d_input);
    cudaFree(d_partial);
    cudaStreamDestroy(stream);

    return *std::max_element(partial_results.begin(), partial_results.end());
}

__global__ void reduce_max_kernel_streams(const long* input, long* output, int N) {
    extern __shared__ long sdata_max_long[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata_max_long[tid] = (i < N) ? input[i] : LONG_MIN;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata_max_long[tid] = max(sdata_max_long[tid], sdata_max_long[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata_max_long[0];
}

extern long parallel_reduce_max(const std::vector<long>& host_data) {
    int N = host_data.size();
    if (N == 0) return 0;

    int chunk_size = 1 << 20;
    int num_chunks = (N + chunk_size - 1) / chunk_size;
    std::vector<long> partial_results(num_chunks);

    long* d_input = nullptr;
    long* d_partial = nullptr;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc(&d_input, chunk_size * sizeof(long));
    cudaMalloc(&d_partial, ((chunk_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) * sizeof(long));

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        int offset = chunk * chunk_size;
        int current_chunk_size = std::min(chunk_size, N - offset);
        int blocks = (current_chunk_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        cudaMemcpyAsync(d_input, host_data.data() + offset, current_chunk_size * sizeof(long), cudaMemcpyHostToDevice, stream);
        reduce_max_kernel_streams<<<blocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(long), stream>>>(d_input, d_partial, current_chunk_size);

        long* d_final = nullptr;
        cudaMalloc(&d_final, sizeof(long));
        reduce_max_kernel_streams<<<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(long), stream>>>(d_partial, d_final, blocks);
        cudaMemcpyAsync(&partial_results[chunk], d_final, sizeof(long), cudaMemcpyDeviceToHost, stream);
        cudaFree(d_final);
        cudaStreamSynchronize(stream);
    }

    cudaFree(d_input);
    cudaFree(d_partial);
    cudaStreamDestroy(stream);

    return *std::max_element(partial_results.begin(), partial_results.end());
}

__global__ void reduce_max_kernel_streams(const double* input, double* output, int N) {
    extern __shared__ double sdata_max_double[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata_max_double[tid] = (i < N) ? input[i] : -DBL_MAX;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata_max_double[tid] = max(sdata_max_double[tid], sdata_max_double[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata_max_double[0];
}

extern double parallel_reduce_max(const std::vector<double>& host_data) {
    int N = host_data.size();
    if (N == 0) return 0;

    int chunk_size = 1 << 20;
    int num_chunks = (N + chunk_size - 1) / chunk_size;
    std::vector<double> partial_results(num_chunks);

    double* d_input = nullptr;
    double* d_partial = nullptr;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc(&d_input, chunk_size * sizeof(double));
    cudaMalloc(&d_partial, ((chunk_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) * sizeof(double));

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        int offset = chunk * chunk_size;
        int current_chunk_size = std::min(chunk_size, N - offset);
        int blocks = (current_chunk_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        cudaMemcpyAsync(d_input, host_data.data() + offset, current_chunk_size * sizeof(double), cudaMemcpyHostToDevice, stream);
        reduce_max_kernel_streams<<<blocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double), stream>>>(d_input, d_partial, current_chunk_size);

        double* d_final = nullptr;
        cudaMalloc(&d_final, sizeof(double));
        reduce_max_kernel_streams<<<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double), stream>>>(d_partial, d_final, blocks);
        cudaMemcpyAsync(&partial_results[chunk], d_final, sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaFree(d_final);
        cudaStreamSynchronize(stream);
    }

    cudaFree(d_input);
    cudaFree(d_partial);
    cudaStreamDestroy(stream);

    return *std::max_element(partial_results.begin(), partial_results.end());
}
