#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <cfloat>
#include <iostream>

// overload the kernel to handle int data type
__global__ void reduce_sum_kernel(const int* input, int* output, int N) {
    extern __shared__ int sdata_int[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory
    sdata_int[tid] = (i < N) ? input[i] : 0.0;
    __syncthreads();

    // In-place reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata_int[tid] += sdata_int[tid + s];
        __syncthreads();
    }

    // Write result of this block to output
    if (tid == 0)
        output[blockIdx.x] = sdata_int[0];
}


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

// min kernel for int data type
__global__ void reduce_min_kernel(const int* input, int* output, int N) {
    extern __shared__ int sdata_int[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory
    sdata_int[tid] = (i < N) ? input[i] : INT_MAX;
    __syncthreads();

    // In-place reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata_int[tid] = min(sdata_int[tid], sdata_int[tid + s]);
        __syncthreads();
    }

    // Write result of this block to output
    if (tid == 0)
        output[blockIdx.x] = sdata_int[0];
}

// min kernel for long data type
__global__ void reduce_min_kernel(const long* input, long* output, int N) {
    extern __shared__ long sdata_long[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory
    sdata_long[tid] = (i < N) ? input[i] : LONG_MAX;
    __syncthreads();

    // In-place reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata_long[tid] = min(sdata_long[tid], sdata_long[tid + s]);
        __syncthreads();
    }

    // Write result of this block to output
    if (tid == 0)
        output[blockIdx.x] = sdata_long[0];
}

// min kernel for double data type
__global__ void reduce_min_kernel(const double* input, double* output, int N) {
    extern __shared__ double sdata_double[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory
    sdata_double[tid] = (i < N) ? input[i] : DBL_MAX;
    __syncthreads();

    // In-place reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata_double[tid] = min(sdata_double[tid], sdata_double[tid + s]);
        __syncthreads();
    }

    // Write result of this block to output
    if (tid == 0)
        output[blockIdx.x] = sdata_double[0];
}

// device function to compare two strings
__device__ int string_less(const char* strings, int* offsets, int i, int j) {
    const char* a = strings + offsets[i];
    const char* b = strings + offsets[j];
    while (*a && *b) {
        if (*a < *b) return 1;
        if (*a > *b) return 0;
        a++; b++;
    }
    return *a == '\0' && *b != '\0';  // shorter string is "less"
}

// device function to compare two strings
__device__ int string_greater(const char* strings, int* offsets, int i, int j) {
    const char* a = strings + offsets[i];
    const char* b = strings + offsets[j];
    while (*a && *b) {
        if (*a > *b) return 1;
        if (*a < *b) return 0;
        a++; b++;
    }
    return *a != '\0' && *b == '\0';  // longer string is "greater"
}


// min kernel for string data type
__global__ void reduce_min_index_kernel(const char* strings, int* offsets, int* indices, int* output, int N) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load indices into shared memory
    sdata[tid] = (i < N) ? indices[i] : -1;
    __syncthreads();

    // In-place reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] != -1) {
            int a = sdata[tid];
            int b = sdata[tid + s];
            if (!string_less(strings, offsets, a, b)) {
                sdata[tid] = b;
            }
        }
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

// max kernel for int data type
__global__ void reduce_max_kernel(const int* input, int* output, int N) {
    extern __shared__ int sdata_int[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory
    sdata_int[tid] = (i < N) ? input[i] : INT_MIN;
    __syncthreads();

    // In-place reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata_int[tid] = max(sdata_int[tid], sdata_int[tid + s]);
        __syncthreads();
    }

    // Write result of this block to output
    if (tid == 0)
        output[blockIdx.x] = sdata_int[0];
}

// max kernel for long data type
__global__ void reduce_max_kernel(const long* input, long* output, int N) {
    extern __shared__ long sdata_long[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory
    sdata_long[tid] = (i < N) ? input[i] : LONG_MIN;
    __syncthreads();

    // In-place reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata_long[tid] = max(sdata_long[tid], sdata_long[tid + s]);
        __syncthreads();
    }

    // Write result of this block to output
    if (tid == 0)
        output[blockIdx.x] = sdata_long[0];
}

// max kernel for double data type
__global__ void reduce_max_kernel(const double* input, double* output, int N) {
    extern __shared__ double sdata_double[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory
    sdata_double[tid] = (i < N) ? input[i] : -DBL_MAX;
    __syncthreads();

    // In-place reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata_double[tid] = max(sdata_double[tid], sdata_double[tid + s]);
        __syncthreads();
    }

    // Write result of this block to output
    if (tid == 0)
        output[blockIdx.x] = sdata_double[0];
}

// min kernel for string data type
__global__ void reduce_max_index_kernel(const char* strings, int* offsets, int* indices, int* output, int N) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load indices into shared memory
    sdata[tid] = (i < N) ? indices[i] : -1;
    __syncthreads();

    // In-place reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] != -1) {
            int a = sdata[tid];
            int b = sdata[tid + s];
            if (!string_greater(strings, offsets, a, b)) {
                sdata[tid] = b;
            }
        }
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
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

// Wrapper for reduction (int)
extern int parallel_sum(const std::vector<int>& data) {
    int N = data.size();
    if (N == 0) return 0.0;

    int *d_input, *d_output;
    int blockSize = 256;
    int current_size = N;
    int result = 0.0;

    // Allocate input array on device
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMemcpy(d_input, data.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // We will reduce until one value remains
    while (current_size > 1) {
        int gridSize = (current_size + blockSize - 1) / blockSize;
        cudaMalloc(&d_output, gridSize * sizeof(int));

        // Call reduction kernel
        reduce_sum_kernel<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, current_size);

        // Cleanup old input
        cudaFree(d_input);

        // Prepare for next iteration
        d_input = d_output;
        current_size = gridSize;
    }

    // Copy final result back
    cudaMemcpy(&result, d_input, sizeof(int), cudaMemcpyDeviceToHost);
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

// wrapper for min (int)
extern int parallel_min(const std::vector<int>& data) {
    int N = data.size();
    if (N == 0) return INT_MAX;

    int *d_input, *d_output;
    int blockSize = 256;
    int current_size = N;
    int result = INT_MAX;

    cudaMalloc(&d_input, N * sizeof(int));
    cudaMemcpy(d_input, data.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    while (current_size > 1) {
        int gridSize = (current_size + blockSize - 1) / blockSize;
        cudaMalloc(&d_output, gridSize * sizeof(int));

        reduce_min_kernel<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, current_size);

        cudaFree(d_input);

        d_input = d_output;
        current_size = gridSize;
    }

    cudaMemcpy(&result, d_input, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_input);

    return result;
}

// wrapper for min (long)
extern long parallel_min(const std::vector<long>& data) {
    int N = data.size();
    if (N == 0) return LONG_MAX;

    long *d_input, *d_output;
    int blockSize = 256;
    int current_size = N;
    long result = LONG_MAX;

    cudaMalloc(&d_input, N * sizeof(long));
    cudaMemcpy(d_input, data.data(), N * sizeof(long), cudaMemcpyHostToDevice);

    while (current_size > 1) {
        int gridSize = (current_size + blockSize - 1) / blockSize;
        cudaMalloc(&d_output, gridSize * sizeof(long));

        reduce_min_kernel<<<gridSize, blockSize, blockSize * sizeof(long)>>>(d_input, d_output, current_size);

        cudaFree(d_input);

        d_input = d_output;
        current_size = gridSize;
    }

    cudaMemcpy(&result, d_input, sizeof(long), cudaMemcpyDeviceToHost);
    cudaFree(d_input);

    return result;
}

// wrapper for min (double)
extern double parallel_min(const std::vector<double>& data) {
    int N = data.size();
    if (N == 0) return LONG_MAX;

    double *d_input, *d_output;
    int blockSize = 256;
    int current_size = N;
    double result = DBL_MAX;

    cudaMalloc(&d_input, N * sizeof(double));
    cudaMemcpy(d_input, data.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    while (current_size > 1) {
        int gridSize = (current_size + blockSize - 1) / blockSize;
        cudaMalloc(&d_output, gridSize * sizeof(double));

        reduce_min_kernel<<<gridSize, blockSize, blockSize * sizeof(double)>>>(d_input, d_output, current_size);

        cudaFree(d_input);

        d_input = d_output;
        current_size = gridSize;
    }

    cudaMemcpy(&result, d_input, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_input);

    return result;
}

// wrapper for min (string)
extern std::string parallel_min(const std::vector<std::string>& data) {
    if (data.empty()) return "";

    int N = data.size();
    int blockSize = 256;

    // 1. Flatten strings and build offsets
    std::vector<char> flat_chars;
    std::vector<int> offsets(N);
    int offset = 0;
    for (int i = 0; i < N; ++i) {
        offsets[i] = offset;
        flat_chars.insert(flat_chars.end(), data[i].begin(), data[i].end());
        flat_chars.push_back('\0');  // Null-terminate for safety
        offset += data[i].size() + 1;
    }

    // 2. Prepare indices
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0); // 0..N-1

    // 3. Allocate device memory
    char* d_strings = nullptr;
    int* d_offsets = nullptr;
    int* d_indices = nullptr;
    int* d_output = nullptr;

    int total_chars = flat_chars.size();
    cudaMalloc(&d_strings, total_chars * sizeof(char));
    cudaMalloc(&d_offsets, N * sizeof(int));
    cudaMalloc(&d_indices, N * sizeof(int));
    cudaMemcpy(d_strings, flat_chars.data(), total_chars * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, indices.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    int current_size = N;
    while (current_size > 1) {
        int gridSize = (current_size + blockSize - 1) / blockSize;
        cudaMalloc(&d_output, gridSize * sizeof(int));

        // Launch kernel
        reduce_min_index_kernel<<<gridSize, blockSize, blockSize * sizeof(int)>>>(
            d_strings, d_offsets, d_indices, d_output, current_size
        );

        cudaFree(d_indices); // Free previous input indices
        d_indices = d_output;
        current_size = gridSize;
    }

    // 4. Copy back result
    int min_index = -1;
    cudaMemcpy(&min_index, d_indices, sizeof(int), cudaMemcpyDeviceToHost);

    // 5. Cleanup
    cudaFree(d_strings);
    cudaFree(d_offsets);
    cudaFree(d_indices);  // d_output == d_indices now

    // 6. Return minimum string
    if (min_index < 0 || min_index >= N) {
        throw std::runtime_error("Invalid min index from GPU");
    }

    return data[min_index];
}

// wrapper for max (int)
extern int parallel_max(const std::vector<int>& data) {
    int N = data.size();
    if (N == 0) return INT_MIN;

    int *d_input, *d_output;
    int blockSize = 256;
    int current_size = N;
    int result = INT_MIN;

    cudaMalloc(&d_input, N * sizeof(int));
    cudaMemcpy(d_input, data.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    while (current_size > 1) {
        int gridSize = (current_size + blockSize - 1) / blockSize;
        cudaMalloc(&d_output, gridSize * sizeof(int));

        reduce_max_kernel<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, current_size);

        cudaFree(d_input);

        d_input = d_output;
        current_size = gridSize;
    }

    cudaMemcpy(&result, d_input, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_input);

    return result;
}

// wrapper for max (long)
extern long parallel_max(const std::vector<long>& data) {
    int N = data.size();
    if (N == 0) return LONG_MIN;

    long *d_input, *d_output;
    int blockSize = 256;
    int current_size = N;
    long result = LONG_MIN;

    cudaMalloc(&d_input, N * sizeof(long));
    cudaMemcpy(d_input, data.data(), N * sizeof(long), cudaMemcpyHostToDevice);

    while (current_size > 1) {
        int gridSize = (current_size + blockSize - 1) / blockSize;
        cudaMalloc(&d_output, gridSize * sizeof(long));

        reduce_max_kernel<<<gridSize, blockSize, blockSize * sizeof(long)>>>(d_input, d_output, current_size);

        cudaFree(d_input);

        d_input = d_output;
        current_size = gridSize;
    }

    cudaMemcpy(&result, d_input, sizeof(long), cudaMemcpyDeviceToHost);
    cudaFree(d_input);

    return result;
}

// wrapper for max (double)
extern double parallel_max(const std::vector<double>& data) {
    int N = data.size();
    if (N == 0) return LONG_MIN;

    double *d_input, *d_output;
    int blockSize = 256;
    int current_size = N;
    double result = -DBL_MAX;

    cudaMalloc(&d_input, N * sizeof(double));
    cudaMemcpy(d_input, data.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    while (current_size > 1) {
        int gridSize = (current_size + blockSize - 1) / blockSize;
        cudaMalloc(&d_output, gridSize * sizeof(double));

        reduce_max_kernel<<<gridSize, blockSize, blockSize * sizeof(double)>>>(d_input, d_output, current_size);

        cudaFree(d_input);

        d_input = d_output;
        current_size = gridSize;
    }

    cudaMemcpy(&result, d_input, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_input);

    return result;
}

// wrapper for max (string)
extern std::string parallel_max(const std::vector<std::string>& data) {
    if (data.empty()) return "";

    int N = data.size();
    int blockSize = 256;

    // 1. Flatten strings and build offsets
    std::vector<char> flat_chars;
    std::vector<int> offsets(N);
    int offset = 0;
    for (int i = 0; i < N; ++i) {
        offsets[i] = offset;
        flat_chars.insert(flat_chars.end(), data[i].begin(), data[i].end());
        flat_chars.push_back('\0');  // Null-terminate for safety
        offset += data[i].size() + 1;
    }

    // 2. Prepare indices
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0); // 0..N-1

    // 3. Allocate device memory
    char* d_strings = nullptr;
    int* d_offsets = nullptr;
    int* d_indices = nullptr;
    int* d_output = nullptr;

    int total_chars = flat_chars.size();
    cudaMalloc(&d_strings, total_chars * sizeof(char));
    cudaMalloc(&d_offsets, N * sizeof(int));
    cudaMalloc(&d_indices, N * sizeof(int));
    cudaMemcpy(d_strings, flat_chars.data(), total_chars * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, indices.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    int current_size = N;
    while (current_size > 1) {
        int gridSize = (current_size + blockSize - 1) / blockSize;
        cudaMalloc(&d_output, gridSize * sizeof(int));

        // Launch kernel
        reduce_max_index_kernel<<<gridSize, blockSize, blockSize * sizeof(int)>>>(
            d_strings, d_offsets, d_indices, d_output, current_size
        );

        cudaFree(d_indices); // Free previous input indices
        d_indices = d_output;
        current_size = gridSize;
    }

    // 4. Copy back result
    int max_index = -1;
    cudaMemcpy(&max_index, d_indices, sizeof(int), cudaMemcpyDeviceToHost);

    // 5. Cleanup
    cudaFree(d_strings);
    cudaFree(d_offsets);
    cudaFree(d_indices);  // d_output == d_indices now

    // 6. Return minimum string
    if (max_index < 0 || max_index >= N) {
        throw std::runtime_error("Invalid max index from GPU");
    }

    return data[max_index];
}
