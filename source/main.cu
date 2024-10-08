#include "constants.h"
#include <stdio.h>
#include <vector>
#include <string>

// kernel launch overheads

__global__ void emptyKernel()
{
}

void TIME_emptykernel_overhead()
{
    emptyKernel<<<1, 1>>>();

    TIMER_EVENTS_CREATE
    TIMER_EVENTS_RECORD_START
    
    for (int i = 0; i < kNumIterations; i++)
    {
        emptyKernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }

    TIMER_EVENTS_RECORD_STOP

    REPORT_TIME("launch an empty kernel")
}

// memory allocation overheads
void time_allocation_overhead(int num_bytes, std::string label, int numIterations = kNumIterations)
{
    std::vector<char*> d_dataVec(numIterations, nullptr);

    TIMER_EVENTS_CREATE
    TIMER_EVENTS_RECORD_START

    for (int i = 0; i < numIterations; i++)
        checkCudaErrors(cudaMalloc(&d_dataVec[i], num_bytes));

    TIMER_EVENTS_RECORD_STOP
    REPORT_TIME(label.c_str())
    
    for (int i = 0; i < numIterations; i++)
    checkCudaErrors(cudaFree(d_dataVec[i]));
}

#define TIME_ALLOCATION_OVERHEAD(SIZE, DESC, NUM_ITERATIONS) \
    void TIME_##SIZE##byte_allocation_overhead()  \
    {                                         \
        time_allocation_overhead(SIZE, DESC, NUM_ITERATIONS); \
    }

TIME_ALLOCATION_OVERHEAD(1, "allocate 1 byte", kNumIterations)
TIME_ALLOCATION_OVERHEAD(1024, "allocate 1 KB", kNumIterations)
TIME_ALLOCATION_OVERHEAD(kOneM, "allocate 1 MB", kNumIterations)
TIME_ALLOCATION_OVERHEAD(kHundredM, "allocate 100 MB", kNumIterations)
TIME_ALLOCATION_OVERHEAD(kOneG, "allocate 1 GB", 1)

// memory copy overheads

void time_copy_overhead(int num_bytes, std::string label, int numIterations = kNumIterations)
{
    char *h_data = new char[num_bytes], *d_data = nullptr;
    checkCudaErrors(cudaMalloc(&d_data, num_bytes));

    TIMER_EVENTS_CREATE
    TIMER_EVENTS_RECORD_START

    for (int i = 0; i < numIterations; i++)
    {
        checkCudaErrors(cudaMemcpy(d_data, h_data, num_bytes, cudaMemcpyHostToDevice));
    }

    TIMER_EVENTS_RECORD_STOP
    REPORT_TIME(label.c_str())

    delete[] h_data;
    checkCudaErrors(cudaFree(d_data));
}

#define TIME_COPY_OVERHEAD(SIZE, DESC, ITERATIONS) \
    void TIME_##SIZE##byte_copy_overhead()  \
    {                                         \
        time_copy_overhead(SIZE, DESC, ITERATIONS); \
    }

TIME_COPY_OVERHEAD(1, "copy 1 byte", kNumIterations)
TIME_COPY_OVERHEAD(1024, "copy 1 KB", kNumIterations)
TIME_COPY_OVERHEAD(kOneM, "copy 1 MB", kNumIterations)
TIME_COPY_OVERHEAD(kHundredM, "copy 100 MB", kNumIterations)
TIME_COPY_OVERHEAD(kOneG, "copy 1 GB", 1)

// min-max test
// TODO: Optimize to use hierarchical min-max reduction
__global__ void minMaxKernel(int *data, int n, int *min, int *max)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        atomicMin(min, data[i]);
        atomicMax(max, data[i]);
    }
}

void TIME_minmax_kernel(int n, std::string label)
{
    int *h_data = new int[n];
    for (int i = 0; i < n; i++)
        h_data[i] = rand() % 1000;

    int *d_data = nullptr, *d_min = nullptr, *d_max = nullptr;
    checkCudaErrors(cudaMalloc(&d_data, n * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_min, sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_max, sizeof(int)));

    checkCudaErrors(cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_min, 0x7f, sizeof(int)));
    checkCudaErrors(cudaMemset(d_max, 0x80, sizeof(int)));

    TIMER_EVENTS_CREATE
    TIMER_EVENTS_RECORD_START

    minMaxKernel<<<(n + 255) / 256, 256>>>(d_data, n, d_min, d_max);
    cudaDeviceSynchronize();

    TIMER_EVENTS_RECORD_STOP
    REPORT_TIME(label.c_str())

    delete[] h_data;
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFree(d_min));
    checkCudaErrors(cudaFree(d_max));
}

// main test runner

void run_tests()
{
    // kernel launch overheads
    TIME_emptykernel_overhead();

    // memory allocation overheads
    TIME_1byte_allocation_overhead();
    TIME_1024byte_allocation_overhead();
    TIME_kOneMbyte_allocation_overhead();
    TIME_kHundredMbyte_allocation_overhead();
    TIME_kOneGbyte_allocation_overhead();

    // memory copy overheads
    TIME_1byte_copy_overhead();
    TIME_1024byte_copy_overhead();
    TIME_kOneMbyte_copy_overhead();
    TIME_kHundredMbyte_copy_overhead();
    TIME_kOneGbyte_copy_overhead();

    // min-max test
    TIME_minmax_kernel(kOneM, "find min max in a 1 MB array");
    TIME_minmax_kernel(kHundredM, "find min max in a 100 MB array");
    TIME_minmax_kernel(kOneG, "find min max in a 1 GB array");

}

// TODO: Add more tests
// 1. Finding min-max in a kernel
// 2. Sorting in a kernel
// 3. Compilation time of an empty kernel
// 4. Just walking over different sizes of datasets on the GPU
