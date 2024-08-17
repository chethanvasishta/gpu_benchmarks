#include <stdio.h>
#include <vector>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <exception>
#define kNumIterations 10

#define TIMER_EVENTS_CREATE \
    cudaEvent_t start, stop; \
    checkCudaErrors(cudaEventCreate(&start)); \
    checkCudaErrors(cudaEventCreate(&stop));

#define TIMER_EVENTS_RECORD_START \
    checkCudaErrors(cudaEventRecord(start));

#define TIMER_EVENTS_RECORD_STOP \
    checkCudaErrors(cudaEventRecord(stop)); \
    checkCudaErrors(cudaEventSynchronize(stop));

#define TIMER_EVENTS_DESTROY \
    checkCudaErrors(cudaEventDestroy(start)); \
    checkCudaErrors(cudaEventDestroy(stop));

#define REPORT_TIME(label) \
    float milliseconds = 0; \
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop)); \
    milliseconds /= kNumIterations; \
    printf("Time to %s: %f ms\n", label, milliseconds);

#define checkCudaErrors(ARG)                                                \
    if (cudaError_t const err = ARG; err != cudaSuccess)                     \
    {                                                                        \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));        \
        exit(EXIT_FAILURE);                                                  \
    }

// kernel launch overheads

__global__ void emptyKernel()
{
}

void TIME_emptykernel_overhead()
{
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

#define TIME_ALLOCATION_OVERHEAD(SIZE, DESC) \
    void TIME_##SIZE##byte_allocation_overhead()  \
    {                                         \
        time_allocation_overhead(SIZE, DESC); \
    }

constexpr int kOneM = 1024 * 1024;
constexpr int kHundredM = 1024 * 1024 * 100;
constexpr int kOneG = 1024 * 1024 * 1024;
TIME_ALLOCATION_OVERHEAD(1, "allocate 1 byte")
TIME_ALLOCATION_OVERHEAD(1024, "allocate 1 KB")
TIME_ALLOCATION_OVERHEAD(kOneM, "allocate 1 MB")
TIME_ALLOCATION_OVERHEAD(kHundredM, "allocate 100 MB")
TIME_ALLOCATION_OVERHEAD(kOneG, "allocate 1 GB")

// memory copy overheads

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
}