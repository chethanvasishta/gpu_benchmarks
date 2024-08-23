#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

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

constexpr int kOneM = 1024 * 1024;
constexpr int kHundredM = 1024 * 1024 * 100;
constexpr int kOneG = 1024 * 1024 * 1024;
