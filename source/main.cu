#include <stdio.h>
#define kNumIterations 10

// kernel launch overheads

__global__ void emptyKernel()
{
}

void test_time_emptykernel_overhead()
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    for (int i = 0; i < kNumIterations; i++)
    {
        emptyKernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds /= kNumIterations;
    printf("Time to launch an empty kernel: %f ms\n", milliseconds);
}

// memory allocation overheads



// memory copy overheads

void run_tests()
{
    // kernel launch overheads
    test_time_emptykernel_overhead();
}