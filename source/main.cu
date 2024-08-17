#include <stdio.h>
__global__ void simpleKernel()
{
    printf("Hello, World!\n");
}

int caller()
{
    simpleKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}