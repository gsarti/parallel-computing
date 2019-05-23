#include <stdio.h>

#define SIZE_X 2048 * 4
#define SIZE_Y 2048 * 4
#define TILE 32
#define BLOCK 8
#define SIZE SIZE_X * SIZE_Y

__global__ void transposeNaive(double * in, double * out, int size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    out[col * size + row] = in[row * size + col];
}

__global__ void transposeOptimized(double * in, double * out, int size)
{
    __shared__ double temp[BLOCK][BLOCK];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    temp[threadIdx.x][threadIdx.y] = in[row * size + col];

    __syncthreads();

    out[col * size + row] = temp[threadIdx.x][threadIdx.y];
}

int check(double * a, double * b, int size)
{
    for(int i = 0; i < size * size; ++i)
    {
        if(a[i] != b[(i % size) * size + i / size])
        {
            return 0;
        }
    }
    return 1;
}

int main (int argc, char ** argv)
{
    void (*kernel)(double *, double *, int);
    const char * kernelNames[2] = {"Naive transpose\n", "Optimized transpose\n"};
    dim3 grid(SIZE_X / TILE, SIZE_Y / TILE);
    dim3 threads(TILE, BLOCK); 

    cudaEvent_t start, stop;
    float exec_time = 0;

    const int mem_size = sizeof(double) * SIZE;

    double * h_in = (double *)malloc(mem_size);
    double * h_out = (double *)malloc(mem_size);

    double * d_in, * d_out;
    cudaMalloc((void **)&d_in, mem_size);
    cudaMalloc((void **)&d_out, mem_size);

    for(int i = 0; i < SIZE; ++i)
    {
        h_in[i] = (double)i;
    }

    cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);

    printf("\nMatrix size: %dx%d, tile: %dx%d, block: %dx%d\n\n", SIZE_X, SIZE_Y, TILE, TILE, TILE, BLOCK);
    printf("Kernel: ");
    
    for (int k = 0; k < 2; ++k)
    {
        switch (k) {
            case 0:
                kernel = &transposeNaive;
                break;
            case 1:
                kernel = &transposeOptimized;
                break;
        }

        printf("%s", kernelNames[k]);

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        kernel<<<grid, threads>>>(d_in, d_out, SIZE);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&exec_time, start, stop);

        cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

        printf("\nCORRECT: %s\n", check(h_in, h_out, SIZE) ? "TRUE" : "FALSE");
        printf("memsize: %d, exec_time: %f", memsize, exec_time);
        printf("\nBandwidth: %d GB/s\n", 2. * mem_size / exec_time / 1000000);
    }

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}