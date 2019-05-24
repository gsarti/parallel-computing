#include <stdio.h>

#define TILE 32
#define BLOCK_X 8
#define BLOCK_Y 16
#define N 8192

__global__ void transposeOptimized(double *a, double *b, int size) 
{
    __shared__ double tile[BLOCK_X][BLOCK_Y];
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    tile[threadIdx.x][threadIdx.y] = a[row * size + col];
    __syncthreads();
    b[col * size + row] = tile[threadIdx.x][threadIdx.y];
}

__global__ void transposeNaive(double *a, double *b, int size) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    b[col * size + row] = a[row * size + col];
}

void transposeCpu(double *a, double *b, int size) 
{
    for (int i = 0; i < size; ++i)
      for (int j = 0; j < size; ++j)
        b[j * size + i] = a[i * size + j];
}

int isCorrect(double * a, double * b, int size)
{
    for(int i = 0; i < size; ++i)
        for(int j = 0; j < size; ++j)
            if(b[i * size + j] != a[i * size + j])
                return 0;
    return 1;
}

int main(int argc, char *argv[]) {

    double * h_in, * h_out_naive, * h_out_opt;
    double * dev_in, * dev_out_naive, * dev_out_opt;
    double * cpu;
    int size = N * N;
    int memsize = size * sizeof(double); 
    
    dim3 block(TILE, TILE);
    dim3 grid(N / block.x, N / block.y);
    dim3 blockOpt(BLOCK_X, BLOCK_Y);
    dim3 gridOpt(N / blockOpt.x, N / blockOpt.y);
    
    h_in = (double *)malloc(memsize);
    h_out_naive = (double *)malloc(memsize);
    h_out_opt = (double *)malloc(memsize);
    cpu = (double *)malloc(memsize);
    cudaMalloc((void **)&dev_in, memsize);
    cudaMalloc((void **)&dev_out_naive, memsize);
    cudaMalloc((void **)&dev_out_opt, memsize);
    
    for(int i = 0; i < size; ++i)
        h_in[i] = i;
    
    cudaMemcpy(dev_in, h_in, memsize, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    float exec_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Transposing matrix on CPU for validation...\n");
    transposeCpu(h_in, cpu, N);
    printf("\nMatrix size: %dx%d, tile: %dx%d\n", N, N, TILE, TILE);
    printf("\nKernel: Naive transpose\n\n");

    cudaEventRecord(start);
    transposeNaive<<<grid, block>>>(dev_in, dev_out_naive, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&exec_time, start, stop);
    cudaMemcpy(h_out_naive, dev_out_naive, memsize, cudaMemcpyDeviceToHost);
    printf("Naive transpose: %s\n", isCorrect(h_out_naive, cpu, N) ? "CORRECT" : "FAIL");
    printf("GPU Time: %f\n", exec_time);
    printf("Bandwidth (GB/s): %f\n", memsize * 2 / exec_time / 1000000);

    free(h_out_naive);
    cudaFree(dev_out_naive);

    printf("\nKernel: Optimized transpose\n\n");

    cudaEventRecord(start);
    transposeOptimized<<<gridOpt, blockOpt>>>(dev_in, dev_out_opt, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&exec_time, start, stop);
    cudaMemcpy(h_out_opt, dev_out_opt, memsize, cudaMemcpyDeviceToHost);
    printf("Optimized transpose: %s\n", isCorrect(h_out_opt, cpu, N) ? "CORRECT" : "FAIL");
    printf("GPU Time: %f\n", exec_time);
    printf("Bandwidth (GB/s): %f\n", memsize * 2 / exec_time / 1000000);

    free(h_in);
    free(h_out_opt);
    free(cpu);
    cudaFree(dev_in);
    cudaFree(dev_out_opt);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}