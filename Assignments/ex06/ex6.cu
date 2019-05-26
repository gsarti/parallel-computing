#include <stdio.h>

#define TILE 32 // Thread block dimension
#define N 8192 // Side of the matrix
#define MATSIZE N * N // Total size of the matrix
#define MEMSIZE MATSIZE * sizeof(double) // Size of matrix in memory

// Generic function to be called for bandwidth testing on GPUs.
typedef void (*kernelFunc)(double *, double *, int);

/**
 * @brief Performs an optimized version of a matrix transposition.
 * @param a The input matrix.
 * @param b The transposed matrix in output.
 * @param size The size of the matrix side.
 *
 * By exploiting GPU shared memory we may decompose the transposition in
 * multiple submatrices transpositions, minimizing global memory accesses
 * by doing them simultaneously for same-tile threads.
 */
__global__ void transposeOptimized(double *a, double *b, int size) 
{
    __shared__ double tile[TILE][TILE];
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    tile[threadIdx.x][threadIdx.y] = a[row * size + col];
    __syncthreads();
    b[col * size + row] = tile[threadIdx.x][threadIdx.y];
}

/**
 * @brief Performs a naive version of a matrix transposition on GPU.
 * @param a The input matrix.
 * @param b The transposed matrix in output.
 * @param size The size of the matrix side.
 */
__global__ void transposeNaive(double *a, double *b, int size) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    b[col * size + row] = a[row * size + col];
}

/**
 * @brief Performs a serial version of a matrix transposition on CPU.
 * @param a The input matrix.
 * @param b The transposed matrix in output.
 * @param size The size of the matrix side.
 */
void transposeCpu(double *a, double *b, int size) 
{
    for (int i = 0; i < size; ++i)
      for (int j = 0; j < size; ++j)
        b[j * size + i] = a[i * size + j];
}

/**
 * @brief Validates the equality of two matrices in input.
 * @param a Matrix a.
 * @param b Matrix b.
 * @param size The size of the matrix side.
 */
int isCorrect(double * a, double * b, int size)
{
    for(int i = 0; i < size; ++i)
        for(int j = 0; j < size; ++j)
            if(b[i * size + j] != a[i * size + j])
                return 0;
    return 1;
}

/**
 * @brief Tests execution time and bandwidth of a transposition kernel on a GPU. 
 * @param kernel The kernel to be tested.
 * @param kernelName The name of the kernel to be tested.
 * @param block_x The x-dimension of the block used to perform blocking for cache.
 * @param block_y The y-dimension of the block used to perform blocking for cache.
 *
 * The transposition is performed as specified by the kernel function and then is
 * validated against a correctly-transposed matrix. GPU time and bandwidth are
 * provided as outputs of the function.
 */
int testCudaBandwidth(kernelFunc kernel, const char * kernelName, int block_x, int block_y)
{
    double * h_in, * h_out;
    double * dev_in, * dev_out;
    double * cpu;

    dim3 block(block_x, block_y);
    dim3 grid(N / block.x, N / block.y);

    h_in = (double *)malloc(MEMSIZE);
    h_out = (double *)malloc(MEMSIZE);
    cpu = (double *)malloc(MEMSIZE);
    cudaMalloc((void **)&dev_in, MEMSIZE);
    cudaMalloc((void **)&dev_out, MEMSIZE);
    
    // Fill input matrix with some indices (for validating transposition).
    for(int i = 0; i < MATSIZE; ++i)
        h_in[i] = i;
    
    // Initial setup of input matrix and timing events.
    cudaMemcpy(dev_in, h_in, MEMSIZE, cudaMemcpyHostToDevice);    
    cudaEvent_t start, stop;
    float exec_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Print some informations about the current task.
    printf("\nTransposing matrix on CPU for validation...\n");
    transposeCpu(h_in, cpu, N);
    printf("\nMatrix size: %dx%d, tile: %dx%d, block:%dx%d \n", N, N, TILE, TILE, block_x, block_y);
    printf("\nKernel: %s\n\n", kernelName);

    // Time kernel execution.
    cudaEventRecord(start);
    kernel<<<grid, block>>>(dev_in, dev_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&exec_time, start, stop);
    cudaMemcpy(h_out, dev_out, MEMSIZE, cudaMemcpyDeviceToHost);
    printf("%s: %s\n", kernelName, isCorrect(h_out, cpu, N) ? "CORRECT" : "FAIL");
    printf("GPU Time: %f\n", exec_time);

    // Bandwidth given by reading and writing a matrix during exec_time,
    // converted to GB/s for readability.
    printf("Bandwidth (GB/s): %f\n", MEMSIZE * 2 / exec_time / 1000000);
    printf("-------------------------------\n");

    // Cleanup
    free(h_in);
    free(h_out);
    free(cpu);
    cudaFree(dev_in);
    cudaFree(dev_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

int main(int argc, char *argv[]) 
{
    testCudaBandwidth(&transposeNaive, "Naive Transpose", 32, 32);
    
    int a[5] = {2, 4, 8, 16, 32};

    for(int i = 0; i < 5; ++i)
        for(int j = 0; j < 5; ++j)
            testCudaBandwidth(&transposeOptimized, "Optimized Transpose", a[i], a[j]);

    return 0;
}