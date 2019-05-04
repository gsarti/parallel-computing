#include <stdio.h>
#include <math.h>

#define N 8
#define THREAD_PER_BLOCK 2

__global__ void transpose(int * in, int * out, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    out[index] = in[(index / size) + size * (index % size)];
}

int main()
{
    int * in, * out;
    int * d_in, * d_out;
    int size = N * N * sizeof(int);
    int i;

    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);
    
    in = (int *)malloc(size);
    out = (int *)malloc(size);
    
    for(i = 0; i<N*N; ++i)
    {
	in[i] = i;	
    } 
    
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
    
    transpose<<< N*N/THREAD_PER_BLOCK, THREAD_PER_BLOCK >>>(d_in, d_out, N);
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    for(i=0; i<N*N; ++i)
    {
	printf("%2d  ", in[i]);
        if((i+1)%N == 0) {
	    printf("\n");
	}
    }
    printf("--------\n");
    for(i=0; i<N*N; ++i)
    {
        printf("%2d  ", out[i]);
        if((i+1)%N == 0) {
            printf("\n");
        }
    }
    free(in); free(out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
