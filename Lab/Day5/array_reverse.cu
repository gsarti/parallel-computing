#include <stdio.h>
#include <math.h>

#define N (2048*2048)
#define THREAD_PER_BLOCK 512

__global__ void reverse(int * in, int * out, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    out[index] = in[size - index - 1];
}

int main()
{
    int * in, * out;
    int * d_in, * d_out;
    int size = N * sizeof(int);
    int i;

    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);
    
    in = (int *)malloc(size);
    out = (int *)malloc(size);
    
    for(i = 0; i<N; ++i)
    {
	in[i] = i;	
    } 
    
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
    
    reverse<<< N/THREAD_PER_BLOCK, THREAD_PER_BLOCK >>>(d_in, d_out, N);
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    for(i=0; i<N; ++i)
    {
	if(out[i] != in[N-i-1]) {
	    printf("error\n");
	    break;
	}
    }
    if(i == N){
        printf("correct\n");
    }

    free(in); free(out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
