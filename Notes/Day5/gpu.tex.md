# Graphics Processing Units (Accelerators)

* We use OpenMP/MPI for fine-grained parallelism using a shared memory between processors belonging to the same system.

* We use MPI for coarse-grained parallelism between different systems belonging to a distributed environment.

* We use CUDA/OPENACC/OPENCC to speed up computations through the use of GPUs.

GPUs are designed for massively parallel data processing, initially conceived for gaming consoles. While they are among the most powerful processors in the world, there is a big difference between nominal power and actual power, which relies on the work of the overall system.

Generally server GPU cards are fanless (no active-cooling) since servers are conceived to provide cooling for all components. This allows to reduce the space occupied by the component inside the rack, which is a crucial aspect.

A GPU basically have the same power consumption of a desktop computer, so when adding one to a PC one needs to double the wattage of the charger.

A GPU has its own RAM and runs data-parallel portions of an application using many lightweight threads (in the order of 1000s) with very little overhead.

Despite an increase in the speed of communication between CPU/GPU and their respective memories, communication between GPU and CPU memory is still an Infiniband of 8/16 GB. Since we need to copy data from CPU to GPU memory and to copy back the results at the end of computation, this is the reason why sometimes using GPUs doesn't actually improve the performances of execution.

## CUDA

CUDA is a C-like language except for special keywords that are language-specific.

Functions can be of three kinds: `__host__`, the default value of normal CPU functions, `__device__` are functions called and executed on GPU, and `__global__` are functions called on CPU but executed on the GPU.

There is a special declaration to make functions execute on GPU, in the form `<<<bt, nt>>>(<args>)`, where nt represents the number of threads per block and bt represents the number of thread blocks. `add<<<N, t>>>()`  will execute the `add` kernel on t threads in each on N blocks.

The compiler used to compile CUDA code is `nvcc` available on Ulysses in the `cuda-toolkit` package.

An hello world exploiting CUDA is:

```cuda
__global__ void kernel( void ) {

}

int main ( void ) {
    kernel <<<1,1>>>();
    printf( "Hello, World!\n" );
    return 0;
}
```

Memory allocation happens under the hood with custom CUDA functions as `cudaMalloc` to allocate memory to device, `cudaFree` to free memory from device and `cudaMemcpy` to copy memory from host to device and vice versa. We don't really know how memory is allocated since the software is proprietary.

The `cudaMalloc` function takes a reference of the pointer we want to allocate since the function is executed on the GPU and the address allocated cannot be returned on the CPU as in normal malloc.

Change of paradigm from normal parallel computing: since GPU are high-throughput devices (practically how many threads as you want), we don't want to split the domain by the number of threads anymore, but instead we can assign exactly one element to each thread instead of looping on the splitted domain

Before (MPI-like):

```c
n_threads = OMP_GET_NUM_THREADS();

start = thread_id * (N/ n_threads);
end = start + (N/n_threads);

for(i=start; i < end; i++) {
    C[i] = A[i] + B[i];
}
```

After (CUDA-like) using the built-in blockIdx CUDA variable:

```cuda
__global__ void add (int * A, int * B, int * C) {
    C[blockIdx.x] =  A[blockIdx.x] + B[blockIdx.x];
}

add<<<N, 1>>>( dev_a, dev_b, dev_c);
```

After (CUDA-like) using the built-in threadIdx.x CUDA variable:

```cuda
__global__ void add (int * A, int * B, int * C) {
    C[threadIdx.x] =  A[threadIdx.x] + B[threadIdx.x];
}

add<<<1, N>>>( dev_a, dev_b, dev_c);
```

Using multiple threads in multiple blocks, we should build a local index starting from the local index threadIdx.x and summing it to the block id (blockIdx.x) times the number of threads per blocks (blockDim.x):

```cuda
__global__ void add(int * A, int * B, int * C) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    C[index] =  A[index] + B[index];
}

N = (2048 * 2048) // Dimension of the problem
THREADS_PER_BLOCK = 512 // Depends on GPU architecture

add<<< N/THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>(dev_a, dev_b, dev_c);
```

## Exercise on Ulysses

We use the gpu or reserved2 queues since they are the ones having GPUs.

`module load cudatoolkit/10.0`

Go to `/u/shared/programs/x86_64/cuda/10.0.130/samples`
to find a lot of tools that can be used with CUDA (e.g. cudaOpenMP to use both cuda and openMP)

* `qsub -q gpu -l nodes=1:ppn=20,walltime=2:00:00 -I` to open an interactive session on the gpu queue.

* `nvidia-smi` to see GPU availability, `watch -n1 nvidia-smi` to see the card updated every second.

* `nvidia-smi topo -mp` to see the topology between GPUs and the CPU.

We will use the `deviceQuery` tool from the CUDA toolkit to obtain information about our devices.

We can see that the GPU is composed by a very large number of cores (2496 CUDA cores) which are very slow (0.71 GHz).

* Max number of threads per block: 1024

The GPU is not composed by raw cores but contains multiprocessors (13 in our case), but there is a nested level of parallelism since processors organize block execution while blocks organize thread execution.

ECC support means the error management capabilities: for SC it is fundamental and should be enabled, which makes the GPUs much more expensives than their gaming counterparts in which this feature isn't present.

To compile the code, we use `nvcc -o add_simple.x add_simple.cu`.
