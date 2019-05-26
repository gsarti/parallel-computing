# Exercise 6: Optimized matrix transpose with CUDA

## Task

The purpose of the exercise was to implement a matrix transpose in CUDA that could abide to the blocking for cache approach to exploit GPU shared memory. We used a naive transpose function as baseline and both execution time (s) and bandwidth (GB/s) as metrics of performance, varying the block size (i.e. number of threads) to 64, 128, 256, 512 and 1024.

## Results

The original naive transpose algorithm using a thread block of the size of the tile (32x32) achieved a bandwidth of roughly 50.11 GB/s.

Table 1 presents the bandwidth results of execution for the improved version of the algorithm which is tuned to exploit blocking for cache in order to maximize the use of shared memory on the GPU.

|  Block size | 2     | 4     | 8     | 16    | 32    |
|-------------|-------|-------|-------|-------|-------|
| 2           | 4.33  | 8.40  | 15.64 | 27.40 | 50.37 |
| 4           | 7.62  | 15.01 | 28.17 | 53.00 | 81.05 |
| 8           | 12.42 | 25.09 | 47.69 | 77.33 | 94.58 |
| 16          | 18.36 | 35.60 | 60.90 | 77.28 | 78.43 |
| 32          | 21.70 | 41.08 | 41.94 | 43.99 | 42.33 |

As we can see, using a block of size 8x32 we achieve a bandwidth that is almost the double if compared to the naive implementation. This exercise shows how an intelligent use of the cache may greatly improve the results for this kind of operation on a GPU.

Even better performance may be achieved by implementing a **coalesced version** of the transpose algorithm, that greatly reduces the number of global memory accesses by coalescing half-warps of threads into a couple transactions. This even better version is described and implemented thoroughly in an [official NVIDIA document](https://www.cs.colostate.edu/~cs675/MatrixTranspose.pdf) and reaches easily 130 GB/s of bandwidth. The code wasn't test since it was out of the scope for the current exercise and didn't require a lot of thinking, being already implemented in the said document.

## Reproducibility

In order to test the code, take the following steps:

* Login inside Ulysses using your account.

* Clone this repo using `git clone git@github.com:gsarti/parallel-computing.git`.

* Move inside the assignment folders using `cd parallel-computing/Assignments/`.

* Launch the script using `qsub -q gpu -l nodes=1:ppn=20,walltime=0:10:00 ex06/ex6.sh`.

* Bandwidth and execution times are will be contained in the generated file `ex6.sh.o*`.