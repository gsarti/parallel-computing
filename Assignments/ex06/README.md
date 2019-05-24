# Exercise 6: Optimized matrix transpose with CUDA

## Task

The purpose of the exercise was to implement a matrix transpose in CUDA that could abide to the blocking for cache approach to exploit GPU shared memory. We used a naive transpose function as baseline and both execution time (s) and bandwidth (GB/s) as metrics of performance, varying the block size (i.e. number of threads) to 64, 128, 256, 512 and 1024.

## Results

By exploiting the blocking for cache mechanism inside the GPU, we are able to double the bandwidth for the original matrix transposition performed with a naive implementation in CUDA, halving the time required to carry out the computation.

## Reproducibility

In order to test the code, take the following steps:

* Login inside Ulysses using your account.

* Clone this repo using `git clone git@github.com:gsarti/parallel-computing.git`.

* Move inside the assignment folders using `cd parallel-computing/Assignments/`.

* Launch the script using `qsub -q gpu -l nodes=1:ppn=20,walltime=0:10:00 ex06/ex6.sh`.

* Bandwidth and execution times are will be contained in the generated file `ex6.sh.o*`.