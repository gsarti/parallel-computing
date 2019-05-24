# Exercise 6: Optimized matrix transpose with CUDA

In order to test the code, take the following steps:

* Login inside Ulysses using your account.

* Clone this repo using `git clone git@github.com:gsarti/parallel-computing.git`.

* Move inside the assignment folders using `cd parallel-computing/Assignments/`.

* Launch the script using `qsub -q gpu -l nodes=1:ppn=20,walltime=0:10:00 ex06/ex6.sh`.

* Bandwidth and execution times are will be contained in the generated file `results.txt` inside the same folder.

By exploiting the blocking for cache mechanism inside the GPU, we are able to double the bandwidth for the original matrix transposition performed with a naive implementation in CUDA.