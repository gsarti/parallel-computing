# Assignment 3: Approximate Pi with MPI vs OpenMP

In the third exercise for the course of Parallel Programming, our aim is to compare the times of computing a Pi approximation with OpenMP versus MPI.

## Procedure and Results

**Table 1** presents the times of execution for OpenMP Pi approximation obtained from Assignment 1 using a size of $n = 1000000000$ on mpicc compiler with an optimization level of -O3 on a single 20-core processor.

| # Threads | OMP Reduction time (s) |
|-----------|------------------------|
| 1         | 1.95                   |
| 2         | 1.00                   |
| 4         | 0.51                   |
| 8         | 0.27                   |
| 12        | 0.19                   |
| 16        | 0.14                   |
| 20        | 0.11                   |

> Table 1: Results obtained by re-running the code of Assignment 1, adding the 12-thread step as required by the assignment.

**Table 2** presents the times of execution for MPI Pi approximation using a size of $n = 1000000000$ on mpicc compiler with an optimization level of -O3 on two processors with 20 cores each.

| # Processes | MPI Reduction time (s) |
|-------------|------------------------|
| 2           | 1.00                   |
| 4           | 0.50                   |
| 8           | 0.27                   |
| 12          | 0.19                   |
| 16          | 0.14                   |
| 20          | 0.11                   |
| 24          | 0.08                   |
| 28          | 0.08                   |
| 32          | 0.07                   |
| 36          | 0.06                   |
| 40          | 0.06                   |
> Table 1: Results of Assignment 3.

From the results we can observe that OpenMP and MPI seem to have roughly the same speed up to 20 processes/threads. After that MPI performances keep going down, reaching 0.06 seconds at 40 processes.

An even better result may be obtained by exploiting a hybrid approach using both OpenMP for intra-node multithreading and MPI for inter-node communication, since could possibly reduce the execution time by avoiding unnecessary MPI-communications inside nodes. However, there are multiple factors including topology of the system and type of approach used to carry out multilevel parallelism (e.g. Hybrid master-only, Overlapping communication and computations) that influence the effectiveness of the latter. Often, a pure MPI approach may still be faster. Source: [this interesting Stanford tutorial](http://mc.stanford.edu/cgi-bin/images/7/78/Hybrid_MPI_openMP.pdf)

## Reproducibility

In order to test the code, take the following steps:

* Login inside Ulysses using your account.

* Clone this repo using `git clone git@github.com:gsarti/parallel-computing.git`.

* Move inside the assignment folders using `cd parallel-computing/Assignments/`.

* Launch the script using `qsub -l nodes=2:ppn=20,walltime=0:10:00 ex03/ex3.sh`.

* Execution times and results will be available in the file `ex3.sh.o*`.
