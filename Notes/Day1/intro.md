# Intro - Parallel Programming 101

To connect to Ulysses: `ssh ULY`
Directory of course material: `/scratch/igirotto/DSSC/`
To copy a folder from Ulysses to local machine: `scp -r username@frontend2.hpc.sissa.it:/scratch/igirotto/DSSC/ /local/path/`

Historically, the most important feature that was improved in computers was frequency, the timer representing the number of operations that can be performed per second.

Since then, our capabilities of incrementing frequency has flattened, and thus the importance of parallelizing tasks has emerged as a key aspect in computation. As of today, all computers have some parallel components.

This course looks at parallel tasks from the programming point of view, with a focus on threads.

## Processes and threads

A process is composed of multiple parts: **instructions, data, files, registers and stack.**

A thread is a lighter representation of a process containing the minimum information to execute instruction. Multiple threads coming from the same process will be able to execute its instructions independently.

**Multithreading** involves the splitting of a process into multiple threads that split the job between them. It can be performed through OpenMP.

**Multiprocessing** involves the parallel execution of multiple instances of the same process which are independent from each other. It can be performed with mpirun.

Each thread has private memory (in the stack) and shared memory (in the heap). Shared memory can be used to allow communication between threads (writing or reading among threads). A **master thread** coordinates the thread group.

An **embarassingly parallel** problem is a problem which is easily splittable between tasks, the most favorable scenario when parallelizing a process. 

The total time of a parallel run is:

<p align="center"><img src="/Notes/Day1/tex/bcb3e48adda8d689a8fe973ced2bc6d8.svg?invert_in_darkmode&sanitize=true" align=middle width=121.24755555pt height=37.210633349999995pt/></p>

where <img src="/Notes/Day1/tex/45daa205a2eacb8e053a24d9ae312e8e.svg?invert_in_darkmode&sanitize=true" align=middle width=12.140467349999989pt height=20.221802699999984pt/> is the serail time of execution, <img src="/Notes/Day1/tex/96fe04ff98e655ae86e2c52bb373f766.svg?invert_in_darkmode&sanitize=true" align=middle width=38.80492439999999pt height=22.465723500000017pt/> is the number of processing elements and <img src="/Notes/Day1/tex/6df6ddacc987bd7a5070beafef47fcc1.svg?invert_in_darkmode&sanitize=true" align=middle width=12.48864374999999pt height=20.221802699999984pt/> is the **overhead** introduced by thread spawning and coordination. Our purpose is to minimize this overhead to make parallelization as fast as possible.

The **granularity** is the level of decomposition that can be applied to the problem. A problem should be fine-grained in presence of many processing elements.

**Functional parallelism**: different tasks are performing different operations at the same time. Seldom seen in computer science.

**Data parallelism**: different tasks are performing the same operation in parallel on equivalent objects.

If a problem is parallelized, the slowest task is the one that will be taken in account since other task will need to wait for it completion (**time of synchronization**). For this reason, it is important to perform **load balancing** to avoid this situation.

## OMP

Values passed as private in the omp scope become uninitialized.

```c
int * a = (int *)malloc(sizeof(int));

#pragma omp parallel private(a)
{
    a[0] = 1; //Segmentation fault, a is not initialized
}
```

We can use firstprivate to initialize each copy of the variable inside threads private memories with the value it held in the previous scope. It shouldn't be used interchangeably with private since the copy task is more demanding than simple allocation.

Iterations can be splitted in chunks both statically and dynamically and distributed among threads. The dynamic approach is seldom used in the scientific computing world since it gives less control on thread management.