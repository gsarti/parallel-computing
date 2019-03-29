# Intro - Parallel Programming 101

To connect to Ulysses: `ssh <img src="/Notes/Day1/tex/096c752cfb712cb4c7bd47e07de24187.svg?invert_in_darkmode&sanitize=true" align=middle width=816.5088821999999pt height=874.3379073000001pt/><img src="/Notes/Day1/tex/b7d44c376f0d566f2784da11d1dd0594.svg?invert_in_darkmode&sanitize=true" align=middle width=115.85168759999999pt height=27.634635599999985pt/><img src="/Notes/Day1/tex/d58337d48610f0bf4202bfdfca7f94fe.svg?invert_in_darkmode&sanitize=true" align=middle width=30.182742149999992pt height=39.45205439999997pt/>t_s<img src="/Notes/Day1/tex/2312bddaf76b9c62fe4c5bd0864dc261.svg?invert_in_darkmode&sanitize=true" align=middle width=205.9431957pt height=22.831056599999986pt/>N_{proc}<img src="/Notes/Day1/tex/8f57319d50061a9cec3a9535319d4319.svg?invert_in_darkmode&sanitize=true" align=middle width=282.00289919999994pt height=22.831056599999986pt/>t_0$ is the **overhead** introduced by thread spawning and coordination. Our purpose is to minimize this overhead to make parallelization as fast as possible.

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