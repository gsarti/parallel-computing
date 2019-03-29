# Synchronization

A **race condition** exists when either one or more threads read from a location where at least one of them is writing. Code causing a race condition is called **critical section**.

In OMP, the `critical` directive will specify sequential execution (one thread at a time, mutually-exclusive access) for a critical section of code.

In case of single simple read-write operations, we use the `atomic` directive to convert them in special hardware atomic instructions.

```c
#pragma omp parallel
{
    #pragma omp atomic
    x += my_result; // This works

    #pragma omp atomic
    x += func() // This doesn't since func is not atomic!
}
```

The `barrier` directive sets a point where all thread should synchronize before proceeding. There isn't a valid reason to use this directive except for debugging.

It is better to store a result in a local variable and then add it to a shared variable using `atomic` than directly summing the function result using `critical`, since it is more expensive. An even better way is to use the `reduction` clause, which creates a local copy for each thread and puts them together at the end of the block.

When there are works-sharing loops (`#pragma omp for`) inside a parallel section, they generate implicit `barrier` statements after the loops. In order to avoid this behavior, we use the `nowait` clause.

```c
#pragma omp parallel for nowait
{
    /* Doesn't make sense since the
       threads will disappear after
       the block anyways.
    */
}
```

The `single` clause make only the first arrived thread to execute a specific block of code.

The `master` clause is executed only by the master thread.