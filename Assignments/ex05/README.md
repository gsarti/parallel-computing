# Exercise 5: Distributed All-Reduce Sum of Vectors

## Task

Implement a communication pattern of all_reduce using non-blocking point to point communication, first exchanging one single element (i.e., the rank Id of a given process) among processes. Try to optimize the code for sending in the ring a large set of data and
overlapping the computation (Σ) and the communication (send-recv). In case of a dataset larger than one element the local sum is
considered a vector sum (element by element).

## Results

The task was completed, including the overlapping part. Refer to the reproducibility section to test the results yourself.

It is evident how, for a large problem size (the parameter passed to the executable) using the overlapping communication is much more efficient in terms of time.

## Reproducibility

In order to test the code, take the following steps:

* Login inside Ulysses using your account.

* Clone this repo using `git clone git@github.com:gsarti/parallel-computing.git`.

* Enter interactive execution mode using `qsub -l nodes=1:ppn=20,walltime=0:10:00 -I`.

* Move inside the assignment folders using `cd parallel-computing/Assignments/`.

* Compile the exercise using `make ex5`.

* Run the two exercises:

    * `mpirun -np 20 ./ex5.x 1000000`
    * `mpirun -np 20 ./ex5_overlapping.x 1000000`

* Results and times are printed in console.