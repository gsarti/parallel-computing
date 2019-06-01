# Exercise 4: Distributed Initialization of Identity Matrix

## Task

Implement a code to initialize a distributed identity matrix of size (N,N). Print the matrix ordered on standard output if N is smaller than 10, otherwise on a binary file. (Plus) Implement the I/O overlapping the receiving data on process 0 with no-blocking communication, therefore overlapping I/O operations on disk with data echange between the processes.

## Results

The task was completed, including the overlapping part. Refer to the reproducibility section to test the results yourself.

## Reproducibility

In order to test the code, take the following steps:

* Login inside Ulysses using your account.

* Clone this repo using `git clone git@github.com:gsarti/parallel-computing.git`.

* Enter interactive execution mode using `qsub -l nodes=1:ppn=20,walltime=0:10:00 -I`.

* Move inside the assignment folders using `cd parallel-computing/Assignments/`.

* Compile the exercise using `make ex4`.

* Run the two exercises using the available conformations:

    * `mpirun -np 10 ./ex4.x 8`
    * `mpirun -np 10 ./ex4.x 100`
    * `mpirun -np 10 ./ex4_overlapping.x 8`
    * `mpirun -np 10 ./ex4_overlapping.x 100`

* Results are printed in console and inside the `ex4.dat` files.