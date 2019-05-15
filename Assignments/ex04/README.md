# Exercise 4: Distributed Initialization of Identity Matrix

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