# Exercise 4: Distributed All-Reduce Sum of Vectors

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

It is evident how, for a large N (the parameter passed to the executable) using the overlapping communication is much more efficient in terms of time.