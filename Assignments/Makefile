CC = gcc -std=c99
NVCC = nvcc
MPICC = mpicc -std=c99
OPT = -O3
OMP = -fopenmp

all: ex1 ex2 ex3 ex4 ex5 ex6

ex1:
	$(MPICC) $(OPT) $(OMP) ex01/ex1.c -o ex1.x

ex2:
	$(CC) $(OMP) ex02/ex2.c -o ex2.x

ex3:
	$(MPICC) $(OPT) ex03/ex3.c -o ex3.x

ex4:
	$(MPICC) $(OPT) ex04/ex4.c -o ex4.x
	$(MPICC) $(OPT) ex04/ex4_overlapping.c -o ex4_overlapping.x

ex5:
	$(MPICC) $(OPT) ex05/ex5.c -o ex5.x
	$(MPICC) $(OPT) ex05/ex5_overlapping.c -o ex5_overlapping.x

ex6:
	$(NVCC) $(OPT) ex06/ex6.cu -o ex6.x

clean:
	find . -type f -executable -exec rm '{}' \;

PHONY: clean all
