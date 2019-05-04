CC = gcc -std=c99

all: ex1 ex2 ex3

ex1:
	$(CC) -fopenmp ex01/ex1.c -o ex1.x

ex2:
	$(CC) -fopenmp ex02/ex2.c -o ex2.x

ex3:
	mpiicc -O3 ex03/ex3.c -o ex3.x

clean:
	find . -type f -executable -exec rm '{}' \;

PHONY: clean all
