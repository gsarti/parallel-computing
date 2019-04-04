CC = gcc -std=c99

ex1:
	$(CC) -fopenmp -std=c99 ex01/ex1.c -o ex1.x

ex2:
	$(CC) -fopenmp -std=c99 ex02/ex2.c -o ex2.x

all: ex1 ex2

clean:
	find . -type f -executable -exec rm '{}' \;
