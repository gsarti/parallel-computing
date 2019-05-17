#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

/**
 * @brief Returns the rank of the process to the left w.r.t. the current one.
 * @param rank The rank of the current process.
 * @param nproc The total number of processes spawned by MPI.
 */
int left_proc(int rank, int nproc)
{
    return (rank - 1 + nproc) % nproc;
}

/**
 * @brief Returns the rank of the process to the right w.r.t. the current one.
 * @param rank The rank of the current process.
 * @param nproc The total number of processes spawned by MPI.
 */
int right_proc(int rank, int nproc)
{
    return (rank + 1) % nproc;
}

/**
 * @brief Swaps two pointers to pointers to int
 * @param a The first pointer to pointers to int
 * @param b The second pointer to pointers to int
 */
void swap(int ** a, int ** b)
{
    int * tmp = *a;
    *a = *b;
    *b = tmp;
}

int main(int argc, char* argv[]) {

    int rank;
    int nproc;

    int N = atoi(argv[1]);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Request req;
    MPI_Status stat;

    int root = 0; // Process on which the final sum will be accumulated

    int * X = (int*)malloc(N * sizeof(int));
    int * recv = (int*)malloc(N * sizeof(int));
    int * sum = (int*)malloc(N * sizeof(int));

    for (int i = 0; i < N; ++i) {
        X[i] = rank;
        sum[i] = 0;
    }

    double start = MPI_Wtime();
    for (int i = 0; i < nproc; ++i)
    {
        MPI_Isend(X, N, MPI_INT, right_proc(rank, nproc), 101, MPI_COMM_WORLD, &req);
        for (int i = 0; i < N; ++i) sum[i] += X[i];
        MPI_Recv(recv, N, MPI_INT, left_proc(rank, nproc), 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Wait(&req, &stat);
        swap(&X, &recv);   
    }
    
    printf("Process: %d\nReceived from: %d\nSum: %d\n", rank, right_proc(rank, nproc), sum[0]);
    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    if (rank == root)
    {
        printf("Elapsed time: %f\n", end - start);
    }
    MPI_Finalize();
    free(X);
    free(recv);
    free(sum);
    return 0;
}