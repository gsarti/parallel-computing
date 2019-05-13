#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

double local_sum_midpoint(int low_bound, int up_bound, double h)
{
    double pi = 0.0;
    double mp;

    for(int i = low_bound; i < up_bound; ++i)
    {
        mp = i * h + h / 2.0;
        pi +=  h / (1 + mp * mp);
    }
    return 4 * pi;
}

int main( int argc, char * argv[] ){

    int rank; // Identifier of the process
    int nproc; // Total number of MPI processes

    int N = atoi(argv[1]);

    double t1, t2;

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nproc );

    int root = nproc - 1; // Process performing reduction
    int print = 0 ; // Process performing printing


    t1 = MPI_Wtime();
    double global_pi;
    double local_pi = 0.0;
    double h = 1.0 / N;
    int local_n = N / nproc;
    int start_pos = local_n * rank;
    int end_pos = start_pos + local_n;

    if(rank != root)
    {
        local_pi += local_sum_midpoint(start_pos, end_pos, h);
    }
    else
    {
        local_pi += local_sum_midpoint(start_pos, N, h);
    }
    MPI_Reduce(&local_pi, &global_pi, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
    t2 = MPI_Wtime();

    // Send to printer
    if(rank == root && print != root)
    {
        MPI_Send(&global_pi, 1, MPI_DOUBLE, print, 101, MPI_COMM_WORLD);
    }

    // Receive and print
    if(rank == print && print != root)
    {
        MPI_Recv(&global_pi, 1, MPI_DOUBLE, root, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Approximated value of Pi: %.8f\nElapsed_time: %.2f\n", global_pi, t2 - t1);
    }
    MPI_Finalize();
    return 0;
}
