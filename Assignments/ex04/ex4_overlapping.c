#include<stdlib.h>
#include<stdio.h>
#include<mpi.h>

#define MIN(x, y) (((x) < (y)) ? (x) : (y)) // Standard min operation on two values

/**
 * @brief Prints the formatted content of a double squared matrix
 * @param mat The double matrix given in input
 * @param size Matrix side length
 */
void print_matrix(double * mat, int size)
{
    for(int i = 0; i < size; i++)
    {
        for(int j = 0; j < size; j++)
        {
            printf("%.1f", mat[i * size + j]);
        }
        printf("\n");
    }
}

/**
 * @brief Copies a 1D double matrix from a pointer into another one;
 * @param out The destination pointer.
 * @param in The input pointer.
 * @param size The total size to be copied.
 */
void copy_mat(double * out, double * in, int size)
{
    for(int i = 0; i < size; i++)
        out[i] = in[i];
}

/**
 * @brief Swaps two pointers to pointers to doubles
 * @param a The first pointer to pointers to doubles
 * @param b The second pointer to pointers to doubles
 */
void swap(double ** a, double ** b)
{
    double * tmp = *a;
    *a = *b;
    *b = tmp;
}

/**
 * Execution should be performed with at least two processes
 * to be successful.
 */
int main(int argc, char* argv[])
{
    
    int rank; // Identifier of the process
    int nproc; // Total number of MPI processes

    int N = atoi(argv[1]);
    int size = N * N; 

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nproc );

    int root = 0; // Receiver process

    int glob_id;
    int loc_id;
    int rem = size % nproc; // The remainder from size distribution on processes
    int loc_rem = rank < rem; // 1 if local process has a remainder to manage, else 0
    int loc_size_no_rem = size / nproc; // Local size without remainders for each process
    int loc_size = loc_size_no_rem + loc_rem; // Size of the matrix built locally by the process
    double* loc_mat = (double*) malloc(loc_size * sizeof(double)); // Matrix built by local process

    for(loc_id = 0; loc_id < loc_size; loc_id++)
    {
        glob_id = loc_size_no_rem * rank + MIN(rank, rem) + loc_id; // Convert local index to global
        loc_mat[loc_id] = (glob_id % N == glob_id / N); // Set diagonal values only to 1
    }

    int proc_size = 0; // Matrix size for a specific process in the loop below
    int proc_rem = 0; // Reminder for a specific process in the loop below

    if(rank != root) // Non-root processes simply sent their local matrices to root
    {
        MPI_Send(loc_mat, loc_size, MPI_DOUBLE, root, 101, MPI_COMM_WORLD);
    }
    else // Root has to assemble all the submatrices produced
    {
        if(N < 10) // Print to standard output the content
        {
            double * glob_mat = (double*) malloc(size * sizeof(double)); // Global identity matrix
            int start = 0;

            for(int i = 0; i < nproc; i++)
            {
                proc_rem = i < rem; // 1 if process i has a remainder to manage, else 0
                proc_size = loc_size_no_rem + proc_rem;

                if(i == root)
                {
                    // Copy local matrix to right offset in final matrix
                    copy_mat(glob_mat + start, loc_mat, proc_size); 
                }
                else
                {
                    // Receive processes matrices to right offset in final matrix
                    MPI_Recv(glob_mat + start, proc_size, MPI_DOUBLE, i, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
                }

                start += proc_size; // Move to new offset
            }
            print_matrix(glob_mat, N);
            free(glob_mat); 
        }
        else // Print to data file
        {
            MPI_Request req;
            MPI_Status stat;
            FILE * data;
            data = fopen("ex4.dat","wb");
            int max_loc_size = (loc_size_no_rem + 1);
            double * wrt_buff = loc_mat; // Buffer used to write to file with overlap
            // Buffer used to receive process matrices with overlap
            double * recv_buff = (double*) malloc(max_loc_size * sizeof(double)); 
            
            int wrt_proc_size = loc_size_no_rem + (0 < rem); // Matrix size for root process
            int recv_proc_size = loc_size_no_rem + (1 < rem); // Matrix size for process 1
            fwrite(wrt_buff, sizeof(double), wrt_proc_size, data);
            MPI_Irecv(recv_buff, recv_proc_size, MPI_DOUBLE, 1, 101, MPI_COMM_WORLD, &req);
            MPI_Wait(&req, &stat);

            int j;

            // Move old to write, receive new buffer and write old to data file
            for(int i = 1; i < nproc - 1; i++)
            {
                j = i + 1;
                wrt_proc_size = loc_size_no_rem + (i < rem);
                recv_proc_size = loc_size_no_rem + (j < rem);
                swap(&recv_buff, &wrt_buff);   
                fwrite(wrt_buff, sizeof(double), wrt_proc_size, data);
                MPI_Irecv(recv_buff, recv_proc_size, MPI_DOUBLE, j, 101, MPI_COMM_WORLD, &req);
                MPI_Wait(&req, &stat);
            }
            
            // Write the last received buffer to data file
            wrt_proc_size = loc_size_no_rem + ((nproc - 1) < rem);
            fwrite(recv_buff, sizeof(double), wrt_proc_size, data);

            fclose(data);
            free(wrt_buff);
            free(recv_buff);
        }
    }
    free(loc_mat);
    MPI_Finalize();
}