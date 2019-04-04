#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

void serial_exec(const int N, int * a)
{
    int thread_id = 0;
    int nthreads = 1;
    for(int i = 0; i < N; ++i)
    {
        a[i] = thread_id;
    }
}

void static_exec(const int N, int * a)
{
    int nthreads = 10;
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        #pragma omp for schedule(static)
        for(int i = 0; i < N; ++i)
        {
            a[i] = thread_id;
        }
    }
}

void static_exec_chunk(const int N, int * a, int chunk_size)
{
    int nthreads = 10;
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        #pragma omp for schedule(static, chunk_size)
        for(int i = 0; i < N; ++i)
        {
            a[i] = thread_id;
        }
    }
}

void dynamic_exec(const int N, int * a)
{
    int nthreads = 10;
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        #pragma omp for schedule(dynamic)
        for(int i = 0; i < N; ++i)
        {
            a[i] = thread_id;
        }
    }
}

void dynamic_exec_chunk(const int N, int * a, int chunk_size)
{
    int nthreads = 10;
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        #pragma omp for schedule(dynamic, chunk_size)
        for(int i = 0; i < N; ++i)
        {
            a[i] = thread_id;
        }
    }
}

void print_usage( int * a, int N, int nthreads ) {

  int tid, i;
  for( tid = 0; tid < nthreads; ++tid ) {

    fprintf( stdout, "%d: ", tid );

    for( i = 0; i < N; ++i ) {

      if( a[ i ] == tid) fprintf( stdout, "*" );
      else fprintf( stdout, " ");
    }
    printf("\n");
  }
}

int main()
{
    const int N = 250;
    int a[N];
    printf("serial:\n");
    serial_exec(N, a);
    print_usage(a, N, 1);
    printf("schedule(static):\n");
    static_exec(N, a);
    print_usage(a, N, 10);
    printf("schedule(static, 1):\n");
    static_exec_chunk(N, a, 1);
    print_usage(a, N, 10);
    printf("schedule(static, 10):\n");
    static_exec_chunk(N, a, 10);
    print_usage(a, N, 10);
    printf("schedule(dynamic):\n");
    dynamic_exec(N, a);
    print_usage(a, N, 10);
    printf("schedule(dynamic, 1):\n");
    dynamic_exec_chunk(N, a, 1);
    print_usage(a, N, 10);
    printf("schedule(dynamic, 10):\n");
    dynamic_exec_chunk(N, a, 10);
    print_usage(a, N, 10);
}

