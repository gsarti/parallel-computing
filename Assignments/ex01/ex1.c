#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

/**
 * @brief Locally performs the Montecarlo approximation of Pi
 * @param low_bound The lower bound of the problem size interval (for multiple threads)
 * @param up_bound The upper bound of the problem size interval (for multiple threads)
 * @param h 1 / problem size
 * @return double The local Pi approximation
 */
double local_sum(int low_bound, int up_bound, double h)
{
    double pi = 0.0;
    double mp;

    for(int i = low_bound; i < up_bound; ++i)
    {
        mp = i * h + h / 2.0; // Midpoint approximation
        pi +=  h / (1 + mp * mp);
    }
    return 4 * pi;
}

/**
 * @brief Performs serial approximation of Pi
 * @param N Problem size
 * @return double Serial Pi approximation
 */
double serial_approx_pi(int N)
{
    double h = 1.0 / N;
    return local_sum(0, N, h);
}

/**
 * @brief Performs multithreaded approximation of Pi using critical keyword
 * @param N Problem size
 * @return double Multithreaded Pi approximation
 */
double critical_parallel_approx_pi(int N)
{
    double pi = 0.0;
    double h = 1.0/N;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int local_n = N / nthreads;
        int start_pos = local_n * tid;
        int end_pos = start_pos + local_n;

        #pragma omp critical // Bad performances since wait for each thread (= serial)
        pi += local_sum(start_pos, end_pos, h);
    }
    return pi;
}

/**
 * @brief Performs multithreaded approximation of Pi using atomic keyword
 * @param N Problem size
 * @return double Multithreaded Pi approximation
 */
double atomic_parallel_approx_pi(int N)
{
    double pi = 0.0;
    double h = 1.0/N;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int local_n = N / nthreads;
        int start_pos = local_n * tid;
        int end_pos = start_pos + local_n;

        double local_pi = local_sum(start_pos, end_pos, h);

        #pragma omp atomic
        pi += local_pi;
    }
    return pi;
}

/**
 * @brief Performs multithreaded approximation of Pi using reduction keyword
 * @param N Problem size
 * @return double Multithreaded Pi approximation
 */
double reduction_parallel_approx_pi(int N)
{
    double pi = 0.0;
    double h = 1.0/N;

    #pragma omp parallel reduction(+:pi) // All results are accumulated in pi
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int local_n = N / nthreads;
        int start_pos = local_n * tid;
        int end_pos = start_pos + local_n;

        pi += local_sum(start_pos, end_pos, h);
    }
    return pi;
}

int main( int argc, char * argv[] )
{
    double pi;
    int N = atoi(argv[1]);
    printf("Size: %d\n", N);

    double start = omp_get_wtime();
    pi = serial_approx_pi(N);
    double tot_time = omp_get_wtime() - start;
    printf("Serial execution(%.5f s): %.8f\n", tot_time, pi);

    start = omp_get_wtime();
    pi = critical_parallel_approx_pi(N);
    tot_time = omp_get_wtime() - start;
    printf("Critical parallel execution(%.5f s): %.8f\n", tot_time, pi);

    start = omp_get_wtime();
    pi = atomic_parallel_approx_pi(N);
    tot_time = omp_get_wtime() - start;
    printf("Atomic parallel execution(%.5f s): %.8f\n", tot_time, pi);

    start = omp_get_wtime();
    pi = reduction_parallel_approx_pi(N);
    tot_time = omp_get_wtime() - start;
    printf("Reduction parallel execution(%.5f s): %.8f\n", tot_time, pi);

    return 0;
}
