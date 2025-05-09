#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>

void pairwise_exchange(int rank, int size);

int main(void)
{
    struct timespec start, init_end,
        ex_start, ex1_end, ex2_end;
    double times[3], total_times[3];
    int size, rank;

    clock_gettime(CLOCK_MONOTONIC, &start);
    MPI_Init(NULL, NULL);
    clock_gettime(CLOCK_MONOTONIC, &init_end);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    clock_gettime(CLOCK_MONOTONIC, &ex_start);
    pairwise_exchange(rank, size);
    clock_gettime(CLOCK_MONOTONIC, &ex1_end);
    pairwise_exchange(rank, size);
    clock_gettime(CLOCK_MONOTONIC, &ex2_end);

    times[0] = (init_end.tv_sec - start.tv_sec) + (init_end.tv_nsec - start.tv_nsec) / 1000000000.0;
    times[1] = (ex1_end.tv_sec - ex_start.tv_sec) + (ex1_end.tv_nsec - ex_start.tv_nsec) / 1000000000.0;
    times[2] = (ex2_end.tv_sec - ex1_end.tv_sec) + (ex2_end.tv_nsec - ex1_end.tv_nsec) / 1000000000.0;
    MPI_Reduce(times, total_times, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("world init %f, pairwise1 %f, pairwise2 %f\n", total_times[0] / size, total_times[1] / size, total_times[2] / size);
    }

    MPI_Finalize();

    return 0;
}

void pairwise_exchange(int rank, int size)
{
    for (int step = 1; step < size; ++step) {
        int send_to = (rank + step) % size;
        int recv_from = (rank - step + size) % size;

        MPI_Sendrecv(NULL, 0, MPI_BYTE, send_to, 0, NULL, 0, MPI_BYTE, recv_from, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}
