#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>

void pairwise_exchange(int rank, int size, MPI_Comm comm);

int main(void)
{
    struct timespec start, init_end,
        ex_start, ex1_end, ex2_end;
    double times[3], total_times[3];
    MPI_Session session;
    MPI_Group group;
    MPI_Comm comm_world;
    int rank, size;

    clock_gettime(CLOCK_MONOTONIC, &start);
    MPI_Session_init(MPI_INFO_NULL, MPI_ERRORS_ABORT, &session);
    MPI_Group_from_session_pset(session, "mpi://WORLD", &group);
    MPI_Comm_create_from_group(group, "comm_world", MPI_INFO_NULL, MPI_ERRORS_ABORT, &comm_world);
    clock_gettime(CLOCK_MONOTONIC, &init_end);

    MPI_Comm_rank(comm_world, &rank);
    MPI_Comm_size(comm_world, &size);

    clock_gettime(CLOCK_MONOTONIC, &ex_start);
    pairwise_exchange(rank, size, comm_world);
    clock_gettime(CLOCK_MONOTONIC, &ex1_end);
    pairwise_exchange(rank, size, comm_world);
    clock_gettime(CLOCK_MONOTONIC, &ex2_end);

    times[0] = (init_end.tv_sec - start.tv_sec) + (init_end.tv_nsec - start.tv_nsec) / 1000000000.0;
    times[1] = (ex1_end.tv_sec - ex_start.tv_sec) + (ex1_end.tv_nsec - ex_start.tv_nsec) / 1000000000.0;
    times[2] = (ex2_end.tv_sec - ex1_end.tv_sec) + (ex2_end.tv_nsec - ex1_end.tv_nsec) / 1000000000.0;
    MPI_Reduce(times, total_times, 3, MPI_DOUBLE, MPI_SUM, 0, comm_world);
    if (rank == 0) {
        printf("session world init %f, pairwise1 %f, pairwise2 %f\n", total_times[0] / size, total_times[1] / size, total_times[2] / size);
    }

    MPI_Comm_free(&comm_world);
    MPI_Group_free(&group);
    MPI_Session_finalize(&session);

    return 0;
}

void pairwise_exchange(int rank, int size, MPI_Comm comm)
{
    for (int step = 1; step < size; ++step) {
        int send_to = (rank + step) % size;
        int recv_from = (rank - step + size) % size;
        MPI_Sendrecv(NULL, 0, MPI_BYTE, send_to, 0, NULL, 0, MPI_BYTE, recv_from, 0,
                     comm, MPI_STATUS_IGNORE);
    }
}
