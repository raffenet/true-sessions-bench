#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    struct timespec allred_start, allred_end;
    double time, total_time;

    MPI_Session session;
    MPI_Group group;
    MPI_Comm comm_world;
    int world_rank, world_size;
    int nnodes = atoi(argv[1]);
    int ppn = atoi(argv[2]);

    MPI_Session_init(MPI_INFO_NULL, MPI_ERRORS_ABORT, &session);
    MPI_Group_from_session_pset(session, "mpi://WORLD", &group);
    MPI_Comm_create_from_group(group, "comm_world", MPI_INFO_NULL, MPI_ERRORS_ABORT, &comm_world);

    int a=1, b;
    clock_gettime(CLOCK_MONOTONIC, &allred_start);
    MPI_Allreduce(&a, &b, 1, MPI_INT, MPI_SUM, comm_world);
    clock_gettime(CLOCK_MONOTONIC, &allred_end);

    times[0] = (allred_end.tv_sec - allred_start.tv_sec) + (allred_end.tv_nsec - allred_start.tv_nsec) / 1000000000.0;
    MPI_Reduce(&time, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, comm_world);
    MPI_Comm_rank(comm_world, &world_rank);
    MPI_Comm_size(comm_world, &world_size);
    if (world_rank == 0) {
        printf("dense allreduce %f\n", total_time / world_size);
    }

    MPI_Group_free(&group);
    MPI_Comm_free(&comm_world);
    MPI_Session_finalize(&session);

    return 0;
}
