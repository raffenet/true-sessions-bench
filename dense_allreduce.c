#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    struct timespec start, init_end,
        ex_start, ex1_end, ex2_end;
    double times[3], total_times[3];

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
    MPI_Allreduce(&a, &b, 1, MPI_INT, MPI_SUM, comm_world);

    times[0] = (init_end.tv_sec - start.tv_sec) + (init_end.tv_nsec - start.tv_nsec) / 1000000000.0;
    times[1] = (ex1_end.tv_sec - ex_start.tv_sec) + (ex1_end.tv_nsec - ex_start.tv_nsec) / 1000000000.0;
    times[2] = (ex2_end.tv_sec - ex1_end.tv_sec) + (ex2_end.tv_nsec - ex1_end.tv_nsec) / 1000000000.0;
    MPI_Reduce(times, total_times, 3, MPI_DOUBLE, MPI_SUM, 0, comm_world);
    MPI_Comm_rank(comm_world, &world_rank);
    MPI_Comm_size(comm_world, &world_size);
    if (world_rank == 0) {
        printf("world init %f, pairwise1 %f, pairwise2 %f\n", total_times[0] / world_size,
               total_times[1] / world_size, total_times[2] / world_size);
    }

    MPI_Group_free(&group);
    MPI_Comm_free(&comm_world);
    MPI_Session_finalize(&session);

    return 0;
}
