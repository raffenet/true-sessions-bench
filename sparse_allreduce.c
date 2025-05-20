#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    struct timespec allred_start, allred_end;
    double times[3], total_times[3];

    MPI_Session session;
    MPI_Group group;
    int world_rank, world_size;
    int nnodes = atoi(argv[1]);
    int ppn = atoi(argv[2]);

    MPI_Session_init(MPI_INFO_NULL, MPI_ERRORS_ABORT, &session);
    MPI_Group_from_session_pset(session, "mpi://WORLD", &group);
    MPI_Group_rank(group, &world_rank);
    MPI_Group_size(group, &world_size);

    MPI_Group node_group, roots_group;
    MPI_Comm node_comm, roots_comm;
    int *proc_list = malloc(sizeof(int) * world_size);
    int node_root = (world_rank / ppn) * ppn;
    int is_root = (world_rank == node_root);

    /* node comm */
    for (int i = 0; i < ppn; i++) {
        proc_list[i] = node_root + i;
    }
    MPI_Group_incl(group, ppn, proc_list, &node_group);
    MPI_Comm_create_from_group(node_group, "node", MPI_INFO_NULL, MPI_ERRORS_ARE_FATAL, &node_comm);

    /* roots comm */
    if (is_root) {
        for (int i = 0; i < nnodes; i++) {
            proc_list[i] = i * ppn;
        }
        MPI_Group_incl(group, nnodes, proc_list, &roots_group);
        MPI_Comm_create_from_group(roots_group, "roots", MPI_INFO_NULL, MPI_ERRORS_ARE_FATAL,
                                   &roots_comm);
    }

    int a=1, b;
    clock_gettime(CLOCK_MONOTONIC, &allred_start);
    MPI_Reduce(&a, &b, 1, MPI_INT, MPI_SUM, 0, node_comm);
    if (is_root) {
        MPI_Allreduce(MPI_IN_PLACE, &b, 1, MPI_INT, MPI_SUM, roots_comm);
    }
    MPI_Bcast(&b, 1, MPI_INT, 0, node_comm);
    clock_gettime(CLOCK_MONOTONIC, &allred_end);

    time = (allred_end.tv_sec - allred_start.tv_sec) + (allred_end.tv_nsec - allred_start.tv_nsec) / 1000000000.0;
    MPI_Reduce(&time, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, roots_comm);
    if (world_rank == 0) {
        printf("sparse allreduce %f\n", total_time / world_size);
    }

    free(proc_list);
    MPI_Group_free(&group);
    MPI_Group_free(&node_group);
    MPI_Group_free(&roots_group);
    MPI_Comm_free(&node_comm);
    MPI_Comm_free(&roots_comm);
    MPI_Session_finalize(&session);

    return 0;
}
