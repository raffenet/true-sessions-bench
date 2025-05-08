#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>

void pairwise_exchange(int rank, int size, MPI_Comm comm);

int main(void)
{
  struct timespec start, end;
  double my_time, total_time;
  MPI_Session session;
  MPI_Group group;
  MPI_Comm comm_world;
  int rank, size;

  clock_gettime(CLOCK_MONOTONIC, &start);
  MPI_Session_init(MPI_INFO_NULL, MPI_ERRORS_ABORT, &session);
  MPI_Group_from_session_pset(session, "mpi://WORLD", &group);
  MPI_Comm_create_from_group(group, "comm_world", MPI_INFO_NULL, MPI_ERRORS_ABORT, &comm_world);
  MPI_Comm_rank(comm_world, &rank);
  MPI_Comm_size(comm_world, &size);
  pairwise_exchange(rank, size, comm_world);
  clock_gettime(CLOCK_MONOTONIC, &end);

  my_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;
  MPI_Reduce(&my_time, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, comm_world);
  if (rank == 0) {
    printf("session world init %f, size %d\n", total_time / size, size);
  }

  MPI_Comm newcomm;
  MPI_Barrier(comm_world);
  clock_gettime(CLOCK_MONOTONIC, &start);
  MPI_Comm_dup(comm_world, &newcomm);
  clock_gettime(CLOCK_MONOTONIC, &end);
  MPI_Comm_free(&newcomm);

  my_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;
  MPI_Reduce(&my_time, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, comm_world);
  if (rank == 0) {
    printf("session world comm dup %f\n", total_time / size);
  }

  MPI_Comm_free(&comm_world);
  MPI_Group_free(&group);
  MPI_Session_finalize(&session);

  return 0;
}

void pairwise_exchange(int rank, int size, MPI_Comm comm)
{
  int dummy;
  for (int step = 1; step < size; ++step) {
    int send_to = (rank + step) % size;
    int recv_from = (rank - step + size) % size;
    MPI_Sendrecv(&dummy, 0, MPI_BYTE, send_to, 0, &dummy, 0, MPI_BYTE, recv_from, 0,
                 comm, MPI_STATUS_IGNORE);
  }
}
