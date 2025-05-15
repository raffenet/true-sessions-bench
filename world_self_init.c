#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>

int main(void)
{
  struct timespec start, end;
  double my_time;

  clock_gettime(CLOCK_MONOTONIC, &start);
  MPI_Init(NULL, NULL);
  /* send to self to verify communication is functional */
  MPI_Sendrecv(NULL, 0, MPI_INT, 0, 0, NULL, 0, MPI_INT, 0, 0, MPI_COMM_SELF, MPI_STATUS_IGNORE);
  clock_gettime(CLOCK_MONOTONIC, &end);

  my_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
      printf("world self init %f\n", my_time);
  }

  MPI_Finalize();

  return 0;
}
