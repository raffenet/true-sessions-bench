#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>

void pairwise_exchange(int rank, int size);

int main(void)
{
  struct timespec start, end;
  double my_time, total_time;
  int size, rank;

  clock_gettime(CLOCK_MONOTONIC, &start);
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  pairwise_exchange(rank, size);
  clock_gettime(CLOCK_MONOTONIC, &end);

  my_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;
  MPI_Reduce(&my_time, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    printf("world init %f\n", total_time / size);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  clock_gettime(CLOCK_MONOTONIC, &start);
  MPI_Comm newcomm;
  MPI_Comm_dup(MPI_COMM_WORLD, &newcomm);
  clock_gettime(CLOCK_MONOTONIC, &end);

  my_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;
  MPI_Reduce(&my_time, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    printf("world comm dup %f\n", total_time / size);
  }

  MPI_Finalize();

  return 0;
}

void pairwise_exchange(int rank, int size)
{
  for (int step = 1; step < size; ++step) {
    int send_to = (rank + step) % size;
    int recv_from = (rank - step + size) % size;

    MPI_Sendrecv(NULL, 0, MPI_BYTE, send_to, 0, NULL, 0, MPI_BYTE, recv_from, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
  }
}
