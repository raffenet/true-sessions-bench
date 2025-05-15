#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>

int main(void)
{
  struct timespec start, end;
  double my_time;
  MPI_Session session;
  MPI_Group group;
  MPI_Comm comm_self;

  clock_gettime(CLOCK_MONOTONIC, &start);
  MPI_Session_init(MPI_INFO_NULL, MPI_ERRORS_ABORT, &session);
  MPI_Group_from_session_pset(session, "mpi://SELF", &group);
  MPI_Comm_create_from_group(group, "comm_self", MPI_INFO_NULL, MPI_ERRORS_ABORT, &comm_self);
  /* send to self to verify communication is functional */
  MPI_Sendrecv(NULL, 0, MPI_INT, 0, 0, NULL, 0, MPI_INT, 0, 0, comm_self, MPI_STATUS_IGNORE);
  clock_gettime(CLOCK_MONOTONIC, &end);

  my_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;

  /* hack to find rank 0 in absence of world comm */
  int rank;
  if (getenv("PMI_RANK")) {
    rank = atoi(getenv("PMI_RANK"));
  } else if (getenv("PMIX_RANK")) {
    rank = atoi(getenv("PMIX_RANK"));
  } else {
    assert(0);
  }
  if (rank == 0) {
    printf("self init %f\n", my_time);
  }

  MPI_Comm_free(&comm_self);
  MPI_Group_free(&group);
  MPI_Session_finalize(&session);

  return 0;
}
