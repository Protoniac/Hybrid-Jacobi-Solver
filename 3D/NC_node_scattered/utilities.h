#include<mpi.h>
#include"jacobi.h"

extern void BlockInit(int, int, int, int *,int *,int *);
/* Compute the number of blocks decomposing each dimension following the total number of blocks required. */

extern void AreaInit(int[3], int, int, int, int, int, int, int, int, int, int,
                     int *,int *,int *, int *, int *, int *);
/* Attribute to each node/socket/rank its corresponding Height/Row/Col First/Last */

extern void InitSubarraysDatatype(struct JacobiData *);
/* Initialize Datatype for transfering Non-Contiguous pieces of data */

extern void InitIndexes(struct JacobiData *);
/* Init indexes to access neighbours shmem areas properly */

extern void FirstQueryNeighbours(struct JacobiData *);

extern void QueryNeighbours(struct JacobiData *);
