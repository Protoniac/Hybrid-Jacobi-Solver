#include<mpi.h>
#include"jacobi.h"

extern void BlockInit(int, int, int, int *,int *);
/* Compute the number of blocks which will decompose the global matrix following the number of blocks required */

extern void AreaInit(int, int, int, int, int, int, int, int, int,
                     int *, int *, int *, int *);
/* Attribute to each node/socket/process its corresponding Row/Col First/Last */

extern void FirstQueryNeighbours(struct JacobiData *);
/* Query the shmem adresses of neighbours and store some useful variables for indexes initialization. */

extern void QueryNeighbours(struct JacobiData *);
/* Query the shmem adresses of neighbours and without storing those variables. */

extern void InitSubarraysDatatype(struct JacobiData *);
/* Init Datatypes for transfering Non-Contiguous Datatype. */

extern void InitIndexes(struct JacobiData *);
/* Init Indexes to access Neighbours Shmem area properly. */
