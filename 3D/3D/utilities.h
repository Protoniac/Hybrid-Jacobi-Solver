#include"jacobi.h"
extern void BlockInit(int, int, int, int *,int *,int *);
/* Compute the number of blocks decomposing each dimension following the number of blocks required */

extern void AreaInit(int[3], int, int, int, int, int, int, int, int, int, int,
                     int *,int *,int *, int *, int *, int *);
/* Attribute to each process/node/socket its corresponding Height/Row/Col First/Last */

extern void InitNeighbours(struct JacobiData *);
/* Init Process Neighbours */

extern void InitSubarraysDatatype(struct JacobiData *);
/* Init Datatype for transfering Non-Contiguous pieces of Data */
