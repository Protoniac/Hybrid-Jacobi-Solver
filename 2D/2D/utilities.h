#include<mpi.h>
#include"jacobi.h"
extern void BlockInit(int, int, int, int *,int *);
/*Scatter the matrix into blocks according to the number of blocks required*/
extern void AreaInit(int, int, int, int, int, int, int, int, int,
                     int *, int *, int *, int *);
/*Attribute to each node/socket/process its corresponding Row/Col First/Last*/
extern void InitSubarraysDatatype(struct JacobiData *);
/*Initialize Datatypes for transferring Non-Contiguous piece of data*/
