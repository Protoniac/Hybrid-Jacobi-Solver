#include <mpi.h>
#if defined(WIN32) || defined (_CLUSTER_OPENMP)
/* fix a visualstudio 2005 bug */
#include <omp.h>
#endif
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include "realtime.h"
#include "jacobi.h"
#include "utilities.h"

#define U(k,j,i)     data->afU[(k - data->iHeightFirst) * (data->iRowLast - data->iRowFirst + 1)*(data->iColLast - data->iColFirst + 1) + ((j) - data->iRowFirst) * (data->iColLast - data->iColFirst + 1 ) + (i - data->iColFirst)]
#define F(k,j,i)     data->afF[(k - data->iHeightFirst) * (data->iRowLast - data->iRowFirst + 1)*(data->iColLast - data->iColFirst + 1) + ((j) - data->iRowFirst) * (data->iColLast - data->iColFirst + 1 ) + (i - data->iColFirst)]
#define UOLD(k,j,i)  data->afUold[(k - data->iHeightFirst) * (data->iRowLast - data->iRowFirst + 1)*(data->iColLast - data->iColFirst + 1) + ((j) - data->iRowFirst) * (data->iColLast - data->iColFirst + 1 ) + (i - data->iColFirst)]

void Init(struct JacobiData *data, int *argc, char **argv)
{
    int i;
    int block_lengths[7];
    MPI_Datatype MY_JacobiData;
    MPI_Datatype typelist[7] = { MPI_INT,MPI_INT, MPI_INT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };

    MPI_Aint displacements[7];

    /* MPI Initialization */
    MPI_Init(argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &data->iMyRank);
    MPI_Comm_size(MPI_COMM_WORLD, &data->iNumProcs);

    if (data->iMyRank == 0)
    {
#ifdef READ_INPUT
        printf("Input n - matrix size in x direction:                 ");
        scanf("%d", &data->iCols);
        printf("\nInput m - matrix size in y direction:               ");
        scanf("%d", &data->iRows);
	printf("\nInput l - matrix size in z direction:		      ");
	scanf("%d", &data->iHeights);
        printf("\nInput alpha - Helmholtz constant:                   ");
        scanf("%lf", &data->fAlpha);
        printf("\nInput relax - Successive over-relaxation parameter: ");
        scanf("%lf", &data->fRelax);
        printf("\nInput tol - error tolerance for iterrative solver:  ");
        scanf("%lf", &data->fTolerance);
        printf("\nInput mits - Maximum iterations for solver:         ");
        scanf("%d", &data->iIterMax);
#endif
        printf("\n-> matrix size: %dx%dx%d"
               "\n-> alpha: %f"
               "\n-> relax: %f"
               "\n-> tolerance: %f"
               "\n-> #of iterations: %d \n\n",
               data->iCols, data->iRows, data->iHeights,data->fAlpha, data->fRelax,
               data->fTolerance, data->iIterMax);
    }

    /* Build MPI Datastructure */
    for(i = 0; i < 7; i++)
    {
         block_lengths[i] = 1;
    }
    displacements [0] = (MPI_Aint)offsetof(struct JacobiData, iHeights);
    displacements [1] = (MPI_Aint)offsetof(struct JacobiData, iRows);
    displacements [2] = (MPI_Aint)offsetof(struct JacobiData, iCols);
    displacements [3] = (MPI_Aint)offsetof(struct JacobiData, iIterMax);
    displacements [4] = (MPI_Aint)offsetof(struct JacobiData, fAlpha);
    displacements [5] = (MPI_Aint)offsetof(struct JacobiData, fRelax);
    displacements [6] = (MPI_Aint)offsetof(struct JacobiData, fTolerance);
    
    MPI_Type_create_struct(7, block_lengths, displacements, typelist, &MY_JacobiData);
    MPI_Type_commit(&MY_JacobiData);
    
    MPI_Bcast(data, 1, MY_JacobiData, 0, MPI_COMM_WORLD);
     
    BlockInit(data->iRows,data->iCols,data->iNumProcs,&(data->iNBlockHeight),&(data->iNBlockRow),&(data->iNBlockCol));
   
    int ndims = 3;
    int dims[3] = {data->iNBlockHeight,data->iNBlockRow,data->iNBlockCol};
    int periods[3] = {1,1,1};

    MPI_Cart_create(MPI_COMM_WORLD,ndims,dims,periods,0,&(data->iCartComm));	
    MPI_Comm_rank(data->iCartComm,&(data->iMyCartRank));
    MPI_Cart_coords(data->iCartComm,data->iMyCartRank,ndims,data->iMyCartCoords);

    AreaInit(data->iMyCartCoords, data->iNBlockHeight,data->iNBlockRow,data->iNBlockCol,data->iHeights, data->iRows, data->iCols, 
	      0,-1,-2,-1,&(data->iHeightFirst), &(data->iHeightLast), &(data->iRowFirst), &(data->iRowLast),&(data->iColFirst),&(data->iColLast));

    MPI_Comm_split(data->iCartComm,data->iMyCartCoords[1],data->iMyCartCoords[1],&(data->iTempHeightComm));
    MPI_Comm_split(data->iCartComm,data->iMyCartCoords[2],data->iMyCartCoords[2],&(data->iTempRowComm));
    MPI_Comm_split(data->iCartComm,data->iMyCartCoords[0],data->iMyCartCoords[0],&(data->iTempColComm));

    MPI_Comm_split(data->iTempHeightComm,data->iMyCartCoords[2],data->iMyCartCoords[2],&(data->iHeightComm));
    MPI_Comm_split(data->iTempRowComm,data->iMyCartCoords[0],data->iMyCartCoords[0],&(data->iRowComm));
    MPI_Comm_split(data->iTempColComm,data->iMyCartCoords[1],data->iMyCartCoords[1],&(data->iColComm));
    InitNeighbours(data);

    InitSubarraysDatatype(data);
 
    data->afF = (double*) malloc((data->iHeightLast - data->iHeightFirst + 1) * (data->iRowLast - data->iRowFirst + 1) 
				 * (data->iColLast - data->iColFirst + 1) * sizeof(double));
    data->afU = (double*) malloc((data->iHeightLast - data->iHeightFirst + 1) * (data->iRowLast - data->iRowFirst + 1) 
				 * (data->iColLast - data->iColFirst + 1) * sizeof(double));
    /* calculate dx and dy */
    data->fDz = 2.0 / (data->iHeights - 1);
    data->fDx = 2.0 / (data->iCols - 1); // inverser ? 
    data->fDy = 2.0 / (data->iRows - 1);

    data->iIterCount = 0;
    return;
}

/*
 * final cleanup routines
 */
void Finish(struct JacobiData * data)
{
    if(data->iMyCartCoords[0] != 0){
 	MPI_Type_free (&(data->iHeightSquareFirstRecv));
    	MPI_Type_free (&(data->iHeightSquareLastSend));
    }
    if(data->iMyCartCoords[0] != data->iNBlockHeight - 1){
    	MPI_Type_free (&(data->iHeightSquareLastRecv));
    	MPI_Type_free (&(data->iHeightSquareFirstSend)); 
    }
    if(data->iMyCartCoords[1] != 0){
 	MPI_Type_free (&(data->iRowSquareFirstRecv));
    	MPI_Type_free (&(data->iRowSquareLastSend));
    }
    if(data->iMyCartCoords[1] != data->iNBlockRow - 1){
    	MPI_Type_free (&(data->iRowSquareLastRecv));
    	MPI_Type_free (&(data->iRowSquareFirstSend)); 
    }
    if(data->iMyCartCoords[2] != 0){
 	MPI_Type_free (&(data->iColSquareFirstRecv));
    	MPI_Type_free (&(data->iColSquareLastSend));
    }
    if(data->iMyCartCoords[2] != data->iNBlockCol - 1){
    	MPI_Type_free (&(data->iColSquareLastRecv));
    	MPI_Type_free (&(data->iColSquareFirstSend)); 
    }
    
    free(data->afF);
    free(data->afU);

    MPI_Finalize();
    
    return;
}

void Verbose(const struct JacobiData * data){
    int namelen; 
    char name[MPI_MAX_PROCESSOR_NAME];                  /* ... MPI processor_name */
    MPI_Get_processor_name(name, &namelen);             /* ... MPI processor_name */

    int core_id = sched_getcpu();                           /* ... core_id */

    printf("MPI process %4i / %4i ON core %4i of node %s\n",data->iMyRank,data->iNumProcs,core_id,name);
}

/*
 * print result summary
 */

void PrintResults(const struct JacobiData * data)
{
    if (data->iMyRank == 0)
    {
        printf(" Number of iterations : %d\n", data->iIterCount);
        printf(" Residual             : %le\n", data->fResidual);
        printf(" Elapsed Time         : %5.7lf\n", 
               data->fTimeStop-data->fTimeStart);
        printf(" MFlops               : %6.6lf\n", 
            0.000016 * data->iIterCount * (data->iHeights - 2) * (data->iCols - 2) * (data->iRows - 2)
            / (data->fTimeStop - data->fTimeStart));
    }
    Verbose(data);  
    return;
}

/*
 * Initializes matrix
 * Assumes exact solution is u(x,y) = (1-x^2)*(1-y^2)
 */
void InitializeMatrix(struct JacobiData * data)
{
    int i, j, k, xx, yy, zz, xx2, yy2, zz2;
    double initial_condition_row = 0.0;
    double initial_condition_col = 0.0;
    /* Initialize initial condition and RHS */
    for (k = data->iHeightFirst; k <= data->iHeightLast; k++){
    	for (j = data->iRowFirst; j <= data->iRowLast; j++){
        	for (i = data->iColFirst; i <= data->iColLast; i++){
            		xx = (int) (-1.0 + data->fDx * i);
            		yy = (int) (-1.0 + data->fDy * j);
			zz = (int) (-1.0 + data->fDz * k);

            		xx2 = xx * xx;
            		yy2 = yy * yy;
			zz2 = zz * zz;

            		U(k,j,i) = 0.0;
            		F(k,j,i) = -data->fAlpha * (1.0 - xx2) * (1.0 - yy2) * (1.0 - zz2)
                     		 + 2.0 * (-2.0 + xx2 + yy2 + zz2);
        	}
    	}
    }
}

int main (int argc, char** argv)
{
    int retVal = 0;    /* return value */

    struct JacobiData myData;

    Init(&myData, &argc, argv);

    if (myData.afU && myData.afF)
    {
        // matrix init
        InitializeMatrix(&myData);
        // starting timer
        myData.fTimeStart = GetRealTime();
	
        // running calculations
        Jacobi(&myData);

        // stopping timer 
        myData.fTimeStop = GetRealTime();    	

        // print result summary
        PrintResults(&myData);
    }
    else
    {
        printf(" Memory allocation failed ...\n");
        retVal = -1;
    }

    /* cleanup */
    Finish(&myData);
    
    return retVal;
}
