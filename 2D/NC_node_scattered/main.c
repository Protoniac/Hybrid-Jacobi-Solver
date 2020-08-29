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

#define U(j,i) data->afU[((j) - data->iSharedRowFirst) * (data->iSharedColLast - data->iSharedColFirst + 1 ) + (i - data->iSharedColFirst)]
#define F(j,i) data->afF[((j) - data->iSharedRowFirst) * (data->iSharedColLast - data->iSharedColFirst + 1 ) + (i - data->iSharedColFirst)]
#define UOLD(j,i) data->afUold[((j) - data->iSharedRowFirst) * (data->iSharedColLast - data->iSharedColFirst + 1) + (i - data->iSharedColFirst)]

void Init(struct JacobiData *data, int *argc, char **argv)
{
    int i;
    int block_lengths[6];
    MPI_Datatype MY_JacobiData;
    MPI_Datatype typelist[6] = { MPI_INT,  MPI_INT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };

    MPI_Aint displacements[6];

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
        printf("\nInput alpha - Helmholtz constant:                   ");
        scanf("%lf", &data->fAlpha);
        printf("\nInput relax - Successive over-relaxation parameter: ");
        scanf("%lf", &data->fRelax);
        printf("\nInput tol - error tolerance for iterrative solver:  ");
        scanf("%lf", &data->fTolerance);
        printf("\nInput mits - Maximum iterations for solver:         ");
        scanf("%d", &data->iIterMax);
#endif
        printf("\n-> matrix size: %dx%d"
               "\n-> alpha: %f"
               "\n-> relax: %f"
               "\n-> tolerance: %f"
               "\n-> #of iterations: %d \n\n",
               data->iCols, data->iRows, data->fAlpha, data->fRelax,
               data->fTolerance, data->iIterMax);
    }

    /* Build MPI Datastructure */
    for(i = 0; i < 6; i++)
    {
         block_lengths[i] = 1;
    }
    displacements [0] = (MPI_Aint)offsetof(struct JacobiData, iRows);
    displacements [1] = (MPI_Aint)offsetof(struct JacobiData, iCols);
    displacements [2] = (MPI_Aint)offsetof(struct JacobiData, iIterMax);
    displacements [3] = (MPI_Aint)offsetof(struct JacobiData, fAlpha);
    displacements [4] = (MPI_Aint)offsetof(struct JacobiData, fRelax);
    displacements [5] = (MPI_Aint)offsetof(struct JacobiData, fTolerance);
    
    MPI_Type_create_struct(6, block_lengths, displacements, typelist, &MY_JacobiData);
    MPI_Type_commit(&MY_JacobiData);
    
    MPI_Bcast(data, 1, MY_JacobiData, 0, MPI_COMM_WORLD);
     
    int ndims = 2;
    int dims[2] = {0,0};
    int periods[2] = {1,1};
    
    MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&(data->iSharedComm));
    MPI_Comm_rank(data->iSharedComm, &(data->iMySharedRank));
    MPI_Comm_size(data->iSharedComm, &(data->iSharedNumProcs));

    int key,color;
    if(data->iMySharedRank == 0){
	color = 0;
	key = data->iMyRank/data->iSharedNumProcs;
    }
    else{
	color = MPI_UNDEFINED;
        key = -1;
    }

    MPI_Comm_split(MPI_COMM_WORLD,color,key,&(data->iHeadComm));

    if(data->iMySharedRank == 0){

    	MPI_Comm_rank(data->iHeadComm, &(data->iMyHeadRank));
    	MPI_Comm_size(data->iHeadComm, &(data->iHeadNumProcs));

	BlockInit(data->iRows,data->iCols,data->iHeadNumProcs,&(data->iHeadNBlockRow),&(data->iHeadNBlockCol));
	
	AreaInit(data->iMyHeadRank,data->iHeadNBlockRow,data->iHeadNBlockCol,data->iRows,data->iCols,0,0,-2,-1,
		 &(data->iRowFirst),&(data->iRowLast),&(data->iColFirst),&(data->iColLast));

    	dims[0] = data->iHeadNBlockRow;
    	dims[1] = data->iHeadNBlockCol;

    	MPI_Cart_create(data->iHeadComm,ndims,dims,periods,0,&(data->iHeadCartComm));	
    	MPI_Comm_rank(data->iHeadCartComm,&(data->iMyHeadCartRank));
    	MPI_Cart_coords(data->iHeadCartComm,data->iMyHeadCartRank,ndims,data->iMyHeadCartCoords);

    }

    displacements [0] = (MPI_Aint)offsetof(struct JacobiData, iHeadNBlockRow);
    displacements [1] = (MPI_Aint)offsetof(struct JacobiData, iHeadNBlockCol);
    displacements [2] = (MPI_Aint)offsetof(struct JacobiData, iRowFirst);
    displacements [3] = (MPI_Aint)offsetof(struct JacobiData, iRowLast);
    displacements [4] = (MPI_Aint)offsetof(struct JacobiData, iColFirst);
    displacements [5] = (MPI_Aint)offsetof(struct JacobiData, iColLast); 

    MPI_Type_create_struct(6, block_lengths, displacements, typelist, &MY_JacobiData);
    MPI_Type_commit(&MY_JacobiData);

    MPI_Bcast(data,1,MY_JacobiData,0,data->iSharedComm);
    MPI_Bcast(data->iMyHeadCartCoords,2,MPI_DOUBLE,0,data->iSharedComm);

    int RowsInShared = (data->iRowLast-data->iRowFirst - 1);
    int ColsInShared = (data->iColLast-data->iColFirst - 1);

    if(data->iMyRank == 0) BlockInit(RowsInShared,ColsInShared,data->iSharedNumProcs,&(data->iSharedNBlockRow),&(data->iSharedNBlockCol));
    MPI_Bcast(&(data->iSharedNBlockRow),1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&(data->iSharedNBlockCol),1,MPI_INT,0,MPI_COMM_WORLD);

    AreaInit(data->iMySharedRank,data->iSharedNBlockRow,data->iSharedNBlockCol,RowsInShared,ColsInShared,0,0,1,2,
	     &(data->iSharedRowFirst),&(data->iSharedRowLast),&(data->iSharedColFirst),&(data->iSharedColLast));
    
    data->iSharedRowFirst += data->iRowFirst;
    data->iSharedRowLast  += data->iRowFirst;

    data->iSharedColFirst += data->iColFirst;
    data->iSharedColLast  += data->iColFirst;

    dims[0] = data->iSharedNBlockRow;
    dims[1] = data->iSharedNBlockCol;

    MPI_Cart_create(data->iSharedComm,ndims,dims,periods,0,&(data->iSharedCartComm));	
    MPI_Comm_rank(data->iSharedCartComm,&(data->iMySharedCartRank));
    MPI_Cart_coords(data->iSharedCartComm,data->iMySharedCartRank,ndims,data->iMySharedCartCoords);

    MPI_Comm_split(MPI_COMM_WORLD,data->iMyHeadCartCoords[0],data->iMyHeadCartCoords[1],&(data->iHeadRowComm));
    MPI_Comm_split(MPI_COMM_WORLD,data->iMyHeadCartCoords[1],data->iMyHeadCartCoords[0],&(data->iHeadColComm));

    MPI_Comm_split(data->iHeadRowComm,data->iMySharedCartCoords[0],
		   data->iMySharedCartCoords[1]+data->iMyHeadCartCoords[1]*data->iSharedNBlockCol,&(data->iRowComm));
    MPI_Comm_split(data->iHeadColComm,data->iMySharedCartCoords[1],
		   data->iMySharedCartCoords[0]+data->iMyHeadCartCoords[0]*data->iSharedNBlockRow,&(data->iColComm));

    MPI_Comm_rank(data->iRowComm,&(data->iLeftRank));
    data->iRightRank = data->iLeftRank;
    data->iLeftRank -= 1;
    data->iRightRank += 1;
    MPI_Comm_rank(data->iColComm,&(data->iTopRank));
    data->iBottomRank = data->iTopRank;
    data->iTopRank -= 1;
    data->iBottomRank += 1;
    
    MPI_Aint winSize;
    int disp_unit = sizeof(double);
    int shared_memory_size = (data->iSharedRowLast - data->iSharedRowFirst + 1)*(data->iSharedColLast - data->iSharedColFirst + 1)*sizeof(double);

    MPI_Win_allocate_shared((MPI_Aint)shared_memory_size,disp_unit,MPI_INFO_NULL,data->iSharedComm,&(data->afUold),&(data->iSharedWinUold));
    MPI_Win_allocate_shared((MPI_Aint)shared_memory_size,disp_unit,MPI_INFO_NULL,data->iSharedComm,&(data->afU)   ,&(data->iSharedWinUnew));

    MPI_Win_shared_query(data->iSharedWinUold,data->iMySharedRank,&winSize,&disp_unit,&(data->afUold));
    MPI_Win_shared_query(data->iSharedWinUnew,data->iMySharedRank,&winSize,&disp_unit,&(data->afU));

    data->afF = (double*) malloc(
        (data->iSharedRowLast - data->iSharedRowFirst + 1) * (data->iSharedColLast - data->iSharedColFirst + 1) * sizeof(double));
    
    InitSubarraysDatatype(data);

    FirstQueryNeighbours(data);

    InitIndexes(data);
    /* calculate dx and dy */

    data->fDx = 2.0 / (data->iCols - 1);
    data->fDy = 2.0 / (data->iRows - 1);

    data->iIterCount = 0;

    return;
}

/*
 * final cleanup routines
 */
void Finish(struct JacobiData * data)
{
    if(data->iMyHeadCartCoords[1] != 0 && data->iMySharedCartCoords[1] == 0){
 	MPI_Type_free (&(data->iColFirstRecv));
    	MPI_Type_free (&(data->iColLastSend));
    }
    if(data->iMyHeadCartCoords[1] != data->iHeadNBlockCol - 1  && data->iMySharedCartCoords[1] == data->iSharedNBlockCol - 1){
    	MPI_Type_free (&(data->iColLastRecv));
    	MPI_Type_free (&(data->iColFirstSend)); 
    }
    
    free(data->afF);

    if(data->iMySharedCartCoords[0] != 0){
	free(data->iIndexTopRank);
    }
    if(data->iMySharedCartCoords[0] != data->iSharedNBlockRow-1){
	free(data->iIndexBottomRank);
    }
    if(data->iMySharedCartCoords[1] != 0){
	free(data->iIndexLeftRank);
    }
    if(data->iMySharedCartCoords[1] != data->iSharedNBlockCol-1){
	free(data->iIndexRightRank);
    }

    MPI_Win_free(&(data->iSharedWinUold));
    MPI_Win_free(&(data->iSharedWinUnew));

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
            0.000013 * data->iIterCount * (data->iCols - 2) * (data->iRows - 2)
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
    int i, j, xx, yy, xx2, yy2;
    double initial_condition_row = 0.0;
    double initial_condition_col = 0.0;
    /* Initialize initial condition and RHS */
    for (j = data->iSharedRowFirst+1; j < data->iSharedRowLast; j++)
    {
        for (i = data->iSharedColFirst+1; i < data->iSharedColLast; i++)
        {
            /* TODO: check if this values have to be ints or doubles */
            xx = (int) (-1.0 + data->fDx * i);
            yy = (int) (-1.0 + data->fDy * j);

            xx2 = xx * xx;
            yy2 = yy * yy;

            U(j,i) = 0.5;
            F(j,i) = -data->fAlpha * (1.0 - xx2) * (1.0 - yy2)
                     + 2.0 * (-2.0 + xx2 + yy2);
        }
    }
}
/*
 * Checks error between numerical and exact solution
 */
void CheckError(struct JacobiData * data)
{
    double error = 0.0;
    int i, j;
    double xx, yy, temp;

    for (j = data->iRowFirst; j <= data->iRowLast; j++)
    {
	//????????????????????vvvvvv????????????????????????
        if ((data->iMyRank != 0 && j == data->iRowFirst) || 
            (data->iMyRank != data->iNumProcs - 1 && j == data->iRowLast))
            continue;

        for (i = data->iColFirst; i <= data->iColLast; i++)
        {
            xx   = -1.0 + data->fDx * i;
            yy   = -1.0 + data->fDy * j;
            temp = U(j,i) - (1.0 - xx * xx) * (1.0 - yy * yy);
            error += temp * temp;
        }
    }

    data->fError = error;
    MPI_Reduce(&data->fError, &error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    data->fError = sqrt(error) / (data->iCols * data->iRows);
        
    return;
}

int main (int argc, char** argv)
{
    int retVal = 0;    /* return value */

    struct JacobiData myData;

    /* sets default values or reads from stdin
     * inits MPI and OpenMP if needed
     * distribute MPI data, calculate MPI bounds
     */
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
