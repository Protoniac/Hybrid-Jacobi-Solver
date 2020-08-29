/* 
**************************************************************************
*                                                                        * 
* Program to solve a finite difference discretization of the             *
* Helmholtz equation:   (d2/dx2)u + (d2/dy2)u - alpha u = f              *
* using Jacobi iterative method, with Dirichlect boundary conditions.    *
*                                                                        * 
* Used as an exercise in the course:                                     *
*                     Introduction to Hybrid Programming in HPC          *
*                                                                        * 
* Exercise = the MPI-only version of the code                            *
*   TODO   = add OpenMP directives to get a hybrid MPI+OpenMP code !     *
*          + do various improvements to achieve better performance !     *
*                                                                        * 
* Provided = Exercise (MPI-only) + various Solutions (different levels)  *
*                                                                        * 
* MPI + OpenMP directives are used in this code to achieve parallelism.  *
* All loops are parallelized with default 'static' scheduling.           *
*                                                                        *
* Modified: 2019 - Claudia Blaas-Schenner VSC Team, TU Wien              *
* Modified: 2016 - Georg Hager            RRZE / HPC, Uni. Erlangen      *
* Modified: 2007 - Abdelali Malih         Aachen University (RWTH)       *
* Modified: 1998 - Sanjiv Shah            Kuck and Associates Inc. (KAI) *
* Author:   1998 - Joseph Robicheaux      Kuck and Associates Inc. (KAI) *
*                                                                        * 
*           It is made freely available with the understanding that      *
*           every copy must include this header and that                 *
*           the authors as well as VSC and TU Wien                       *
*           take no responsibility for the use of this program.          *
*                                                                        * 
* Input:  n        Grid dimension in x direction                         *
*         m        Grid dimension in y direction                         *
*         alpha    Helmholtz constant (always greater than 0.0)          *
*         relax    Successice over relaxation parameter                  *
*         tol      Error tolerance for iterative solver                  *
*         mits     Maximum iterations for iterative solver               *
*                                                                        *
* Output: u(n,m) - Dependent variable (solution)                         *
*         f(n,m) - Right hand side function                              *
*                                                                        *
**************************************************************************
*/

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
#define F(k,j,i)     data->afF[(k - data->iSharedHeightFirst) * (data->iSharedRowLast - data->iSharedRowFirst + 1)*(data->iSharedColLast - data->iSharedColFirst + 1) + ((j) - data->iSharedRowFirst) * (data->iSharedColLast - data->iSharedColFirst + 1 ) + (i - data->iSharedColFirst)]
#define UOLD(k,j,i)  data->afUold[(k - data->iHeightFirst) * (data->iRowLast - data->iRowFirst + 1)*(data->iColLast - data->iColFirst + 1) + ((j) - data->iRowFirst) * (data->iColLast - data->iColFirst + 1 ) + (i - data->iColFirst)]

void Init(struct JacobiData *data, int *argc, char **argv)
{
    int i;
    int block_lengths[9];
    MPI_Datatype MY_JacobiData;
    MPI_Datatype typelist[7] = { MPI_INT,MPI_INT, MPI_INT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };

    MPI_Aint displacements[9];

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
    for(i = 0; i < 9; i++)
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
    
    /* Send input parameters to all procs */
    MPI_Bcast(data, 1, MY_JacobiData, 0, MPI_COMM_WORLD);
     
    int ndims = 3;
    int dims[3];
    int periods[3] = {1,1,1};
    
    MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&(data->iSharedComm));
    MPI_Comm_rank(data->iSharedComm, &(data->iMySharedRank));
    MPI_Comm_size(data->iSharedComm, &(data->iSharedNumProcs));

    MPI_Comm socket_comm;   
    int key,color;

    if(data->iMySharedRank < data->iSharedNumProcs / 2 ) color = 0;
    else color = 1;
    key = data->iMySharedRank % (data->iSharedNumProcs / 2);
    
    MPI_Comm_split(data->iSharedComm,color,key,&(socket_comm));

    MPI_Barrier(data->iSharedComm);

    data->iSharedComm = socket_comm;

    MPI_Comm_rank(data->iSharedComm, &(data->iMySharedRank));
    MPI_Comm_size(data->iSharedComm, &(data->iSharedNumProcs));

    if(data->iMySharedRank == 0){
	color = 0;
	key = data->iMyRank/data->iSharedNumProcs;
    }
    else{
	color = MPI_UNDEFINED;
        key = -1;
    }

    int shared_memory_size;

    MPI_Comm_split(MPI_COMM_WORLD,color,key,&(data->iHeadComm));

    if(data->iMySharedRank == 0){

    	MPI_Comm_rank(data->iHeadComm, &(data->iMyHeadRank));
    	MPI_Comm_size(data->iHeadComm, &(data->iHeadNumProcs));

	BlockInit(data->iRows,data->iCols,data->iHeadNumProcs,&(data->iHeadNBlockHeight),&(data->iHeadNBlockRow),&(data->iHeadNBlockCol));
	
	dims[0] = data->iHeadNBlockHeight;
    	dims[1] = data->iHeadNBlockRow;
    	dims[2] = data->iHeadNBlockCol;

    	MPI_Cart_create(data->iHeadComm,ndims,dims,periods,0,&(data->iHeadCartComm));	
    	MPI_Comm_rank(data->iHeadCartComm,&(data->iMyHeadCartRank));
    	MPI_Cart_coords(data->iHeadCartComm,data->iMyHeadCartRank,ndims,data->iMyHeadCartCoords);

	AreaInit(data->iMyHeadCartCoords,data->iHeadNBlockHeight,data->iHeadNBlockRow,data->iHeadNBlockCol,data->iHeights,data->iRows,data->iCols,
		 0,-1,-2,-1,&(data->iHeightFirst),&(data->iHeightLast),&(data->iRowFirst),&(data->iRowLast),&(data->iColFirst),&(data->iColLast));

	shared_memory_size = (data->iRowLast - data->iRowFirst + 1)*(data->iColLast - data->iColFirst + 1)
			     *(data->iHeightLast - data->iHeightFirst + 1)*sizeof(double);
    }
    else shared_memory_size = 0;

    displacements [0] = (MPI_Aint)offsetof(struct JacobiData, iHeadNBlockHeight);
    displacements [1] = (MPI_Aint)offsetof(struct JacobiData, iHeadNBlockRow);
    displacements [2] = (MPI_Aint)offsetof(struct JacobiData, iHeadNBlockCol);
    displacements [3] = (MPI_Aint)offsetof(struct JacobiData, iHeightFirst);
    displacements [4] = (MPI_Aint)offsetof(struct JacobiData, iHeightLast);
    displacements [5] = (MPI_Aint)offsetof(struct JacobiData, iRowFirst);
    displacements [6] = (MPI_Aint)offsetof(struct JacobiData, iRowLast);
    displacements [7] = (MPI_Aint)offsetof(struct JacobiData, iColFirst);
    displacements [8] = (MPI_Aint)offsetof(struct JacobiData, iColLast); 
    MPI_Datatype typelist2[9] = {MPI_INT,MPI_INT,MPI_INT,MPI_INT,MPI_INT,MPI_INT,MPI_INT,MPI_INT,MPI_INT};
    MPI_Type_create_struct(9, block_lengths, displacements, typelist2, &MY_JacobiData);
    MPI_Type_commit(&MY_JacobiData);

    MPI_Bcast(data,1,MY_JacobiData,0,data->iSharedComm);
    MPI_Bcast(data->iMyHeadCartCoords,3,MPI_INT,0,data->iSharedComm);
    
    MPI_Aint winSize;
    int disp_unit = sizeof(double);
    
    MPI_Win_allocate_shared((MPI_Aint)shared_memory_size,disp_unit,MPI_INFO_NULL,data->iSharedComm,&(data->afUold),&(data->iSharedWinUold));
    MPI_Win_allocate_shared((MPI_Aint)shared_memory_size,disp_unit,MPI_INFO_NULL,data->iSharedComm,&(data->afU   ),&(data->iSharedWinUnew));
    MPI_Win_shared_query(data->iSharedWinUold,0,&winSize,&disp_unit,&(data->afUold));
    MPI_Win_shared_query(data->iSharedWinUnew,0,&winSize,&disp_unit,&(data->afU   ));

    int HeightsInShared = (data->iHeightLast-data->iHeightFirst - 1);
    int RowsInShared = (data->iRowLast-data->iRowFirst - 1);
    int ColsInShared = (data->iColLast-data->iColFirst - 1);

    if(data->iMyRank == 0) BlockInit(RowsInShared,ColsInShared,data->iSharedNumProcs,
			             &(data->iSharedNBlockHeight),&(data->iSharedNBlockRow),&(data->iSharedNBlockCol));

    MPI_Bcast(&(data->iSharedNBlockRow),1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&(data->iSharedNBlockCol),1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&(data->iSharedNBlockHeight),1,MPI_INT,0,MPI_COMM_WORLD);

    dims[0] = data->iSharedNBlockHeight;
    dims[1] = data->iSharedNBlockRow;
    dims[2] = data->iSharedNBlockCol;

    MPI_Cart_create(data->iSharedComm,ndims,dims,periods,0,&(data->iSharedCartComm));	
    MPI_Comm_rank(data->iSharedCartComm,&(data->iMySharedCartRank));
    MPI_Cart_coords(data->iSharedCartComm,data->iMySharedCartRank,ndims,data->iMySharedCartCoords);

    AreaInit(data->iMySharedCartCoords,data->iSharedNBlockHeight,data->iSharedNBlockRow,data->iSharedNBlockCol,HeightsInShared,RowsInShared,
	     ColsInShared,1,0,1,0,&(data->iSharedHeightFirst),&(data->iSharedHeightLast),&(data->iSharedRowFirst),&(data->iSharedRowLast),
	     &(data->iSharedColFirst),&(data->iSharedColLast));
    
    data->iSharedHeightFirst += data->iHeightFirst;
    data->iSharedHeightLast  += data->iHeightFirst;

    data->iSharedRowFirst += data->iRowFirst;
    data->iSharedRowLast  += data->iRowFirst;

    data->iSharedColFirst += data->iColFirst;
    data->iSharedColLast  += data->iColFirst;

    MPI_Comm_split(MPI_COMM_WORLD,data->iMyHeadCartCoords[1],data->iMyHeadCartCoords[1],&(data->iHeadHeightComm));
    MPI_Comm_split(MPI_COMM_WORLD,data->iMyHeadCartCoords[2],data->iMyHeadCartCoords[2],&(data->iHeadRowComm));
    MPI_Comm_split(MPI_COMM_WORLD,data->iMyHeadCartCoords[0],data->iMyHeadCartCoords[0],&(data->iHeadColComm));

    MPI_Comm_split(data->iHeadHeightComm,data->iMyHeadCartCoords[2],data->iMyHeadCartCoords[2],&(data->iHeightComm));
    MPI_Comm_split(data->iHeadRowComm,data->iMyHeadCartCoords[0],data->iMyHeadCartCoords[0],&(data->iRowComm));
    MPI_Comm_split(data->iHeadColComm,data->iMyHeadCartCoords[1],data->iMyHeadCartCoords[1],&(data->iColComm));

    MPI_Comm_split(data->iHeightComm,data->iMySharedCartCoords[2],
		   data->iMySharedCartCoords[0]+data->iMyHeadCartCoords[0]*data->iSharedNBlockHeight,&(data->iHeadHeightComm));
    MPI_Comm_split(data->iRowComm,data->iMySharedCartCoords[0],
		   data->iMySharedCartCoords[1]+data->iMyHeadCartCoords[1]*data->iSharedNBlockRow,&(data->iHeadRowComm));
    MPI_Comm_split(data->iColComm,data->iMySharedCartCoords[1],
		   data->iMySharedCartCoords[2]+data->iMyHeadCartCoords[2]*data->iSharedNBlockCol,&(data->iHeadColComm));

    MPI_Comm_split(data->iHeadHeightComm,data->iMySharedCartCoords[1],
	  	   data->iMySharedCartCoords[0]+data->iMyHeadCartCoords[0]*data->iSharedNBlockHeight,&(data->iHeightComm));
    MPI_Comm_split(data->iHeadRowComm,data->iMySharedCartCoords[2],
		   data->iMySharedCartCoords[1]+data->iMyHeadCartCoords[1]*data->iSharedNBlockRow,&(data->iRowComm));
    MPI_Comm_split(data->iHeadColComm,data->iMySharedCartCoords[0],
		   data->iMySharedCartCoords[2]+data->iMyHeadCartCoords[2]*data->iSharedNBlockCol,&(data->iColComm));

    InitNeighbours(data);

    InitSubarraysDatatype(data);
 
    data->afF = (double*) malloc((data->iSharedHeightLast - data->iSharedHeightFirst + 1) * (data->iSharedRowLast - data->iSharedRowFirst + 1) 
				 * (data->iSharedColLast - data->iSharedColFirst + 1) * sizeof(double));
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
    if(data->iMyHeadCartCoords[0] != 0 && data->iMySharedCartCoords[0] == 0){
 	MPI_Type_free (&(data->iHeightSquareFirstRecv));
    	MPI_Type_free (&(data->iHeightSquareLastSend));
    }
    if(data->iMyHeadCartCoords[0] != data->iHeadNBlockHeight - 1  && data->iMySharedCartCoords[0] == data->iSharedNBlockHeight - 1){
    	MPI_Type_free (&(data->iHeightSquareLastRecv));
    	MPI_Type_free (&(data->iHeightSquareFirstSend)); 
    }
    if(data->iMyHeadCartCoords[1] != 0 && data->iMySharedCartCoords[1] == 0){
 	MPI_Type_free (&(data->iRowSquareFirstRecv));
    	MPI_Type_free (&(data->iRowSquareLastSend));
    }
    if(data->iMyHeadCartCoords[1] != data->iHeadNBlockRow - 1  && data->iMySharedCartCoords[1] == data->iSharedNBlockRow - 1){
    	MPI_Type_free (&(data->iRowSquareLastRecv));
    	MPI_Type_free (&(data->iRowSquareFirstSend)); 
    }
    if(data->iMyHeadCartCoords[2] != 0 && data->iMySharedCartCoords[2] == 0){
 	MPI_Type_free (&(data->iColSquareFirstRecv));
    	MPI_Type_free (&(data->iColSquareLastSend));
    }
    if(data->iMyHeadCartCoords[2] != data->iHeadNBlockCol - 1  && data->iMySharedCartCoords[2] == data->iSharedNBlockCol - 1){
    	MPI_Type_free (&(data->iColSquareLastRecv));
    	MPI_Type_free (&(data->iColSquareFirstSend)); 
    }
    
    free(data->afF);

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
        printf(" Solution Error       : %1.12lf\n", data->fError);
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
    for (k = data->iSharedHeightFirst; k <= data->iSharedHeightLast; k++){
    	for (j = data->iSharedRowFirst; j <= data->iSharedRowLast; j++){
        	for (i = data->iSharedColFirst; i <= data->iSharedColLast; i++){
            		/* TODO: check if this values have to be ints or doubles */
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
    /*if(data->iMySharedRank == 0 && data->iMyHeadCartCoords[0] == 0){
	for(i = data->iColFirst ; i<=data->iColLast ; i++){
		UOLD(0,i) = 0.0;
	}
    }
    if(data->iMySharedRank == 0 && data->iMyHeadCartCoords[0] == data->iHeadNBlockRow - 1){
	for(i = data->iColFirst ; i<=data->iColLast ; i++){
		UOLD(data->iRows-1,i) = 0.0;
	}
    }
    if(data->iMySharedRank == 0 && data->iMyHeadCartCoords[1] == 0){
	for(j = data->iRowFirst ; j<=data->iRowLast ; j++){
		UOLD(j,0) = 0.0;
	}
    }
    if(data->iMySharedRank == 0 && data->iMyHeadCartCoords[1] == data->iHeadNBlockCol - 1){
	for(j = data->iRowFirst ; j<=data->iRowLast ; j++){
		UOLD(j,data->iCols-1) = 0.0;
	}
    }
    */
}
void PrintFinalMatrix(struct JacobiData * data)
{
    int k,i, j;
    
    /* Initialize initial condition and RHS */
    for (k = data->iSharedHeightFirst; k <= data->iSharedHeightLast; k++){
	printf("[ ");
    	for (j = data->iSharedRowFirst; j <= data->iSharedRowLast; j++)
    	{
        	for (i = data->iSharedColFirst; i <= data->iSharedColLast; i++)
        	{
        	    printf("%lf ",U(k,j,i));
        	}
    	}
 	printf("]\n");
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
