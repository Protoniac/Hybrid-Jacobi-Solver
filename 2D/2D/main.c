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
#define U(j,i) data->afU[((j) - data->iRowFirst) * (data->iColLast - data->iColFirst + 1 ) + (i - data->iColFirst)]
#define F(j,i) data->afF[((j) - data->iRowFirst) * (data->iColLast - data->iColFirst + 1 ) + (i - data->iColFirst)]

/*
 * setting values, init mpi, omp etc
 */
void Init(struct JacobiData *data, int *argc, char **argv)
{
    int i;
    int block_lengths[12];
    MPI_Datatype MY_JacobiData;
    MPI_Datatype typelist[12] = { MPI_INT,  MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, 
          MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };

    MPI_Aint displacements[12];

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
    for(i = 0; i < 12; i++)
    {
         block_lengths[i] = 1;
    }
    displacements [0] = (MPI_Aint)offsetof(struct JacobiData, iRows);
    displacements [1] = (MPI_Aint)offsetof(struct JacobiData, iCols);
    displacements [2] = (MPI_Aint)offsetof(struct JacobiData, iNBlockRow);
    displacements [3] = (MPI_Aint)offsetof(struct JacobiData, iNBlockCol);
    displacements [4] = (MPI_Aint)offsetof(struct JacobiData, iRowFirst);
    displacements [5] = (MPI_Aint)offsetof(struct JacobiData, iRowLast);
    displacements [6] = (MPI_Aint)offsetof(struct JacobiData, iColFirst);
    displacements [7] = (MPI_Aint)offsetof(struct JacobiData, iColLast);
    displacements [8] = (MPI_Aint)offsetof(struct JacobiData, iIterMax);
    displacements [9] = (MPI_Aint)offsetof(struct JacobiData, fAlpha);
    displacements [10] = (MPI_Aint)offsetof(struct JacobiData, fRelax);
    displacements [11] = (MPI_Aint)offsetof(struct JacobiData, fTolerance);
    
    MPI_Type_create_struct(12, block_lengths, displacements, typelist, &MY_JacobiData);
    MPI_Type_commit(&MY_JacobiData);
    
    /* Send input parameters to all procs */
    MPI_Bcast(data, 1, MY_JacobiData, 0, MPI_COMM_WORLD);

    BlockInit(data->iRows,data->iCols,data->iNumProcs,&(data->iNBlockRow),&(data->iNBlockCol));
    
    AreaInit(data->iMyRank,data->iNBlockRow,data->iNBlockCol,data->iRows,data->iCols,0,-1,-2,-1,
	     &(data->iRowFirst),&(data->iRowLast),&(data->iColFirst),&(data->iColLast));

    int ndims = 2;
    int dims[2] = {data->iNBlockRow,data->iNBlockCol};
    int periods[2] = {1,1};

    MPI_Cart_create(MPI_COMM_WORLD,ndims,dims,periods,0,&(data->iCartComm));	
    MPI_Comm_rank(data->iCartComm,&(data->iMyCartRank));
    MPI_Cart_coords(data->iCartComm,data->iMyCartRank,ndims,data->iMyCartCoords);
		

    int coords[2] = {0,0};

    coords[0] = data->iMyCartCoords[0]+1;
    coords[1] = data->iMyCartCoords[1];
    MPI_Cart_rank(data->iCartComm,coords,&(data->iBottomRank));

    coords[0] = data->iMyCartCoords[0]-1;
    coords[1] = data->iMyCartCoords[1];
    MPI_Cart_rank(data->iCartComm,coords,&(data->iTopRank));
 
    coords[0] = data->iMyCartCoords[0];
    coords[1] = data->iMyCartCoords[1]-1;
    MPI_Cart_rank(data->iCartComm,coords,&(data->iLeftRank));

    coords[0] = data->iMyCartCoords[0];
    coords[1] = data->iMyCartCoords[1]+1;
    MPI_Cart_rank(data->iCartComm,coords,&(data->iRightRank));

    data->afU = (double*) malloc(
        (data->iRowLast - data->iRowFirst + 1) * (data->iColLast - data->iColFirst + 1) * sizeof(double));
    data->afF = (double*) malloc(
        (data->iRowLast - data->iRowFirst + 1) * (data->iColLast - data->iColFirst + 1) * sizeof(double));

    InitSubarraysDatatype(data);

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
    free (data->afU);
    free (data->afF);

    if(data->iMyCartCoords[1] != 0){
    	MPI_Type_free (&(data->iColFirstRecv));
    	MPI_Type_free (&(data->iColLastSend));
    }
    if(data->iMyCartCoords[1] != data->iNBlockCol - 1){
    	MPI_Type_free (&(data->iColLastRecv));
    	MPI_Type_free (&(data->iColFirstSend));
    }


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

    /* Initialize initial condition and RHS */
    for (j = data->iRowFirst; j <= data->iRowLast; j++)
    {
        for (i = data->iColFirst; i <= data->iColLast; i++)
        {
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
