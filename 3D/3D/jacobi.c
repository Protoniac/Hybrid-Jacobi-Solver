/* 
**************************************************************************
*                                                                        *
* Subroutine to solve the Helmholtz equation:                            *
* (d2/dx2)u + (d2/dy2)u - alpha u = f                                    *
*                                                                        *
* Solves poisson equation on rectangular grid assuming:                  *
* (1) Uniform discretization in each direction, and                      *
* (2) Dirichlect boundary conditions.                                    *
* Jacobi iterative method is used in this routine.                       *
*                                                                        *
* Input:  n,m         Number of grid points in the X/Y directions        *
*         dx,dy       Grid spacing in the X/Y directions                 *
*         alpha       Helmholtz eqn. coefficient                         *
*         omega       Relaxation factor                                  *
*         f(n,m)      Right hand side function                           *
*         u(n,m)      Dependent variable (solution)                      *
*         tolerance   Tolerance for iterative solver                     *
*         maxit       Maximum number of iterations                       *
*                                                                        *
* Output: u(n,m) - Solution                                              *
*                                                                        *
**************************************************************************
*/

#include <mpi.h>
#if defined(WIN32) || defined (_CLUSTER_OPENMP)
/* fix a visualstudio 2005 bug */
#include <omp.h>
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "jacobi.h"

#define U(k,j,i) afU[(k - data->iHeightFirst) * (data->iRowLast - data->iRowFirst + 1)*(data->iColLast - data->iColFirst + 1) + ((j) - data->iRowFirst) * (data->iColLast - data->iColFirst + 1 ) + (i - data->iColFirst)]
#define F(k,j,i) afF[(k - data->iHeightFirst) * (data->iRowLast - data->iRowFirst + 1)*(data->iColLast - data->iColFirst + 1) + ((j) - data->iRowFirst) * (data->iColLast - data->iColFirst + 1 ) + (i - data->iColFirst)]
#define UOLD(k,j,i) afUold[(k - data->iHeightFirst) * (data->iRowLast - data->iRowFirst + 1)*(data->iColLast - data->iColFirst + 1) + ((j) - data->iRowFirst) * (data->iColLast - data->iColFirst + 1 ) + (i - data->iColFirst)]

extern void ExchangeJacobiMpiData(struct JacobiData *data, double *uold);

void Jacobi (struct JacobiData *data)
{
    /*use local pointers for performance reasons*/
    double *afU, *afF, *afUold, *afUtemp;
    int i, j, k;
    double fLRes;
    
    double az, ax, ay, b, residual, tmpResd;
    
    afU = data->afU;
    afF = data->afF;
    afUold = (double *)malloc((data->iHeightLast - data->iHeightFirst + 1) 
			     *(data->iRowLast - data->iRowFirst + 1) 
			     *(data->iColLast - data->iColFirst + 1)*sizeof(double));
    if (afUold)
    {
	az = 1.0 / (data->fDz * data->fDx * data->fDy);
        ax = 1.0 / (data->fDx * data->fDz * data->fDx);      /* X-direction coef */
        ay = 1.0 / (data->fDy * data->fDy * data->fDz);      /* Y_direction coef */
        b = -2.0 * (ax + ay) - data->fAlpha;     /* Central coeff */
        residual = 10.0 * data->fTolerance;

        while (data->iIterCount < data->iIterMax && residual > data->fTolerance) 
        {
            residual = 0.0;
            /* copy new solution into old */
	    afUtemp = afUold;
            afUold = data->afU;
            data->afU    = afU    = afUtemp;

            ExchangeJacobiMpiData(data, afUold);
	    
            /* compute stencil, residual and update */
 	    for (k = data->iHeightFirst+1; k < data->iHeightLast; k++){
            	for (j = data->iRowFirst+1; j < data->iRowLast; j++){
                	for (i = data->iColFirst+1; i < data->iColLast; i++){
                    		fLRes = ( az * (UOLD(k-1,j,i) + UOLD(k+1,j,i)) 
				+ ax * (UOLD(k,j, i-1) + UOLD(k,j, i+1))
                            	+ ay * (UOLD(k,j-1, i) + UOLD(k,j+1, i))
                            	+  b * UOLD(k,j, i) - F(k,j, i)) / b;

        		            /* update solution */
                    		U(k,j,i) = UOLD(k,j,i) - data->fRelax * fLRes;
	
                    		/* accumulate residual error */
                   		residual += fLRes * fLRes;
		    		//printf("residual : %lf\n",residual);
                	}
             	}
	    }
            /* error check */
            (data->iIterCount)++;
        } /* while */

        tmpResd = residual;
        MPI_Allreduce(&tmpResd, &residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        residual = sqrt(residual) / (data->iHeights * data->iCols * data->iRows);
        data->fResidual = residual;
    }
    else 
    {
        fprintf(stderr,"Error: can't allocate memory\n");
        Finish(data);
        exit(1);
    }
}

void ExchangeJacobiMpiData(struct JacobiData *data, double *afUold)
{
    MPI_Request request[12];
    MPI_Status  status[12];
    double *afU;
    int iReqCnt = 0;
	
    const int iTagMoveLeft = 10;
    const int iTagMoveRight = 11;
    
    const int iTagMoveTop = 12;
    const int iTagMoveBottom = 13;

    const int iTagMoveBack = 14;
    const int iTagMoveFront = 15;

    afU = data->afU;

    if (data->iMyCartCoords[0] != 0)
    {
	MPI_Irecv(&UOLD(data->iHeightFirst,data->iRowFirst,data->iColFirst),1,data->iHeightSquareFirstRecv,
        	data->iBackRank, iTagMoveFront, data->iHeightComm,
		&request[iReqCnt]);
	iReqCnt++;
	MPI_Isend(&UOLD(data->iHeightFirst,data->iRowFirst,data->iColFirst),1,data->iHeightSquareLastSend,
		data->iBackRank, iTagMoveBack, data->iHeightComm,
		&request[iReqCnt]);
  	iReqCnt++;
    }
    if (data->iMyCartCoords[0] != data->iNBlockHeight - 1)
    {
	MPI_Irecv(&UOLD(data->iHeightFirst,data->iRowFirst,data->iColFirst),1,data->iHeightSquareLastRecv,
		data->iFrontRank, iTagMoveBack, data->iHeightComm,
		&request[iReqCnt]);
	iReqCnt++;
	MPI_Isend(&UOLD(data->iHeightFirst,data->iRowFirst,data->iColFirst),1,data->iHeightSquareFirstSend,
		data->iFrontRank, iTagMoveFront, data->iHeightComm,
		&request[iReqCnt]);
 	iReqCnt++;
    }
    
    if (data->iMyCartCoords[1] != 0)
    {
	MPI_Irecv(&UOLD(data->iHeightFirst,data->iRowFirst,data->iColFirst),1,data->iRowSquareFirstRecv,
        	data->iLeftRank, iTagMoveRight, data->iRowComm,
		&request[iReqCnt]);
	iReqCnt++;
	MPI_Isend(&UOLD(data->iHeightFirst,data->iRowFirst,data->iColFirst),1,data->iRowSquareLastSend,
		data->iLeftRank, iTagMoveLeft, data->iRowComm,
		&request[iReqCnt]);
  	iReqCnt++;
    }
    if (data->iMyCartCoords[1] != data->iNBlockRow - 1)
    {
	MPI_Irecv(&UOLD(data->iHeightFirst,data->iRowFirst,data->iColFirst),1,data->iRowSquareLastRecv,
		data->iRightRank, iTagMoveLeft, data->iRowComm,
		&request[iReqCnt]);
	iReqCnt++;
	MPI_Isend(&UOLD(data->iHeightFirst,data->iRowFirst,data->iColFirst),1,data->iRowSquareFirstSend,
		data->iRightRank, iTagMoveRight, data->iRowComm,
		&request[iReqCnt]);
 	iReqCnt++;
    }
 
    if (data->iMyCartCoords[2] != 0)
    {
	MPI_Irecv(&UOLD(data->iHeightFirst,data->iRowFirst,data->iColFirst),1,data->iColSquareFirstRecv,
        	data->iTopRank, iTagMoveBottom, data->iColComm,
		&request[iReqCnt]);
	iReqCnt++;
	MPI_Isend(&UOLD(data->iHeightFirst,data->iRowFirst,data->iColFirst),1,data->iColSquareLastSend,
		data->iTopRank, iTagMoveTop, data->iColComm,
		&request[iReqCnt]);
  	iReqCnt++;
    }
    if (data->iMyCartCoords[2] != data->iNBlockCol - 1)
    {
	MPI_Irecv(&UOLD(data->iHeightFirst,data->iRowFirst,data->iColFirst),1,data->iColSquareLastRecv,
		data->iBottomRank, iTagMoveTop, data->iColComm,
		&request[iReqCnt]);
	iReqCnt++;
	MPI_Isend(&UOLD(data->iHeightFirst,data->iRowFirst,data->iColFirst),1,data->iColSquareFirstSend,
		data->iBottomRank, iTagMoveBottom, data->iColComm,
		&request[iReqCnt]);
 	iReqCnt++;
    }
 
    MPI_Waitall(iReqCnt, request, status);
}
