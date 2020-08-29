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

#define U(j,i) afU[((j) - data->iSharedRowFirst) * (data->iSharedColLast - data->iSharedColFirst + 1) + (i - data->iSharedColFirst)]
#define F(j,i) afF[((j) - data->iSharedRowFirst) * (data->iSharedColLast - data->iSharedColFirst + 1) + (i - data->iSharedColFirst)]
#define UOLD(j,i) afUold[((j) - data->iSharedRowFirst) * (data->iSharedColLast - data->iSharedColFirst + 1) + (i - data->iSharedColFirst)]

extern void ExchangeJacobiMpiData(struct JacobiData *data, double *uold);

void Jacobi (struct JacobiData *data)
{
    /*use local pointers for performance reasons*/
    double *afU, *afF, *afUold, *afUtemp;
    int i, j;
    double fLRes;
    
    double ax, ay, b, residual, tmpResd;
    
    afU = data->afU;
    afF = data->afF;
    afUold = data->afUold;

    if (afUold)
    {
        ax = 1.0 / (data->fDx * data->fDx);      /* X-direction coef */
        ay = 1.0 / (data->fDy * data->fDy);      /* Y_direction coef */
        b = -2.0 * (ax + ay) - data->fAlpha;     /* Central coeff */
        residual = 10.0 * data->fTolerance;

        while (data->iIterCount < data->iIterMax && residual > data->fTolerance) 
        {
            residual = 0.0;
            /* copy new solution into old */
            
	    QueryNeighbours(data);

            ExchangeJacobiMpiData(data, afUold);

	    afUtemp = data->afUold;
            data->afUold = afUold = data->afU;
            data->afU    = afU    = afUtemp;

            /* compute stencil, residual and update */
            for (j = data->iSharedRowFirst+1; j < data->iSharedRowLast; j++)
            {
                for (i = data->iSharedColFirst+1; i < data->iSharedColLast; i++)
                {
                    fLRes = ( ax * (UOLD(j, i-1) + UOLD(j, i+1))
                            + ay * (UOLD(j-1, i) + UOLD(j+1, i))
                            +  b * UOLD(j, i) - F(j, i)) / b;

                    /* update solution */
                    U(j,i) = UOLD(j,i) - data->fRelax * fLRes;

                    /* accumulate residual error */
                    residual += fLRes * fLRes;
                }
             }
            /* error check */
            (data->iIterCount)++;
        } /* while */

        tmpResd = residual;
        MPI_Allreduce(&tmpResd, &residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        residual = sqrt(residual) / (data->iCols * data->iRows);
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
    MPI_Request request[8];
    MPI_Status  status[8];
    double *afU;
    int iReqCnt = 0;
    int i, j;
	
    const int iTagMoveLeft = 10;
    const int iTagMoveRight = 11;
    const int iTagMoveTop = 12;
    const int iTagMoveBottom = 13;

    afU    = data->afU;

    MPI_Win_fence(0,data->iSharedWinUnew);
     
    if(data->iMySharedCartCoords[0] != 0){
	for(i = data->iSharedColFirst;i<=data->iSharedColLast;i++){
		data->afUoldTop[data->iIndexTopRank[i-data->iSharedColFirst]] = U(data->iSharedRowFirst+1,i);
    	}
    }
    if(data->iMySharedCartCoords[0] != data->iSharedNBlockRow-1){
	for(i = data->iSharedColFirst;i<=data->iSharedColLast;i++){
		data->afUoldBottom[data->iIndexBottomRank[i-data->iSharedColFirst]] = U(data->iSharedRowLast-1,i);
    	}
    }
    if(data->iMySharedCartCoords[1] != 0){
	for(i = data->iSharedRowFirst;i<=data->iSharedRowLast;i++){
		data->afUoldLeft[data->iIndexLeftRank[i-data->iSharedRowFirst]] = U(i,data->iSharedColFirst + 1);
    	}
    }
    if(data->iMySharedCartCoords[1] != data->iSharedNBlockCol-1){
	for(i = data->iSharedRowFirst;i<=data->iSharedRowLast;i++){
		 data->afUoldRight[data->iIndexRightRank[i-data->iSharedRowFirst]] = U(i,data->iSharedColLast - 1);
    	}
    }

    MPI_Win_fence(0,data->iSharedWinUnew);

    if (data->iMyHeadCartCoords[0] != data->iHeadNBlockRow - 1 && data->iMySharedCartCoords[0] == data->iSharedNBlockRow - 1)
    {
       	MPI_Irecv(&UOLD(data->iSharedRowLast, data->iSharedColFirst),(data->iSharedColLast - data->iSharedColFirst + 1),MPI_DOUBLE, 
               	data->iBottomRank, iTagMoveTop, data->iColComm,
               	&request[iReqCnt]);
       	iReqCnt++;
        MPI_Isend(&U(data->iSharedRowLast - 1, data->iSharedColFirst),(data->iSharedColLast - data->iSharedColFirst + 1), MPI_DOUBLE,
              	data->iBottomRank, iTagMoveBottom, data->iColComm,
                &request[iReqCnt]);
        iReqCnt++;
    }

    if (data->iMyHeadCartCoords[0] != 0 && data->iMySharedCartCoords[0] == 0)
    {
       	MPI_Irecv(&UOLD(data->iSharedRowFirst, data->iSharedColFirst),(data->iSharedColLast - data->iSharedColFirst + 1), MPI_DOUBLE,
               	data->iTopRank,iTagMoveBottom, data->iColComm,
               	&request[iReqCnt]);        
	iReqCnt++;
        MPI_Isend(&U(data->iSharedRowFirst + 1, data->iSharedColFirst),(data->iSharedColLast - data->iSharedColFirst + 1), MPI_DOUBLE, 
               	data->iTopRank, iTagMoveTop, data->iColComm,
               	&request[iReqCnt]);
       	iReqCnt++;
    }
 
    if (data->iMyHeadCartCoords[1] != 0 && data->iMySharedCartCoords[1] == 0)
    {
	MPI_Irecv(&UOLD(data->iSharedRowFirst,data->iSharedColFirst),1,data->iColFirstRecv,
        	data->iLeftRank, iTagMoveRight, data->iRowComm,
		&request[iReqCnt]);
	iReqCnt++;
	MPI_Isend(&U(data->iSharedRowFirst,data->iSharedColFirst),1,data->iColLastSend,
		data->iLeftRank, iTagMoveLeft, data->iRowComm,
		&request[iReqCnt]);
  	iReqCnt++;
    }
    if (data->iMyHeadCartCoords[1] != data->iHeadNBlockCol -1 &&  data->iMySharedCartCoords[1] == data->iSharedNBlockCol - 1)
    {
	MPI_Irecv(&UOLD(data->iSharedRowFirst,data->iSharedColFirst),1,data->iColLastRecv,
		data->iRightRank, iTagMoveLeft, data->iRowComm,
		&request[iReqCnt]);
	iReqCnt++;
	MPI_Isend(&U(data->iSharedRowFirst,data->iSharedColFirst),1,data->iColFirstSend,
		data->iRightRank, iTagMoveRight, data->iRowComm,
		&request[iReqCnt]);
 	iReqCnt++;
    }

    MPI_Waitall(iReqCnt, request, status);

}
