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

#define U(j,i)    afU[((j) - data->iRowFirst) * (data->iColLast - data->iColFirst + 1) + (i - data->iColFirst)]
#define UOLD(j,i) afUold[((j) - data->iRowFirst) * (data->iColLast - data->iColFirst + 1) + (i - data->iColFirst)]
#define F(j,i)    afF[((j) - data->iSharedRowFirst) * (data->iSharedColLast - data->iSharedColFirst + 1) + (i - data->iSharedColFirst)]

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
            ExchangeJacobiMpiData(data, afUold);
	    
            afUtemp = data->afUold;
            data->afUold = afUold = data->afU;
            data->afU    = afU    = afUtemp;
            /* compute stencil, residual and update */
            for (j = data->iSharedRowFirst; j <= data->iSharedRowLast; j++)
            {
                for (i = data->iSharedColFirst; i <= data->iSharedColLast; i++)
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

    afU = data->afU;

    if(data->iMySharedRank == 0){
    	if (data->iMyHeadCartCoords[0] != 0)
    	{
        	MPI_Irecv(&UOLD(data->iRowFirst, data->iColFirst),(data->iColLast - data->iColFirst + 1), MPI_DOUBLE,
                	data->iTopRank,iTagMoveBottom, data->iHeadCartComm,
                	&request[iReqCnt]);
        	iReqCnt++;
        	MPI_Isend(&U(data->iRowFirst + 1, data->iColFirst),(data->iColLast - data->iColFirst + 1), MPI_DOUBLE, 
                	data->iTopRank, iTagMoveTop, data->iHeadCartComm,
                	&request[iReqCnt]);
        	iReqCnt++;
    	}

    	if (data->iMyHeadCartCoords[0] != data->iNBlockRow - 1)
    	{
        	MPI_Irecv(&UOLD(data->iRowLast, data->iColFirst),(data->iColLast - data->iColFirst + 1),MPI_DOUBLE, 
                	data->iBottomRank, iTagMoveTop, data->iHeadCartComm,
                	&request[iReqCnt]);
        	iReqCnt++;
        	MPI_Isend(&U(data->iRowLast - 1, data->iColFirst),(data->iColLast - data->iColFirst + 1), MPI_DOUBLE,
                	data->iBottomRank, iTagMoveBottom, data->iHeadCartComm,
                	&request[iReqCnt]);
        	iReqCnt++;
    	}

    	if (data->iMyHeadCartCoords[1] != 0)
    	{
		MPI_Irecv(&UOLD(data->iRowFirst,data->iColFirst),1,data->iColFirstRecv,
        		data->iLeftRank, iTagMoveRight, data->iHeadCartComm,
			&request[iReqCnt]);
		iReqCnt++;
		MPI_Isend(&U(data->iRowFirst,data->iColFirst),1,data->iColLastSend,
			data->iLeftRank, iTagMoveLeft, data->iHeadCartComm,
			&request[iReqCnt]);
  		iReqCnt++;
    	}
    	if (data->iMyHeadCartCoords[1] != data->iNBlockCol - 1)
    	{
		MPI_Irecv(&UOLD(data->iRowFirst,data->iColFirst),1,data->iColLastRecv,
			data->iRightRank, iTagMoveLeft, data->iHeadCartComm,
			&request[iReqCnt]);
		iReqCnt++;
		MPI_Isend(&U(data->iRowFirst,data->iColFirst),1,data->iColFirstSend,
			data->iRightRank, iTagMoveRight, data->iHeadCartComm,
			&request[iReqCnt]);
 		iReqCnt++;
    	}
    	MPI_Waitall(iReqCnt, request, status);

    }
}
