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

#define U(k,j,i) afU[(k - data->iSharedHeightFirst) * (data->iSharedRowLast - data->iSharedRowFirst + 1)*(data->iSharedColLast - data->iSharedColFirst + 1) + ((j) - data->iSharedRowFirst) * (data->iSharedColLast - data->iSharedColFirst + 1 ) + (i - data->iSharedColFirst)]
#define F(k,j,i) afF[(k - data->iSharedHeightFirst) * (data->iSharedRowLast - data->iSharedRowFirst + 1)*(data->iSharedColLast - data->iSharedColFirst + 1) + ((j) - data->iSharedRowFirst) * (data->iSharedColLast - data->iSharedColFirst + 1 ) + (i - data->iSharedColFirst)]
#define UOLD(k,j,i) afUold[(k - data->iSharedHeightFirst) * (data->iSharedRowLast - data->iSharedRowFirst + 1)*(data->iSharedColLast - data->iSharedColFirst + 1) + ((j) - data->iSharedRowFirst) * (data->iSharedColLast - data->iSharedColFirst + 1 ) + (i - data->iSharedColFirst)]

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
    afUold = data->afUold;

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
	    
	    afUtemp = data->afUold;
            data->afUold = afUold = data->afU;
            data->afU    = afU    = afUtemp;
	
	    QueryNeighbours(data);
            /* copy new solution into old */
            ExchangeJacobiMpiData(data, afUold);
	    

            /* compute stencil, residual and update */
 	    for (k = data->iSharedHeightFirst +1; k < data->iSharedHeightLast; k++){
            	for (j = data->iSharedRowFirst +1; j < data->iSharedRowLast; j++){
                	for (i = data->iSharedColFirst+1; i < data->iSharedColLast; i++){
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
    double *afU,*afF;
    int iReqCnt = 0;
    int i, j, k;
	
    const int iTagMoveLeft = 10;
    const int iTagMoveRight = 11;
    
    const int iTagMoveTop = 12;
    const int iTagMoveBottom = 13;

    const int iTagMoveBack = 14;
    const int iTagMoveFront = 15;

    afU = data->afU;
    afF = data->afF;
    
    MPI_Win_fence(0,data->iSharedWinUold);

    if(data->iMySharedCartCoords[0] != 0){
	for(j = data->iSharedRowFirst; j <= data->iSharedRowLast; j++){
		for(i = data->iSharedColFirst; i <= data->iSharedColLast; i++){
			data->afUoldBack[data->iIndexBackRank[j-data->iSharedRowFirst][i-data->iSharedColFirst]]
			= UOLD(data->iSharedHeightFirst + 1,j,i);
		}
	}
    }
    if(data->iMySharedCartCoords[0] != data->iSharedNBlockHeight - 1){
	for(j = data->iSharedRowFirst; j <= data->iSharedRowLast; j++){
		for(i = data->iSharedColFirst; i <= data->iSharedColLast; i++){
			data->afUoldFront[data->iIndexFrontRank[j-data->iSharedRowFirst][i-data->iSharedColFirst]]
			= UOLD(data->iSharedHeightLast - 1,j,i);
		}
	}
    }

    if(data->iMySharedCartCoords[1] != 0){
	for(k = data->iSharedHeightFirst; k <= data->iSharedHeightLast; k++){
		for(i = data->iSharedColFirst; i <= data->iSharedColLast; i++){
			data->afUoldLeft[data->iIndexLeftRank[k-data->iSharedHeightFirst][i-data->iSharedColFirst]]
			= UOLD(k,data->iSharedRowFirst+1,i);
		}
	}
    }

    if(data->iMySharedCartCoords[1] != data->iSharedNBlockRow - 1){
	for(k = data->iSharedHeightFirst; k <= data->iSharedHeightLast; k++){
		for(i = data->iSharedColFirst; i <= data->iSharedColLast; i++){
			data->afUoldRight[data->iIndexRightRank[k-data->iSharedHeightFirst][i-data->iSharedColFirst]]
			= UOLD(k,data->iSharedRowLast-1,i);
		}
	}
    }
    
    if(data->iMySharedCartCoords[2] != 0){
	for(k = data->iSharedHeightFirst; k <= data->iSharedHeightLast; k++){
		for(j = data->iSharedRowFirst; j <= data->iSharedRowLast; j++){
			data->afUoldTop[data->iIndexTopRank[k-data->iSharedHeightFirst][j-data->iSharedRowFirst]] = UOLD(k,j,data->iSharedColFirst+1);
		}
	}
    }

    if(data->iMySharedCartCoords[2] != data->iSharedNBlockCol - 1){
	for(k = data->iSharedHeightFirst; k <= data->iSharedHeightLast; k++){
		for(j = data->iSharedRowFirst; j <= data->iSharedRowLast; j++){
			data->afUoldBottom[data->iIndexBottomRank[k-data->iSharedHeightFirst][j-data->iSharedRowFirst]]
			= UOLD(k,j,data->iSharedColLast-1);
		}
	}
    }
	
    MPI_Win_fence(0,data->iSharedWinUold);

    if (data->iMyHeadCartCoords[0] != 0 && data->iMySharedCartCoords[0] == 0)
    {
	MPI_Irecv(&UOLD(data->iSharedHeightFirst,data->iSharedRowFirst,data->iSharedColFirst),1,data->iHeightSquareFirstRecv,
        	data->iBackRank, iTagMoveFront, data->iHeightComm,
		&request[iReqCnt]);
	iReqCnt++;
	MPI_Isend(&UOLD(data->iSharedHeightFirst,data->iSharedRowFirst,data->iSharedColFirst),1,data->iHeightSquareLastSend,
		data->iBackRank, iTagMoveBack, data->iHeightComm,
		&request[iReqCnt]);
  	iReqCnt++;
    }
    if (data->iMyHeadCartCoords[0] != data->iHeadNBlockHeight -1 &&  data->iMySharedCartCoords[0] == data->iSharedNBlockHeight - 1)
    {
	MPI_Irecv(&UOLD(data->iSharedHeightFirst,data->iSharedRowFirst,data->iSharedColFirst),1,data->iHeightSquareLastRecv,
		data->iFrontRank, iTagMoveBack, data->iHeightComm,
		&request[iReqCnt]);
	iReqCnt++;
	MPI_Isend(&UOLD(data->iSharedHeightFirst,data->iSharedRowFirst,data->iSharedColFirst),1,data->iHeightSquareFirstSend,
		data->iFrontRank, iTagMoveFront, data->iHeightComm,
		&request[iReqCnt]);
 	iReqCnt++;
    }
    
    if (data->iMyHeadCartCoords[1] != 0 && data->iMySharedCartCoords[1] == 0)
    {
	MPI_Irecv(&UOLD(data->iSharedHeightFirst,data->iSharedRowFirst,data->iSharedColFirst),1,data->iRowSquareFirstRecv,
        	data->iLeftRank, iTagMoveRight, data->iRowComm,
		&request[iReqCnt]);
	iReqCnt++;
	MPI_Isend(&UOLD(data->iSharedHeightFirst,data->iSharedRowFirst,data->iSharedColFirst),1,data->iRowSquareLastSend,
		data->iLeftRank, iTagMoveLeft, data->iRowComm,
		&request[iReqCnt]);
  	iReqCnt++;
    }
    if (data->iMyHeadCartCoords[1] != data->iHeadNBlockRow -1 &&  data->iMySharedCartCoords[1] == data->iSharedNBlockRow - 1)
    {
	MPI_Irecv(&UOLD(data->iSharedHeightFirst,data->iSharedRowFirst,data->iSharedColFirst),1,data->iRowSquareLastRecv,
		data->iRightRank, iTagMoveLeft, data->iRowComm,
		&request[iReqCnt]);
	iReqCnt++;
	MPI_Isend(&UOLD(data->iSharedHeightFirst,data->iSharedRowFirst,data->iSharedColFirst),1,data->iRowSquareFirstSend,
		data->iRightRank, iTagMoveRight, data->iRowComm,
		&request[iReqCnt]);
 	iReqCnt++;
    }
    if (data->iMyHeadCartCoords[2] != 0 && data->iMySharedCartCoords[2] == 0)
    {
	MPI_Irecv(&UOLD(data->iSharedHeightFirst,data->iSharedRowFirst,data->iSharedColFirst),1,data->iColSquareFirstRecv,
        	data->iTopRank, iTagMoveBottom, data->iColComm,
		&request[iReqCnt]);
	iReqCnt++;
	MPI_Isend(&UOLD(data->iSharedHeightFirst,data->iSharedRowFirst,data->iSharedColFirst),1,data->iColSquareLastSend,
		data->iTopRank, iTagMoveTop, data->iColComm,
		&request[iReqCnt]);
  	iReqCnt++;
    }
    if (data->iMyHeadCartCoords[2] != data->iHeadNBlockCol -1 &&  data->iMySharedCartCoords[2] == data->iSharedNBlockCol - 1)
    {
	MPI_Irecv(&UOLD(data->iSharedHeightFirst,data->iSharedRowFirst,data->iSharedColFirst),1,data->iColSquareLastRecv,
		data->iBottomRank, iTagMoveTop, data->iColComm,
		&request[iReqCnt]);
	iReqCnt++;
	MPI_Isend(&UOLD(data->iSharedHeightFirst,data->iSharedRowFirst,data->iSharedColFirst),1,data->iColSquareFirstSend,
		data->iBottomRank, iTagMoveBottom, data->iColComm,
		&request[iReqCnt]);
 	iReqCnt++;
    }
    MPI_Waitall(iReqCnt, request, status);
}
