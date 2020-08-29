#ifndef _JACOBI_H
#define _JACOBI_H

struct JacobiData
{
    /* input data */
    int iRows;
    int iCols;
    int iNBlockRow;
    int iNBlockCol;
    int iRowFirst;
    int iRowLast;
    int iColFirst;
    int iColLast;
    int iIterMax;
    double fAlpha;
    double fRelax;
    double fTolerance;
    
    /* calculated dx & dy */
    double fDx;
    double fDy;

    /* pointers to the allocated memory */
    double *afU;
    double *afF;

    MPI_Datatype iColFirstSend; 
    MPI_Datatype iColFirstRecv; 

    MPI_Datatype iColLastSend; 
    MPI_Datatype iColLastRecv; 

    /* start and end timestamps */
    double fTimeStart;
    double fTimeStop;

    /* calculated residual (output jacobi) */
    double fResidual;
    /* effective interation count (output jacobi) */
    int iIterCount;

    /* calculated error (output error_check) */
    double fError;
    
    /* MPI-Variables */
    int iMyRank;   /* current process rank (number) */
    MPI_Comm iCartComm;
    int iMyCartRank;
    int iMyCartCoords[2];
    int iNumProcs; /* how many processes */

    int iTopRank;
    int iBottomRank;
    int iLeftRank;
    int iRightRank;
};

/* jacobi calculation routine */
void Jacobi (struct JacobiData *data);

/* final cleanup routines */
void Finish (struct JacobiData *data);

#endif

