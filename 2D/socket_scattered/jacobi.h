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

    int iHeadNBlockRow;
    int iHeadNBlockCol;
    
    int iSharedNBlockRow;
    int iSharedNBlockCol; 
    int iSharedRowFirst;
    int iSharedRowLast;
    int iSharedColFirst;
    int iSharedColLast;

    int iIterMax;
    double fAlpha;
    double fRelax;
    double fTolerance;
    
    /* calculated dx & dy */
    double fDx;
    double fDy;

    /* pointers to the allocated memory */
    double *afU;
    double *afUold;
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

    MPI_Comm iHeadComm;
    MPI_Comm iSharedComm;
    MPI_Comm iHeadCartComm;
    MPI_Comm iSharedCartComm;
    
    MPI_Comm iWorldCartComm;
    int iMyWorldCartCoords[2];
    int iMyWorldCartRank;

    MPI_Comm iHeadRowComm;
    MPI_Comm iHeadColComm;

    MPI_Comm iRowComm;
    MPI_Comm iColComm;

    int iMyHeadCartRank;
    int iMyHeadCartCoords[2];

    int iMySharedCartRank;
    int iMySharedCartCoords[2];

    int iMyHeadRank;
    int iHeadNumProcs;

    int iMySharedRank;
    int iSharedNumProcs;
    
    /* MPI-Variables */
    int iMyRank;   /* current process rank (number) */
    int iNumProcs; /* how many processes */ 

    int iTopRank;
    int iBottomRank;
    int iLeftRank;
    int iRightRank;

    MPI_Win iSharedWinUold;
    MPI_Win iSharedWinUnew;
};

/* jacobi calculation routine */
void Jacobi (struct JacobiData *data);

/* final cleanup routines */
void Finish (struct JacobiData *data);

#endif

