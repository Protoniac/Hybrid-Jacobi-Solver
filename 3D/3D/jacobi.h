#ifndef _JACOBI_H
#define _JACOBI_H

struct JacobiData
{
    /* input data */
    int iHeights;
    int iRows;
    int iCols;
    int iNBlockHeight;
    int iNBlockRow;
    int iNBlockCol;
    
    int iHeightFirst;
    int iHeightLast;
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
    double fDz;
    /* pointers to the allocated memory */
    double *afU;
    double *afF;

    MPI_Datatype iHeightSquareFirstSend; 
    MPI_Datatype iHeightSquareFirstRecv; 
    MPI_Datatype iHeightSquareLastSend; 
    MPI_Datatype iHeightSquareLastRecv; 

    MPI_Datatype iRowSquareFirstSend; 
    MPI_Datatype iRowSquareFirstRecv; 
    MPI_Datatype iRowSquareLastSend; 
    MPI_Datatype iRowSquareLastRecv; 

    MPI_Datatype iColSquareFirstSend; 
    MPI_Datatype iColSquareFirstRecv; 
    MPI_Datatype iColSquareLastSend; 
    MPI_Datatype iColSquareLastRecv; 

    /* start and end timestamps */
    double fTimeStart;
    double fTimeStop;

    /* calculated residual (output jacobi) */
    double fResidual;
    /* effective interation count (output jacobi) */
    int iIterCount;

    /* calculated error (output error_check) */
    double fError;
    
    MPI_Comm iCartComm;
    int iMyCartCoords[3];
    int iMyCartRank;

    MPI_Comm iTempHeightComm;
    MPI_Comm iTempRowComm;
    MPI_Comm iTempColComm;

    MPI_Comm iHeightComm;
    MPI_Comm iRowComm;
    MPI_Comm iColComm;

    int iMyRank;   /* current process rank (number) */
    int iNumProcs; /* how many processes */ 

    int iBackRank;
    int iFrontRank;
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

