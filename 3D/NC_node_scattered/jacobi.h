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

    int iHeadNBlockHeight;
    int iHeadNBlockRow;
    int iHeadNBlockCol;
    
    int iSharedNBlockHeight;
    int iSharedNBlockRow;
    int iSharedNBlockCol;

    int iSharedHeightFirst;
    int iSharedHeightLast;
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
    double fDz;
    /* pointers to the allocated memory */
    double *afU;
    double *afUold;
    double *afF;
    
    double *afUoldBack;
    double *afUoldFront;
    double *afUoldTop;
    double *afUoldBottom;
    double *afUoldLeft;
    double *afUoldRight;
    
    MPI_Aint iWinSizeBack;
    int iRowSizeLeft;
    int iRowSizeRight;
    int iColSizeTop;
    int iColSizeBottom;  
  
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

    MPI_Comm iHeadComm;
    MPI_Comm iSharedComm;
    MPI_Comm iHeadCartComm;
    MPI_Comm iSharedCartComm;
    
    MPI_Comm iWorldCartComm;
    int iMyWorldCartCoords[3];
    int iMyWorldCartRank;

    MPI_Comm iHeadHeightComm;
    MPI_Comm iHeadRowComm;
    MPI_Comm iHeadColComm;

    MPI_Comm iHeightComm;
    MPI_Comm iRowComm;
    MPI_Comm iColComm;

    int iMyHeadCartRank;
    int iMyHeadCartCoords[3];

    int iMySharedCartRank;
    int iMySharedCartCoords[3];

    int iMyHeadRank;
    int iHeadNumProcs;

    int iMySharedRank;
    int iSharedNumProcs;
    
    /* MPI-Variables */
    int iMyRank;   /* current process rank (number) */
    int iNumProcs; /* how many processes */ 

    int iBackRank;
    int iFrontRank;
    int iTopRank;
    int iBottomRank;
    int iLeftRank;
    int iRightRank;

    int iSharedFrontRank;
    int iSharedBackRank;
    int iSharedTopRank;
    int iSharedBottomRank;
    int iSharedLeftRank;
    int iSharedRightRank;

    int **iIndexBackRank;
    int **iIndexFrontRank;
    int **iIndexLeftRank;
    int **iIndexRightRank;
    int **iIndexBottomRank;
    int **iIndexTopRank;

    MPI_Win iSharedWinUold;
    MPI_Win iSharedWinUnew;
};

/* jacobi calculation routine */
void Jacobi (struct JacobiData *data);

/* final cleanup routines */
void Finish (struct JacobiData *data);

#endif

