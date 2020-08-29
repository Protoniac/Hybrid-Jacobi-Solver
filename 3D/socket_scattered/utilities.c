#include<mpi.h>
#include "utilities.h"

void BlockInit(int rows, int cols, int number_of_block, int *pNBlockHeight, int *pNBlockRow, int *pNBlockCol){
    	//This can be improved to obtain better domain scattering
	*pNBlockHeight = (int) cbrt(number_of_block);
	while(number_of_block / (1.0*(*pNBlockHeight)) != (int) number_of_block / (*pNBlockHeight)) *pNBlockHeight -= 1;
	int plane_number_of_block = number_of_block / (*pNBlockHeight);
	if (rows > cols) {
	*pNBlockCol = (int) sqrt(plane_number_of_block);	
	*pNBlockRow = (int) (plane_number_of_block)/(*pNBlockCol);
	while( *pNBlockRow != (double) (plane_number_of_block)/(*pNBlockCol)){
		*pNBlockCol -= 1;
		*pNBlockRow = (int) (plane_number_of_block)/(*pNBlockCol);
	}
    } else {
	*pNBlockRow = (int) sqrt(plane_number_of_block);
	*pNBlockCol = (int) (plane_number_of_block)/(*pNBlockRow);
	while( *pNBlockCol != (double) (plane_number_of_block)/(*pNBlockRow)){
		*pNBlockRow -= 1;
		*pNBlockCol = (int) (plane_number_of_block)/(*pNBlockRow);
	}
    }
}



void AreaInit(int Coords[3], int NBlockHeight, int NBlockRow, int NBlockCol,int Heights, int Rows, int Cols, 
	      int InitialDisp, int EndDisp, int FirstDisp, int LastDisp,
              int *pHeightFirst, int *pHeightLast, int *pRowFirst, int *pRowLast, int *pColFirst, int *pColLast){

    if(Coords[0] == 0){
	*pHeightFirst = InitialDisp;
	*pHeightLast  = (int) Heights/NBlockHeight + Heights%NBlockHeight + EndDisp;
    }else{
 	*pHeightFirst = (Coords[0])     *(int)(Heights/NBlockHeight) +Heights%NBlockHeight + FirstDisp;
	*pHeightLast  = (Coords[0] + 1) *(int)(Heights/NBlockHeight) +Heights%NBlockHeight + LastDisp ;
    }
    if(Coords[1] == 0){
	*pRowFirst = InitialDisp;
	*pRowLast  = (int) Rows/NBlockRow + Rows%NBlockRow + EndDisp;
    }else{
 	*pRowFirst = (Coords[1])     *(int)(Rows/NBlockRow) +Rows%NBlockRow + FirstDisp;
	*pRowLast  = (Coords[1] + 1) *(int)(Rows/NBlockRow) +Rows%NBlockRow + LastDisp ;
    }
    if(Coords[2] == 0){
	*pColFirst = InitialDisp;
	*pColLast  = (int) Cols/NBlockCol + Cols%NBlockCol + EndDisp;
    }else{
	*pColFirst = (Coords[2])          *(int)(Cols/NBlockCol) + Cols%NBlockCol + FirstDisp;
	*pColLast  = (Coords[2] + 1)      *(int)(Cols/NBlockCol) + Cols%NBlockCol + LastDisp ;
    }

}

void InitNeighbours(struct JacobiData *data){

    MPI_Comm_rank(data->iHeightComm,&(data->iBackRank));
    data->iFrontRank = data->iBackRank;
    data->iBackRank -= 1;
    data->iFrontRank += 1;

    MPI_Comm_rank(data->iRowComm,&(data->iLeftRank));
    data->iRightRank = data->iLeftRank;
    data->iLeftRank -= 1;
    data->iRightRank += 1;

    MPI_Comm_rank(data->iColComm,&(data->iTopRank));
    data->iBottomRank = data->iTopRank;
    data->iTopRank -= 1;
    data->iBottomRank += 1;
}

void InitSubarraysDatatype(struct JacobiData *data){

    int sizes[3] = {data->iHeightLast - data->iHeightFirst + 1,data->iRowLast - data->iRowFirst + 1,data->iColLast - data->iColFirst + 1};
    int subsizes[3] = {data->iSharedHeightLast - data->iSharedHeightFirst + 1,1,data->iSharedColLast - data->iSharedColFirst + 1};
    int starts[3] = {data->iSharedHeightFirst - data->iHeightFirst,0,data->iSharedColFirst - data->iColFirst};

    if(data->iMyHeadCartCoords[1] != 0 && data->iMySharedCartCoords[1] == 0){
        starts[1] = data->iSharedRowFirst + 1 - data->iRowFirst;
	MPI_Type_create_subarray(3,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iRowSquareLastSend));
        MPI_Type_commit(&(data->iRowSquareLastSend));
	//data->iColLastSend  = (double *)malloc((data->iSharedRowLast - data->iSharedRowFirst + 1)*sizeof(double));
	
        starts[1] = data->iSharedRowFirst - data->iRowFirst;
	MPI_Type_create_subarray(3,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iRowSquareFirstRecv));
        MPI_Type_commit(&(data->iRowSquareFirstRecv));
    	//data->iColFirstRecv = (double *)malloc((data->iSharedRowLast - data->iSharedRowFirst + 1)*sizeof(double));
    }
    if(data->iMyHeadCartCoords[1] != data->iHeadNBlockRow - 1  && data->iMySharedCartCoords[1] == data->iSharedNBlockRow - 1){
        starts[1] = data->iSharedRowLast - data->iRowFirst;
	MPI_Type_create_subarray(3,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iRowSquareLastRecv));
        MPI_Type_commit(&(data->iRowSquareLastRecv));
	//data->iColLastRecv  = (double *)malloc((data->iSharedRowLast - data->iSharedRowFirst + 1)*sizeof(double));
        
	starts[1] = data->iSharedRowLast - 1 - data->iRowFirst;
	MPI_Type_create_subarray(3,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iRowSquareFirstSend));
        MPI_Type_commit(&(data->iRowSquareFirstSend));
    	//data->iColFirstSend = (double *)malloc((data->iSharedRowLast - data->iSharedRowFirst + 1)*sizeof(double));
    }

    subsizes[0] = 1;
    subsizes[1] = data->iSharedRowLast - data->iSharedRowFirst + 1;
    starts  [1] = data->iSharedRowFirst - data->iRowFirst;

    if(data->iMyHeadCartCoords[0] != 0 && data->iMySharedCartCoords[0] == 0){
        starts[0] = data->iSharedHeightFirst + 1 - data->iHeightFirst;
	MPI_Type_create_subarray(3,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iHeightSquareLastSend));
        MPI_Type_commit(&(data->iHeightSquareLastSend));
	//data->iColLastSend  = (double *)malloc((data->iSharedRowLast - data->iSharedRowFirst + 1)*sizeof(double));
	
        starts[0] = data->iSharedHeightFirst - data->iHeightFirst;
	MPI_Type_create_subarray(3,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iHeightSquareFirstRecv));
        MPI_Type_commit(&(data->iHeightSquareFirstRecv));
    	//data->iColFirstRecv = (double *)malloc((data->iSharedRowLast - data->iSharedRowFirst + 1)*sizeof(double));
    }
    if(data->iMyHeadCartCoords[0] != data->iHeadNBlockHeight - 1  && data->iMySharedCartCoords[0] == data->iSharedNBlockHeight - 1){
        starts[0] = data->iSharedHeightLast - data->iHeightFirst;
	MPI_Type_create_subarray(3,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iHeightSquareLastRecv));
        MPI_Type_commit(&(data->iHeightSquareLastRecv));
	//data->iColLastRecv  = (double *)malloc((data->iSharedRowLast - data->iSharedRowFirst + 1)*sizeof(double));
        
	starts[0] = data->iSharedHeightLast - 1 - data->iHeightFirst;
	MPI_Type_create_subarray(3,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iHeightSquareFirstSend));
	MPI_Type_commit(&(data->iHeightSquareFirstSend));
    }
    subsizes[2] = 1;
    subsizes[0] = data->iSharedHeightLast - data->iSharedHeightFirst + 1;
    starts  [0] = data->iSharedHeightFirst - data->iHeightFirst;

    if(data->iMyHeadCartCoords[2] != 0 && data->iMySharedCartCoords[2] == 0){
        starts[2] = data->iSharedColFirst + 1 - data->iColFirst;
	MPI_Type_create_subarray(3,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iColSquareLastSend));
        MPI_Type_commit(&(data->iColSquareLastSend));
	//data->iColLastSend  = (double *)malloc((data->iSharedRowLast - data->iSharedRowFirst + 1)*sizeof(double));
	
        starts[2] = data->iSharedColFirst - data->iColFirst;
	MPI_Type_create_subarray(3,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iColSquareFirstRecv));
        MPI_Type_commit(&(data->iColSquareFirstRecv));
    	//data->iColFirstRecv = (double *)malloc((data->iSharedRowLast - data->iSharedRowFirst + 1)*sizeof(double));
    }
    if(data->iMyHeadCartCoords[2] != data->iHeadNBlockCol - 1  && data->iMySharedCartCoords[2] == data->iSharedNBlockCol - 1){
        starts[2] = data->iSharedColLast - data->iColFirst;
	MPI_Type_create_subarray(3,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iColSquareLastRecv));
        MPI_Type_commit(&(data->iColSquareLastRecv));
	//data->iColLastRecv  = (double *)malloc((data->iSharedRowLast - data->iSharedRowFirst + 1)*sizeof(double));
        
	starts[2] = data->iSharedColLast - 1 - data->iColFirst;
	MPI_Type_create_subarray(3,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iColSquareFirstSend));
	MPI_Type_commit(&(data->iColSquareFirstSend));
    }
	
}
