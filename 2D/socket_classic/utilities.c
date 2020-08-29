#include<mpi.h>
#include "utilities.h"

void BlockInit(int rows, int cols, int number_of_block, int *pNBlockRow, int *pNBlockCol){
    if (rows > cols) {
	*pNBlockCol = (int) sqrt(number_of_block);	
	*pNBlockRow = (int) (number_of_block)/(*pNBlockCol);
	while( *pNBlockRow != (double) (number_of_block)/(*pNBlockCol) ){
		*pNBlockCol -= 1;
		*pNBlockRow = (int) (number_of_block)/(*pNBlockCol);
	}
    } else {
	*pNBlockRow = (int) sqrt(number_of_block);
	*pNBlockCol = (int) (number_of_block)/(*pNBlockRow);
	while( *pNBlockCol != (double) (number_of_block)/(*pNBlockRow)){
		*pNBlockRow -= 1;
		*pNBlockCol = (int) (number_of_block)/(*pNBlockRow);
	}
    }
}


void AreaInit(int Rank, int NBlockRow, int NBlockCol, int Rows, int Cols, int InitialDisp, int EndDisp, int FirstDisp, int LastDisp,
              int *pRowFirst, int *pRowLast, int *pColFirst, int *pColLast){

    if(Rank < NBlockCol){
	*pRowFirst = InitialDisp;
	*pRowLast  = (int) Rows/NBlockRow + Rows%NBlockRow + EndDisp;
    }else{
 	*pRowFirst = (int)(Rank/NBlockCol)     *(int)(Rows/NBlockRow) +Rows%NBlockRow + FirstDisp;
	*pRowLast  = (int)(Rank/NBlockCol + 1) *(int)(Rows/NBlockRow) +Rows%NBlockRow + LastDisp ;
    }
    if(Rank%NBlockCol == 0){
	*pColFirst = InitialDisp;
	*pColLast  = (int) Cols/NBlockCol + Cols%NBlockCol + EndDisp;
    }else{
	*pColFirst = (Rank%NBlockCol)          *(int)(Cols/NBlockCol) + Cols%NBlockCol + FirstDisp;
	*pColLast  = (Rank%NBlockCol + 1)      *(int)(Cols/NBlockCol) + Cols%NBlockCol + LastDisp ;
    }

}

void InitSubarraysDatatype(struct JacobiData *data){
    int sizes[2] = {data->iRowLast - data->iRowFirst + 1,data->iColLast - data->iColFirst + 1};
    int subsizes[2] = {data->iRowLast - data->iRowFirst + 1,1};
    int starts[2] = {0,0};

    starts[1] = 1;
    MPI_Type_create_subarray(2,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iColLastSend));
    MPI_Type_commit(&(data->iColLastSend));
	
    starts[1] = 0;
    MPI_Type_create_subarray(2,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iColFirstRecv));
    MPI_Type_commit(&(data->iColFirstRecv));

    starts[1] = data->iColLast - data->iColFirst;
    MPI_Type_create_subarray(2,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iColLastRecv));
    MPI_Type_commit(&(data->iColLastRecv));

    starts[1] = data->iColLast - 1 - data->iColFirst;
    MPI_Type_create_subarray(2,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iColFirstSend));
    MPI_Type_commit(&(data->iColFirstSend));
}
