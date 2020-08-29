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
	*pRowLast  = (int) Rows/NBlockRow + Rows%NBlockRow + LastDisp + EndDisp;
    }else{
 	*pRowFirst = (int)(Rank/NBlockCol)     *(int)(Rows/NBlockRow) +Rows%NBlockRow + FirstDisp;
	*pRowLast  = (int)(Rank/NBlockCol + 1) *(int)(Rows/NBlockRow) +Rows%NBlockRow + LastDisp ;
    }
    if(Rank%NBlockCol == 0){
	*pColFirst = InitialDisp;
	*pColLast  = (int) Cols/NBlockCol + Cols%NBlockCol + LastDisp + EndDisp;
    }else{
	*pColFirst = (Rank%NBlockCol)          *(int)(Cols/NBlockCol) + Cols%NBlockCol + FirstDisp;
	*pColLast  = (Rank%NBlockCol + 1)      *(int)(Cols/NBlockCol) + Cols%NBlockCol + LastDisp ;
    }

}

void FirstQueryNeighbours(struct JacobiData *data){

    MPI_Aint winSize;
    int disp_unit = sizeof(double);
    int coords[2];

    coords[0] = data->iMySharedCartCoords[0] +1;
    coords[1] = data->iMySharedCartCoords[1];

    MPI_Cart_rank(data->iSharedCartComm,coords,&(data->iSharedBottomRank));
    MPI_Win_shared_query(data->iSharedWinUold,data->iSharedBottomRank,&winSize,&disp_unit,&(data->afUoldBottom));

    coords[0] = data->iMySharedCartCoords[0] -1;
    coords[1] = data->iMySharedCartCoords[1];
    MPI_Cart_rank(data->iSharedCartComm,coords,&(data->iSharedTopRank));
    MPI_Win_shared_query(data->iSharedWinUold,data->iSharedTopRank,&(data->iWinSizeTop),&disp_unit,&(data->afUoldTop));
    data->iWinSizeTop /= sizeof(double);

    coords[0] = data->iMySharedCartCoords[0];
    coords[1] = data->iMySharedCartCoords[1] +1;
    MPI_Cart_rank(data->iSharedCartComm,coords,&(data->iSharedRightRank));
    MPI_Win_shared_query(data->iSharedWinUold,data->iSharedRightRank,&winSize,&disp_unit,&(data->afUoldRight));
    data->iRowSizeRight = winSize/(sizeof(double)*(data->iSharedRowLast - data->iSharedRowFirst + 1));

    coords[0] = data->iMySharedCartCoords[0];
    coords[1] = data->iMySharedCartCoords[1] -1;
    MPI_Cart_rank(data->iSharedCartComm,coords,&(data->iSharedLeftRank));
    MPI_Win_shared_query(data->iSharedWinUold,data->iSharedLeftRank,&winSize,&disp_unit,&(data->afUoldLeft));
    data->iRowSizeLeft = winSize/(sizeof(double)*(data->iSharedRowLast - data->iSharedRowFirst + 1));

}

void QueryNeighbours(struct JacobiData *data){
    
    MPI_Aint winSize;
    int disp_unit = sizeof(double);
    int coords[2];

    coords[0] = data->iMySharedCartCoords[0] +1;
    coords[1] = data->iMySharedCartCoords[1];

    MPI_Cart_rank(data->iSharedCartComm,coords,&(data->iSharedBottomRank));
    MPI_Win_shared_query(data->iSharedWinUold,data->iSharedBottomRank,&winSize,&disp_unit,&(data->afUoldBottom));

    coords[0] = data->iMySharedCartCoords[0] -1;
    coords[1] = data->iMySharedCartCoords[1];
    MPI_Cart_rank(data->iSharedCartComm,coords,&(data->iSharedTopRank));
    MPI_Win_shared_query(data->iSharedWinUold,data->iSharedTopRank,&winSize,&disp_unit,&(data->afUoldTop));

    coords[0] = data->iMySharedCartCoords[0];
    coords[1] = data->iMySharedCartCoords[1] +1;
    MPI_Cart_rank(data->iSharedCartComm,coords,&(data->iSharedRightRank));
    MPI_Win_shared_query(data->iSharedWinUold,data->iSharedRightRank,&winSize,&disp_unit,&(data->afUoldRight));

    coords[0] = data->iMySharedCartCoords[0];
    coords[1] = data->iMySharedCartCoords[1] -1;
    MPI_Cart_rank(data->iSharedCartComm,coords,&(data->iSharedLeftRank));
    MPI_Win_shared_query(data->iSharedWinUold,data->iSharedLeftRank,&winSize,&disp_unit,&(data->afUoldLeft));

}

void InitSubarraysDatatype(struct JacobiData *data){

    int sizes[2] = {data->iSharedRowLast - data->iSharedRowFirst + 1,data->iSharedColLast - data->iSharedColFirst + 1};
    int subsizes[2] = {data->iSharedRowLast - data->iSharedRowFirst + 1,1};
    int starts[2] = {0,0};

    if(data->iMyHeadCartCoords[1] != 0 && data->iMySharedCartCoords[1] == 0){
        starts[1] = 1;
	MPI_Type_create_subarray(2,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iColLastSend));
        MPI_Type_commit(&(data->iColLastSend));
	
        starts[1] = 0;
	MPI_Type_create_subarray(2,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iColFirstRecv));
        MPI_Type_commit(&(data->iColFirstRecv));
    }
    if(data->iMyHeadCartCoords[1] != data->iHeadNBlockCol - 1  && data->iMySharedCartCoords[1] == data->iSharedNBlockCol - 1){
        starts[1] = data->iSharedColLast - data->iSharedColFirst;
	MPI_Type_create_subarray(2,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iColLastRecv));
        MPI_Type_commit(&(data->iColLastRecv));
        
	starts[1] = data->iSharedColLast - 1 - data->iSharedColFirst;
	MPI_Type_create_subarray(2,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iColFirstSend));
        MPI_Type_commit(&(data->iColFirstSend));
    }
}

void InitIndexes(struct JacobiData *data){
    int i;
    if(data->iMySharedCartCoords[0] != 0){
	data->iIndexTopRank = (int *)malloc((data->iSharedColLast - data->iSharedColFirst + 1) *sizeof(int));
	for(i = data->iSharedColFirst ; i<=data->iSharedColLast ; i++){
		data->iIndexTopRank[i-data->iSharedColFirst] 
		= data->iWinSizeTop - (data->iSharedColLast - data->iSharedColFirst +1) + (i - data->iSharedColFirst);
    	}
    }
    if(data->iMySharedCartCoords[0] != data->iSharedNBlockRow-1){
	data->iIndexBottomRank = (int *)malloc((data->iSharedColLast - data->iSharedColFirst + 1) *sizeof(int));
	for(i = data->iSharedColFirst ; i<= data->iSharedColLast ; i++){
		data->iIndexBottomRank[i-data->iSharedColFirst] = (i - data->iSharedColFirst);
    	}
    }
    if(data->iMySharedCartCoords[1] != 0){
	data->iIndexLeftRank = (int *)malloc((data->iSharedRowLast - data->iSharedRowFirst + 1) *sizeof(int));
	for(i = data->iSharedRowFirst ; i<=data->iSharedRowLast ; i++){
		data->iIndexLeftRank[i-data->iSharedRowFirst] = (i-data->iSharedRowFirst+1)*data->iRowSizeLeft - 1;
    	}
    }
    if(data->iMySharedCartCoords[1] != data->iSharedNBlockCol-1){
	data->iIndexRightRank = (int *)malloc((data->iSharedRowLast - data->iSharedRowFirst + 1) *sizeof(int));
	for(i = data->iSharedRowFirst ; i<=data->iSharedRowLast;i++){
		 data->iIndexRightRank[i-data->iSharedRowFirst] = (i-data->iSharedRowFirst)*data->iRowSizeRight;
    	}
    }
}
