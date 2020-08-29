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
	*pRowLast  = (int) Rows/NBlockRow + Rows%NBlockRow +  EndDisp;
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

void InitSubarraysDatatype(struct JacobiData *data){

    int sizes[3] = {data->iSharedHeightLast - data->iSharedHeightFirst + 1,
		    data->iSharedRowLast - data->iSharedRowFirst + 1,
		    data->iSharedColLast - data->iSharedColFirst + 1};

    int subsizes[3] = {data->iSharedHeightLast - data->iSharedHeightFirst + 1,1,data->iSharedColLast - data->iSharedColFirst + 1};
    int starts[3] = {0,0,0};

    if(data->iMyHeadCartCoords[1] != 0 && data->iMySharedCartCoords[1] == 0){
        starts[1] = 1;
	MPI_Type_create_subarray(3,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iRowSquareLastSend));
        MPI_Type_commit(&(data->iRowSquareLastSend));
	
        starts[1] = 0;
	MPI_Type_create_subarray(3,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iRowSquareFirstRecv));
        MPI_Type_commit(&(data->iRowSquareFirstRecv));
    }
    if(data->iMyHeadCartCoords[1] != data->iHeadNBlockRow - 1  && data->iMySharedCartCoords[1] == data->iSharedNBlockRow - 1){
        starts[1] = data->iSharedRowLast-data->iSharedRowFirst;
	MPI_Type_create_subarray(3,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iRowSquareLastRecv));
        MPI_Type_commit(&(data->iRowSquareLastRecv));
        
	starts[1] = data->iSharedRowLast - 1-data->iSharedRowFirst;
	MPI_Type_create_subarray(3,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iRowSquareFirstSend));
        MPI_Type_commit(&(data->iRowSquareFirstSend));
    }
    subsizes[0] = 1;
    subsizes[1] = data->iSharedRowLast - data->iSharedRowFirst + 1;
    starts  [1] = 0;

    if(data->iMyHeadCartCoords[0] != 0 && data->iMySharedCartCoords[0] == 0){
        starts[0] = 1;
	MPI_Type_create_subarray(3,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iHeightSquareLastSend));
        MPI_Type_commit(&(data->iHeightSquareLastSend));
	
        starts[0] = 0;
	MPI_Type_create_subarray(3,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iHeightSquareFirstRecv));
        MPI_Type_commit(&(data->iHeightSquareFirstRecv));
    }
    if(data->iMyHeadCartCoords[0] != data->iHeadNBlockHeight - 1  && data->iMySharedCartCoords[0] == data->iSharedNBlockHeight - 1){
        starts[0] = data->iSharedHeightLast - data->iSharedHeightFirst;
	MPI_Type_create_subarray(3,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iHeightSquareLastRecv));
        MPI_Type_commit(&(data->iHeightSquareLastRecv));
        
	starts[0] = data->iSharedHeightLast - 1 - data->iSharedHeightFirst;
	MPI_Type_create_subarray(3,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iHeightSquareFirstSend));
        MPI_Type_commit(&(data->iHeightSquareFirstSend));
    }
    subsizes[0] = data->iSharedHeightLast - data->iSharedHeightFirst + 1;
    subsizes[2] = 1;
    starts  [0] = 0;

    if(data->iMyHeadCartCoords[2] != 0 && data->iMySharedCartCoords[2] == 0){
        starts[2] = 1;
	MPI_Type_create_subarray(3,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iColSquareLastSend));
        MPI_Type_commit(&(data->iColSquareLastSend));
	
        starts[2] = 0;
	MPI_Type_create_subarray(3,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iColSquareFirstRecv));
        MPI_Type_commit(&(data->iColSquareFirstRecv));
    }
    if(data->iMyHeadCartCoords[2] != data->iHeadNBlockCol - 1  && data->iMySharedCartCoords[2] == data->iSharedNBlockCol - 1){
        starts[2] = data->iSharedColLast - data->iSharedColFirst;
	MPI_Type_create_subarray(3,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iColSquareLastRecv));
        MPI_Type_commit(&(data->iColSquareLastRecv));
        
	starts[2] = data->iSharedColLast - 1 - data->iSharedColFirst;
	MPI_Type_create_subarray(3,sizes,subsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&(data->iColSquareFirstSend));
        MPI_Type_commit(&(data->iColSquareFirstSend));
    }
}

void InitIndexes(struct JacobiData *data){

    int memory_shift;
    int i,j,k;
    if(data->iMySharedCartCoords[0] != 0){
	memory_shift = data->iWinSizeBack - (data->iSharedColLast - data->iSharedColFirst + 1) * (data->iSharedRowLast - data->iSharedRowFirst + 1);
        data->iIndexBackRank = (int **)malloc((data->iSharedRowLast - data->iSharedRowFirst + 1)*sizeof(int *));

	for(j = data->iSharedRowFirst; j <= data->iSharedRowLast; j++){
		data->iIndexBackRank[j-data->iSharedRowFirst] = (int *)malloc((data->iSharedColLast - data->iSharedColFirst + 1) * sizeof(int));
		for(i = data->iSharedColFirst; i <= data->iSharedColLast; i++){
			data->iIndexBackRank[j-data->iSharedRowFirst][i-data->iSharedColFirst] 
			= memory_shift + (j-data->iSharedRowFirst)*(data->iSharedColLast - data->iSharedColFirst +1) 
		        + (i-data->iSharedColFirst);
		}
	}
    }
    if(data->iMySharedCartCoords[0] != data->iSharedNBlockHeight - 1){
        data->iIndexFrontRank = (int **)malloc((data->iSharedRowLast - data->iSharedRowFirst + 1)*sizeof(int *));

	for(j = data->iSharedRowFirst; j <= data->iSharedRowLast; j++){
		data->iIndexFrontRank[j-data->iSharedRowFirst] = (int *)malloc((data->iSharedColLast - data->iSharedColFirst + 1) * sizeof(int));
		for(i = data->iSharedColFirst; i <= data->iSharedColLast; i++){
			data->iIndexFrontRank[j-data->iSharedRowFirst][i-data->iSharedColFirst] 
			= (j-data->iSharedRowFirst)*(data->iSharedColLast - data->iSharedColFirst +1)
			+ (i-data->iSharedColFirst);
		}
	}
    }
    if(data->iMySharedCartCoords[1] != 0){
	memory_shift = (data->iSharedColLast - data->iSharedColFirst + 1);
        data->iIndexLeftRank = (int **)malloc((data->iSharedHeightLast - data->iSharedHeightFirst + 1)*sizeof(int *));

	for(k = data->iSharedHeightFirst; k <= data->iSharedHeightLast; k++){
		data->iIndexLeftRank[k-data->iSharedHeightFirst] = (int *)malloc((data->iSharedColLast - data->iSharedColFirst + 1) * sizeof(int));
		for(i = data->iSharedColFirst; i <= data->iSharedColLast; i++){
			data->iIndexLeftRank[k-data->iSharedHeightFirst][i-data->iSharedColFirst] 
			= (k-data->iSharedHeightFirst+1)*data->iRowSizeLeft*(data->iSharedColLast -data->iSharedColFirst+1)               				      - memory_shift + (i-data->iSharedColFirst);
		}
	}
    }
    if(data->iMySharedCartCoords[1] != data->iSharedNBlockRow - 1){
        data->iIndexRightRank = (int **)malloc((data->iSharedHeightLast - data->iSharedHeightFirst + 1)*sizeof(int *));

	for(k = data->iSharedHeightFirst; k <= data->iSharedHeightLast; k++){
		data->iIndexRightRank[k-data->iSharedHeightFirst] = (int *)malloc((data->iSharedColLast - data->iSharedColFirst + 1) * sizeof(int));
		for(i = data->iSharedColFirst; i <= data->iSharedColLast; i++){
			data->iIndexRightRank[k-data->iSharedHeightFirst][i-data->iSharedColFirst] 
			= (k-data->iSharedHeightFirst)*data->iRowSizeRight*(data->iSharedColLast -data->iSharedColFirst+1) 			                              + (i-data->iSharedColFirst);
		}
	}
    }
    if(data->iMySharedCartCoords[2] != 0){
	memory_shift = data->iColSizeTop-1; 
        data->iIndexTopRank = (int **)malloc((data->iSharedHeightLast - data->iSharedHeightFirst + 1)*sizeof(int *));

	for(k = data->iSharedHeightFirst; k <= data->iSharedHeightLast; k++){
		data->iIndexTopRank[k-data->iSharedHeightFirst] = (int *)malloc((data->iSharedRowLast - data->iSharedRowFirst + 1) * sizeof(int));
		for(j = data->iSharedRowFirst; j <= data->iSharedRowLast; j++){
			data->iIndexTopRank[k-data->iSharedHeightFirst][j-data->iSharedRowFirst] 
			= (k-data->iSharedHeightFirst)*data->iColSizeTop*(data->iSharedRowLast - data->iSharedRowFirst+1)			     		              + memory_shift + data->iColSizeTop*(j-data->iSharedRowFirst);
		}
	}
    }
    if(data->iMySharedCartCoords[2] != data->iSharedNBlockCol - 1){
        data->iIndexBottomRank = (int **)malloc((data->iSharedHeightLast - data->iSharedHeightFirst + 1)*sizeof(int *));

	for(k = data->iSharedHeightFirst; k <= data->iSharedHeightLast; k++){
		data->iIndexBottomRank[k-data->iSharedHeightFirst] = (int *)malloc((data->iSharedRowLast - data->iSharedRowFirst + 1) * sizeof(int));
		for(j = data->iSharedRowFirst; j <= data->iSharedRowLast; j++){
			data->iIndexBottomRank[k-data->iSharedHeightFirst][j-data->iSharedRowFirst] 
			= (k-data->iSharedHeightFirst)*data->iColSizeBottom*(data->iSharedRowLast - data->iSharedRowFirst+1) 
			+ (j-data->iSharedRowFirst)*data->iColSizeBottom;
		}
	}
    }
}

void CleanIndexes(struct JacobiData *data){
    int j,k;
    if(data->iMySharedCartCoords[0] != 0){
	for(j = data->iSharedRowFirst; j <= data->iSharedRowLast; j++){
		free(data->iIndexBackRank[j-data->iSharedRowFirst]);
	}
	free(data->iIndexBackRank);
    }
    if(data->iMySharedCartCoords[0] != data->iSharedNBlockHeight - 1){
	for(j = data->iSharedRowFirst; j <= data->iSharedRowLast; j++){
		free(data->iIndexFrontRank[j-data->iSharedRowFirst]);
	}
	free(data->iIndexFrontRank);
    }
    if(data->iMySharedCartCoords[1] != 0){
	for(k = data->iSharedHeightFirst; k <= data->iSharedHeightLast; k++){
		free(data->iIndexLeftRank[k-data->iSharedHeightFirst]);
	}
	free(data->iIndexLeftRank);
    }
    if(data->iMySharedCartCoords[1] != data->iSharedNBlockRow - 1){
	for(k = data->iSharedHeightFirst; k <= data->iSharedHeightLast; k++){
		free(data->iIndexRightRank[k-data->iSharedHeightFirst]);
	}
	free(data->iIndexRightRank);
    }
    if(data->iMySharedCartCoords[2] != 0){
	for(k = data->iSharedHeightFirst; k <= data->iSharedHeightLast; k++){
		free(data->iIndexTopRank[k-data->iSharedHeightFirst]);
	}
	free(data->iIndexTopRank);
    }
    if(data->iMySharedCartCoords[2] != data->iSharedNBlockCol - 1){
	for(k = data->iSharedHeightFirst; k <= data->iSharedHeightLast; k++){
		free(data->iIndexBottomRank[k-data->iSharedHeightFirst]);
	}
	free(data->iIndexBottomRank);
    }
}

void FirstQueryNeighbours(struct JacobiData *data){

    int disp_unit = sizeof(double);
    int coords[3];
    MPI_Aint winSize;
	
    coords[0] = data->iMySharedCartCoords[0] +1;
    coords[1] = data->iMySharedCartCoords[1];
    coords[2] = data->iMySharedCartCoords[2];
    MPI_Cart_rank(data->iSharedCartComm,coords,&(data->iSharedFrontRank));
    MPI_Win_shared_query(data->iSharedWinUold,data->iSharedFrontRank,&winSize,&disp_unit,&(data->afUoldFront));

    coords[0] = data->iMySharedCartCoords[0] -1;
    coords[1] = data->iMySharedCartCoords[1];
    coords[2] = data->iMySharedCartCoords[2];
    MPI_Cart_rank(data->iSharedCartComm,coords,&(data->iSharedBackRank));
    MPI_Win_shared_query(data->iSharedWinUold,data->iSharedBackRank,&(data->iWinSizeBack),&disp_unit,&(data->afUoldBack));
    data->iWinSizeBack /= sizeof(double);

    coords[0] = data->iMySharedCartCoords[0];
    coords[1] = data->iMySharedCartCoords[1] +1;
    coords[2] = data->iMySharedCartCoords[2];
    MPI_Cart_rank(data->iSharedCartComm,coords,&(data->iSharedRightRank));
    MPI_Win_shared_query(data->iSharedWinUold,data->iSharedRightRank,&winSize,&disp_unit,&(data->afUoldRight));
    data->iRowSizeRight = winSize/(sizeof(double)*(data->iSharedColLast - data->iSharedColFirst + 1)
			           *(data->iSharedHeightLast-data->iSharedHeightFirst+1));

    coords[0] = data->iMySharedCartCoords[0];
    coords[1] = data->iMySharedCartCoords[1] -1;
    coords[2] = data->iMySharedCartCoords[2];
    MPI_Cart_rank(data->iSharedCartComm,coords,&(data->iSharedLeftRank));
    MPI_Win_shared_query(data->iSharedWinUold,data->iSharedLeftRank,&winSize,&disp_unit,&(data->afUoldLeft));
    data->iRowSizeLeft = winSize/(sizeof(double)*(data->iSharedColLast - data->iSharedColFirst + 1)
				  *(data->iSharedHeightLast-data->iSharedHeightFirst+1));
    
    coords[0] = data->iMySharedCartCoords[0];
    coords[1] = data->iMySharedCartCoords[1];
    coords[2] = data->iMySharedCartCoords[2] +1;
    MPI_Cart_rank(data->iSharedCartComm,coords,&(data->iSharedBottomRank));
    MPI_Win_shared_query(data->iSharedWinUold,data->iSharedBottomRank,&winSize,&disp_unit,&(data->afUoldBottom));
    data->iColSizeBottom = winSize/(sizeof(double)*(data->iSharedRowLast - data->iSharedRowFirst + 1)
			            *(data->iSharedHeightLast-data->iSharedHeightFirst+1));

    coords[0] = data->iMySharedCartCoords[0];
    coords[1] = data->iMySharedCartCoords[1];
    coords[2] = data->iMySharedCartCoords[2] -1;
    MPI_Cart_rank(data->iSharedCartComm,coords,&(data->iSharedTopRank));
    MPI_Win_shared_query(data->iSharedWinUold,data->iSharedTopRank,&winSize,&disp_unit,&(data->afUoldTop));
    data->iColSizeTop = winSize/(sizeof(double)*(data->iSharedRowLast - data->iSharedRowFirst + 1)
			         *(data->iSharedHeightLast-data->iSharedHeightFirst+1));
}

void QueryNeighbours(struct JacobiData *data){
    int disp_unit = sizeof(double);
    int coords[3];
    MPI_Aint winSize;

    coords[0] = data->iMySharedCartCoords[0] +1;
    coords[1] = data->iMySharedCartCoords[1];
    coords[2] = data->iMySharedCartCoords[2];
    MPI_Cart_rank(data->iSharedCartComm,coords,&(data->iSharedFrontRank));
    MPI_Win_shared_query(data->iSharedWinUold,data->iSharedFrontRank,&winSize,&disp_unit,&(data->afUoldFront));

    coords[0] = data->iMySharedCartCoords[0] -1;
    coords[1] = data->iMySharedCartCoords[1];
    coords[2] = data->iMySharedCartCoords[2];
    MPI_Cart_rank(data->iSharedCartComm,coords,&(data->iSharedBackRank));
    MPI_Win_shared_query(data->iSharedWinUold,data->iSharedBackRank,&winSize,&disp_unit,&(data->afUoldBack));

    coords[0] = data->iMySharedCartCoords[0];
    coords[1] = data->iMySharedCartCoords[1] +1;
    coords[2] = data->iMySharedCartCoords[2];
    MPI_Cart_rank(data->iSharedCartComm,coords,&(data->iSharedRightRank));
    MPI_Win_shared_query(data->iSharedWinUold,data->iSharedRightRank,&winSize,&disp_unit,&(data->afUoldRight));

    coords[0] = data->iMySharedCartCoords[0];
    coords[1] = data->iMySharedCartCoords[1] -1;
    coords[2] = data->iMySharedCartCoords[2];
    MPI_Cart_rank(data->iSharedCartComm,coords,&(data->iSharedLeftRank));
    MPI_Win_shared_query(data->iSharedWinUold,data->iSharedLeftRank,&winSize,&disp_unit,&(data->afUoldLeft));
    
    coords[0] = data->iMySharedCartCoords[0];
    coords[1] = data->iMySharedCartCoords[1];
    coords[2] = data->iMySharedCartCoords[2] +1;
    MPI_Cart_rank(data->iSharedCartComm,coords,&(data->iSharedBottomRank));
    MPI_Win_shared_query(data->iSharedWinUold,data->iSharedBottomRank,&winSize,&disp_unit,&(data->afUoldBottom));

    coords[0] = data->iMySharedCartCoords[0];
    coords[1] = data->iMySharedCartCoords[1];
    coords[2] = data->iMySharedCartCoords[2] -1;
    MPI_Cart_rank(data->iSharedCartComm,coords,&(data->iSharedTopRank));
    MPI_Win_shared_query(data->iSharedWinUold,data->iSharedTopRank,&winSize,&disp_unit,&(data->afUoldTop));
}
