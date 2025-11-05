from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from models.database import get_db
from schemas.video import QueryRequest, QueryResponse
from services.query_service import QueryService

router = APIRouter()
query_service = QueryService()

@router.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    """Process a natural language query about video analysis results."""
    
    try:
        result = await query_service.process_query(
            question=request.question,
            video_id=request.video_id,
            db=db
        )
        
        return QueryResponse(
            answer=result["answer"],
            supporting_data=result["supporting_data"],
            relevant_detections=result["relevant_detections"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))