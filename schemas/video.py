from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class VideoBase(BaseModel):
    filename: str
    original_filename: str

class VideoCreate(VideoBase):
    pass

class VideoResponse(VideoBase):
    id: int
    status: str
    duration: Optional[float]
    fps: Optional[float]
    width: Optional[int]
    height: Optional[int]
    upload_time: datetime
    process_start_time: Optional[datetime]
    process_end_time: Optional[datetime]
    error_message: Optional[str]
    
    class Config:
        from_attributes = True

class DetectionBase(BaseModel):
    frame_number: int
    timestamp: float
    object_class: str
    confidence: float
    color: Optional[str]
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int

class DetectionResponse(DetectionBase):
    id: int
    video_id: int
    
    class Config:
        from_attributes = True

class SummaryResponse(BaseModel):
    id: int
    video_id: int
    total_luggage: int
    total_persons: int
    abandoned_count: int
    owner_identified_count: int
    luggage_types: Any
    color_distribution: Any
    timeline_data: Any
    processed_frames: int

    class Config:
        from_attributes = True

class QueryRequest(BaseModel):
    question: str
    video_id: Optional[int] = None

class QueryResponse(BaseModel):
    answer: str
    supporting_data: Optional[Dict[str, Any]]
    relevant_detections: Optional[List[DetectionResponse]]
