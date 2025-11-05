from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class AnomalyBase(BaseModel):
    anomaly_type: str
    severity: str
    confidence: float
    description: str

class AnomalyCreate(AnomalyBase):
    video_id: int
    frame_number: int
    timestamp: float
    track_ids: Optional[List[int]] = []
    additional_data: Optional[Dict[str, Any]] = {}

class AnomalyResponse(AnomalyBase):
    id: int
    video_id: int
    frame_number: int
    timestamp: float
    track_ids: Optional[List[int]]
    additional_data: Optional[Dict[str, Any]]
    email_sent: bool
    email_sent_time: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True

class AnomalySummary(BaseModel):
    total: int
    by_type: Dict[str, int]
    by_severity: Dict[str, int]
    recent_high_severity: List[Dict[str, Any]]

class AnomalyStatistics(BaseModel):
    total_anomalies: int
    hourly_distribution: Dict[int, int]
    daily_trends: List[Dict[str, Any]]
    most_common_types: List[tuple]