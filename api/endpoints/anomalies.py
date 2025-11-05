from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta

from models.database import get_db, Anomaly, Video
from schemas.anomaly import AnomalyResponse, AnomalySummary, AnomalyStatistics
from config import settings

router = APIRouter()

@router.get("/videos/{video_id}/anomalies", response_model=List[AnomalyResponse])
async def get_video_anomalies(
    video_id: int,
    anomaly_type: Optional[str] = None,
    severity: Optional[str] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get anomalies for a specific video."""
    
    query = db.query(Anomaly).filter(Anomaly.video_id == video_id)
    
    if anomaly_type:
        query = query.filter(Anomaly.anomaly_type == anomaly_type)
    
    if severity:
        query = query.filter(Anomaly.severity == severity)
    
    if start_time is not None:
        query = query.filter(Anomaly.timestamp >= start_time)
    
    if end_time is not None:
        query = query.filter(Anomaly.timestamp <= end_time)
    
    anomalies = query.order_by(Anomaly.timestamp).offset(skip).limit(limit).all()
    
    return anomalies

@router.get("/anomalies", response_model=List[AnomalyResponse])
async def get_all_anomalies(
    severity: Optional[str] = None,
    anomaly_type: Optional[str] = None,
    hours: Optional[int] = Query(24, description="Get anomalies from last N hours"),
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all anomalies across all videos."""
    
    query = db.query(Anomaly)
    
    # Filter by time
    if hours:
        since = datetime.utcnow() - timedelta(hours=hours)
        query = query.filter(Anomaly.created_at >= since)
    
    if severity:
        query = query.filter(Anomaly.severity == severity)
    
    if anomaly_type:
        query = query.filter(Anomaly.anomaly_type == anomaly_type)
    
    anomalies = query.order_by(Anomaly.created_at.desc()).offset(skip).limit(limit).all()
    
    return anomalies

@router.get("/anomalies/summary", response_model=AnomalySummary)
async def get_anomaly_summary(
    video_id: Optional[int] = None,
    hours: Optional[int] = Query(24, description="Summary for last N hours"),
    db: Session = Depends(get_db)
):
    """Get anomaly summary statistics."""
    
    query = db.query(Anomaly)
    
    if video_id:
        query = query.filter(Anomaly.video_id == video_id)
    
    if hours:
        since = datetime.utcnow() - timedelta(hours=hours)
        query = query.filter(Anomaly.created_at >= since)
    
    anomalies = query.all()
    
    # Calculate statistics
    stats = {
        'total': len(anomalies),
        'by_type': {},
        'by_severity': {'high': 0, 'medium': 0, 'low': 0},
        'recent_high_severity': []
    }
    
    for anomaly in anomalies:
        # Count by type
        if anomaly.anomaly_type not in stats['by_type']:
            stats['by_type'][anomaly.anomaly_type] = 0
        stats['by_type'][anomaly.anomaly_type] += 1
        
        # Count by severity
        stats['by_severity'][anomaly.severity] += 1
        
        # Get recent high severity
        if anomaly.severity == 'high' and len(stats['recent_high_severity']) < 5:
            stats['recent_high_severity'].append({
                'id': anomaly.id,
                'type': anomaly.anomaly_type,
                'timestamp': anomaly.timestamp,
                'description': anomaly.description,
                'video_id': anomaly.video_id
            })
    
    return AnomalySummary(**stats)

@router.get("/videos/{video_id}/anomaly-frames")
async def get_anomaly_frames(
    video_id: int,
    db: Session = Depends(get_db)
):
    """Get list of frames with anomalies."""
    
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    anomaly_frames_dir = settings.PROCESSED_DIR / f"video_{video_id}_anomalies"
    
    if not anomaly_frames_dir.exists():
        return {"frames": []}
    
    frames = []
    for frame_file in anomaly_frames_dir.glob("*.jpg"):
        frames.append({
            'filename': frame_file.name,
            'url': f"/api/v1/videos/{video_id}/anomaly-frames/{frame_file.name}"
        })
    
    # Sort by filename (which includes timestamp)
    frames.sort(key=lambda x: x['filename'])
    
    return {"frames": frames}

@router.get("/anomalies/statistics", response_model=AnomalyStatistics)
async def get_anomaly_statistics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """Get detailed anomaly statistics."""
    
    query = db.query(Anomaly)
    
    if start_date:
        query = query.filter(Anomaly.created_at >= start_date)
    
    if end_date:
        query = query.filter(Anomaly.created_at <= end_date)
    
    anomalies = query.all()
    
    # Calculate hourly distribution
    hourly_dist = {}
    for anomaly in anomalies:
        hour = anomaly.created_at.hour
        if hour not in hourly_dist:
            hourly_dist[hour] = 0
        hourly_dist[hour] += 1
    
    # Calculate daily trends
    daily_trends = {}
    for anomaly in anomalies:
        date = anomaly.created_at.date()
        if date not in daily_trends:
            daily_trends[date] = {
                'total': 0,
                'high': 0,
                'medium': 0,
                'low': 0
            }
        daily_trends[date]['total'] += 1
        daily_trends[date][anomaly.severity] += 1
    
    return {
        'total_anomalies': len(anomalies),
        'hourly_distribution': hourly_dist,
        'daily_trends': [
            {
                'date': str(date),
                **counts
            }
            for date, counts in sorted(daily_trends.items())
        ],
        'most_common_types': sorted(
            [(atype, count) for atype, count in stats['by_type'].items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
    }