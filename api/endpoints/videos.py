from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
import aiofiles
import os
from pathlib import Path
import uuid


from models.database import get_db, Video, Summary, Detection
from schemas.video import VideoResponse, SummaryResponse
from services.video_analyzer import VideoAnalyzer
from config import settings
from fastapi import Response
from fastapi.responses import FileResponse, JSONResponse
import json

router = APIRouter()
analyzer = VideoAnalyzer()

@router.post("/upload", response_model=VideoResponse)
async def upload_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a video file for analysis."""
    
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")
    
    # Check file size
    file_size = 0
    contents = await file.read()
    file_size = len(contents)
    
    if file_size > settings.MAX_VIDEO_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File too large. Maximum size is {settings.MAX_VIDEO_SIZE_MB}MB")
    
    # Generate unique filename
    file_extension = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = settings.UPLOAD_DIR / unique_filename
    
    # Save file
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(contents)
    
    # Create database entry
    video = Video(
        filename=unique_filename,
        original_filename=file.filename,
        file_path=str(file_path),
        status="uploaded"
    )
    db.add(video)
    db.commit()
    db.refresh(video)
    
    return video

@router.post("/analyze/{video_id}")
async def analyze_video(
    video_id: int,
    alert_email: Optional[str] = None,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Start video analysis process.

    Args:
        video_id: ID of the video to analyze
        alert_email: Email address to send security alerts to (session-based)
    """

    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    if video.status == "processing":
        raise HTTPException(status_code=400, detail="Video is already being processed")

    if video.status == "completed":
        raise HTTPException(status_code=400, detail="Video has already been analyzed")

    # Start analysis in background with alert email
    background_tasks.add_task(analyzer.analyze_video, video_id, db, alert_email)

    return {"message": "Analysis started", "video_id": video_id, "alert_email": alert_email}

@router.get("/videos/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: int,
    db: Session = Depends(get_db)
):
    """Get video details."""
    
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return video

@router.get("/videos/{video_id}/summary", response_model=SummaryResponse)
async def get_video_summary(
    video_id: int,
    db: Session = Depends(get_db)
):
    """Get video analysis summary."""

    summary = db.query(Summary).filter(Summary.video_id == video_id).first()
    if not summary:
        raise HTTPException(status_code=404, detail="Summary not found. Video may not be analyzed yet.")
    
    return summary

@router.get("/videos", response_model=List[VideoResponse])
async def list_videos(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """List all videos."""
    
    videos = db.query(Video).offset(skip).limit(limit).all()
    return videos

@router.get("/videos/{video_id}/frames")
async def get_video_frames(
    video_id: int,
    db: Session = Depends(get_db)
):
    """Get list of annotated frames for a video."""
    
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if video.status != "completed":
        raise HTTPException(status_code=400, detail="Video analysis not completed")
    
    frames_dir = settings.PROCESSED_DIR / f"video_{video_id}_frames"
    metadata_path = frames_dir / "metadata.json"
    
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Frame metadata not found")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Add full URL for each frame
    for frame in metadata['frames']:
        frame['url'] = f"/api/v1/videos/{video_id}/frames/{frame['filename']}"
    
    return JSONResponse(content=metadata)

@router.get("/videos/{video_id}/frames/{filename}")
async def get_frame_image(
    video_id: int,
    filename: str,
    db: Session = Depends(get_db)
):
    """Get a specific annotated frame image."""
    
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    frame_path = settings.PROCESSED_DIR / f"video_{video_id}_frames" / filename
    
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")
    
    return FileResponse(
        path=str(frame_path),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"}
    )

@router.get("/videos/{video_id}/detections")
async def get_video_detections(
    video_id: int,
    skip: int = 0,
    limit: int = 100,
    object_class: Optional[str] = None,
    color: Optional[str] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    db: Session = Depends(get_db)
):
    """Get detections for a video with optional filters."""
    
    query = db.query(Detection).filter(Detection.video_id == video_id)
    
    if object_class:
        query = query.filter(Detection.object_class == object_class)
    
    if color:
        query = query.filter(Detection.color == color)
    
    if start_time is not None:
        query = query.filter(Detection.timestamp >= start_time)
    
    if end_time is not None:
        query = query.filter(Detection.timestamp <= end_time)
    
    total = query.count()
    detections = query.order_by(Detection.timestamp).offset(skip).limit(limit).all()
    
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "detections": detections
    }