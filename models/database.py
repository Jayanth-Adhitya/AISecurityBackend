from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Text, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from config import settings

engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Video(Base):
    __tablename__ = "videos"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    duration = Column(Float, nullable=True)
    fps = Column(Float, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    status = Column(String, default="uploaded")  # uploaded, processing, completed, failed
    upload_time = Column(DateTime, default=datetime.utcnow)
    process_start_time = Column(DateTime, nullable=True)
    process_end_time = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Relationships
    detections = relationship("Detection", back_populates="video", cascade="all, delete-orphan")
    summaries = relationship("Summary", back_populates="video", cascade="all, delete-orphan")

class Detection(Base):
    __tablename__ = "detections"
    
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False, index=True)
    frame_number = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)  # seconds from start
    track_id = Column(Integer, nullable=True, index=True)
    object_class = Column(String, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    color = Column(String, nullable=True)
    bbox_x1 = Column(Integer)
    bbox_y1 = Column(Integer)
    bbox_x2 = Column(Integer)
    bbox_y2 = Column(Integer)
    
    # Relationship
    video = relationship("Video", back_populates="detections")

class Summary(Base):
    __tablename__ = "summaries"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False, unique=True)
    total_luggage = Column(Integer, default=0)  # Total luggage detected
    total_persons = Column(Integer, default=0)
    abandoned_count = Column(Integer, default=0)  # Abandoned luggage count
    owner_identified_count = Column(Integer, default=0)  # Luggage with identified owners
    luggage_types = Column(JSON)  # {"suitcase": 10, "backpack": 5, ...}
    color_distribution = Column(JSON)  # {"red": 3, "blue": 2, ...}
    timeline_data = Column(JSON)  # Per-minute or per-interval counts
    processed_frames = Column(Integer, default=0)

    # Relationship
    video = relationship("Video", back_populates="summaries")
    
class Anomaly(Base):
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False, index=True)
    frame_number = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)
    anomaly_type = Column(String, nullable=False, index=True)  # e.g., "abandoned_luggage"
    severity = Column(String, nullable=False)  # high, medium, low
    confidence = Column(Float, nullable=False)
    description = Column(Text)
    track_ids = Column(JSON)  # List of involved track IDs
    additional_data = Column(JSON)  # Any extra information
    owner_image_path = Column(String, nullable=True)  # Path to owner's captured image
    luggage_description = Column(Text, nullable=True)  # Color, size, type of luggage
    email_sent = Column(Boolean, default=False)
    email_sent_time = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    video = relationship("Video", backref="anomalies")
# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()