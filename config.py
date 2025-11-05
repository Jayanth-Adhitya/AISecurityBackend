import os
from pathlib import Path
from dotenv import load_dotenv
import urllib.parse

load_dotenv()

class Settings:
    # App settings
    APP_NAME = "Luggage Monitoring System"
    VERSION = "2.0.0"
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"

    # Database
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

    # Check if DATABASE_URL is explicitly set first
    DATABASE_URL = os.getenv("DATABASE_URL")

    if DATABASE_URL is None:
        # No DATABASE_URL set, check environment
        if ENVIRONMENT == "production":
            # Production: Use Aiven MySQL with SSL (only if DATABASE_URL not set)
            MYSQL_USER = os.getenv("MYSQL_USER", "avnadmin")
            MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
            MYSQL_HOST = os.getenv("MYSQL_HOST", "")
            MYSQL_PORT = os.getenv("MYSQL_PORT", "12345")
            MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "defaultdb")

            # URL encode password to handle special characters
            encoded_password = urllib.parse.quote_plus(MYSQL_PASSWORD) if MYSQL_PASSWORD else ""

            # Aiven requires SSL
            DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{encoded_password}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?ssl_ca=&ssl_verify_cert=true&ssl_verify_identity=true"
        else:
            # Development: Use SQLite
            DATABASE_URL = "sqlite:///./luggage_monitoring.db"
    
    # File paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    UPLOAD_DIR = BASE_DIR / "uploads"
    PROCESSED_DIR = BASE_DIR / "processed"
    TEMP_DIR = BASE_DIR / "temp"
    
    # Create directories if they don't exist
    UPLOAD_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)
    
    # Video processing
    FRAME_SKIP = int(os.getenv("FRAME_SKIP", "20"))  # Process every Nth frame
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    MAX_VIDEO_SIZE_MB = int(os.getenv("MAX_VIDEO_SIZE_MB", "500"))
    
    # YOLO settings
    # Use smaller model for production to fit in memory limits
    default_model = "yolov8n.pt" if ENVIRONMENT == "production" else "yolov8m.pt"
    YOLO_MODEL = os.getenv("YOLO_MODEL", default_model)
    YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cpu")  # Use CPU for free tier
    TARGET_CLASSES = ["suitcase", "backpack", "handbag", "person"]  # Luggage detection

    # Luggage detection settings
    ABANDONMENT_THRESHOLD = float(os.getenv("ABANDONMENT_THRESHOLD", "10.0"))  # seconds
    OWNER_PROXIMITY_DISTANCE = float(os.getenv("OWNER_PROXIMITY_DISTANCE", "150"))  # pixels
    STATIONARY_THRESHOLD = float(os.getenv("STATIONARY_THRESHOLD", "20"))  # pixels movement
    MIN_LUGGAGE_CONFIDENCE = float(os.getenv("MIN_LUGGAGE_CONFIDENCE", "0.5"))
    
    # LLM settings
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    MODEL_NAME = "gemini-2.5-flash"
    
    # CORS
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,https://aisecurity.mehh.ae,http://focc0cosgwcskkws0wwkg4k4.31.97.62.24.sslip.io:7000").split(",")
    
    # Pose detection
    POSE_MODEL = os.getenv("POSE_MODEL", "yolov8m-pose.pt")

    # Anomaly detection
    ANOMALY_EMAIL_COOLDOWN = int(os.getenv("ANOMALY_EMAIL_COOLDOWN", "0"))  # No cooldown for security alerts
    ANOMALY_SEVERITY_THRESHOLD = os.getenv("ANOMALY_SEVERITY_THRESHOLD", "high")  # Only high severity

    # Email alerts
    ENABLE_EMAIL_ALERTS = os.getenv("ENABLE_EMAIL_ALERTS", "True").lower() == "true"  # Enable by default
    EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
    EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
    EMAIL_USER = os.getenv("EMAIL_USER", "")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
    EMAIL_FROM = os.getenv("EMAIL_FROM", os.getenv("EMAIL_USER", ""))
    
    # Face Analysis Settings
    ANALYZE_FACES = True
    FACE_ANALYSIS_INTERVAL = 5  # Analyze faces every N frames for performance
    FACE_DETECTOR_BACKEND = 'opencv'  # Options: 'opencv', 'retinaface', 'mtcnn', 'ssd', 'dlib', 'mediapipe'
    FACE_ANALYSIS_ACTIONS = ['age', 'gender', 'emotion', 'race']

settings = Settings()