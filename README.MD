# Luggage Monitoring System - Backend

AI-powered luggage monitoring system for detecting abandoned luggage in surveillance footage.

## Features

- 🎥 Video upload and processing
- 🤖 AI-powered luggage detection (YOLO)
- 🚨 Abandoned luggage alerts
- 📧 Email notifications
- 💬 Natural language queries (Gemini AI)
- 📊 Analytics and reporting
- 🔍 Face analysis (age, gender, emotion)
- 📹 Real-time WebSocket updates

## Quick Start

### Local Development

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure environment**:
Edit the `.env` file with your API keys and settings.

**Important**: CORS_ORIGINS must have NO SPACES between URLs:
```env
# ✅ Correct
CORS_ORIGINS=http://localhost:3000,http://localhost:5173,https://example.com

# ❌ Wrong (has spaces)
CORS_ORIGINS=http://localhost:3000, http://localhost:5173, https://example.com
```

3. **Run the application**:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Docker Deployment

All Docker files are now in this directory. See [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) for full instructions.

**Quick start**:
```bash
# Make sure you're in the app directory
cd app

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Tech Stack

- **FastAPI** - Modern web framework
- **SQLAlchemy** - Database ORM
- **YOLO (Ultralytics)** - Object detection
- **Google Gemini** - AI-powered queries
- **OpenCV** - Video processing
- **DeepFace** - Face analysis
- **MySQL/SQLite** - Database

## Environment Variables

Key environment variables in `.env`:

- `GOOGLE_API_KEY` - Google Gemini API key
- `DATABASE_URL` - Database connection string
- `CORS_ORIGINS` - Allowed CORS origins (**no spaces!**)
- `SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD` - Email configuration
- `YOLO_MODEL` - YOLO model to use (default: yolov8m.pt)
- `ENABLE_EMAIL_ALERTS` - Enable/disable email alerts

## License

Proprietary - MEHH Company