# Traffic Analysis Backend

A FastAPI backend for analyzing traffic CCTV videos using YOLOv8 and providing natural language query capabilities.

## Features

- Video upload and management
- Automatic vehicle and person detection using YOLOv8
- Vehicle color detection
- Natural language queries using Google Gemini
- RESTful API endpoints
- SQLite/PostgreSQL database support

## Setup

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (optional, for faster processing)
- Google Gemini API key (for natural language queries)

### Installation

1. Clone the repository:
```bash
git clone <repository_url>
cd traffic-analysis-backend