from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
from typing import List
import json
import asyncio
from datetime import datetime, timedelta

from models.database import get_db, Anomaly

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@router.websocket("/ws/anomalies")
async def websocket_anomalies(websocket: WebSocket, db: Session = Depends(get_db)):
    """WebSocket endpoint for real-time anomaly updates."""
    await manager.connect(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_text(json.dumps({
            "type": "connection",
            "message": "Connected to anomaly stream"
        }))
        
        # Keep checking for new anomalies
        last_check = datetime.utcnow()
        
        while True:
            # Check for new anomalies every 2 seconds
            await asyncio.sleep(2)
            
            # Get recent anomalies
            new_anomalies = db.query(Anomaly).filter(
                Anomaly.created_at > last_check
            ).all()
            
            if new_anomalies:
                for anomaly in new_anomalies:
                    anomaly_data = {
                        "type": "anomaly",
                        "data": {
                            "id": anomaly.id,
                            "video_id": anomaly.video_id,
                            "anomaly_type": anomaly.anomaly_type,
                            "severity": anomaly.severity,
                            "confidence": anomaly.confidence,
                            "description": anomaly.description,
                            "timestamp": anomaly.timestamp,
                            "created_at": anomaly.created_at.isoformat()
                        }
                    }
                    await websocket.send_text(json.dumps(anomaly_data))
            
            last_check = datetime.utcnow()
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)