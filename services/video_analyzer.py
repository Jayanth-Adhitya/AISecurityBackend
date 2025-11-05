import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from ultralytics import YOLO
import asyncio
from datetime import datetime
from sqlalchemy.orm import Session
import json
import os
from utils.json_utils import convert_numpy_types, prepare_for_json

from models.database import Video, Detection, Summary, Anomaly
from services.color_detector import ColorDetector
from services.luggage_tracker import LuggageTracker
from services.owner_detector import OwnerDetector
from services.pose_detector import PoseDetector
from services.anomaly_detector import AnomalyDetector
from services.email_service import EmailService
from config import settings

def convert_np_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(i) for i in obj]
    return obj

class VideoAnalyzer:
    def __init__(self):
        # Object detection model
        self.model = YOLO(settings.YOLO_MODEL)

        # Pose detection model (still needed for person detection)
        self.pose_detector = PoseDetector(model_path=settings.POSE_MODEL)

        # Services
        self.color_detector = ColorDetector()
        self.luggage_tracker = LuggageTracker()
        self.owner_detector = OwnerDetector()
        self.anomaly_detector = AnomalyDetector()
        self.email_service = EmailService()

        # Configuration
        self.target_classes = settings.TARGET_CLASSES

        # Anomaly tracking
        self.last_email_sent = {}  # Track last email time for each anomaly type
        self.email_cooldown = settings.ANOMALY_EMAIL_COOLDOWN

        # Session email for alerts
        self.alert_email = None
    
    async def analyze_video(self, video_id: int, db: Session, alert_email: str = None):
        """
        Analyze a video file for abandoned luggage detection.

        Args:
            video_id: ID of the video to analyze
            db: Database session
            alert_email: Email address to send alerts to (session-based)
        """
        self.alert_email = alert_email
        # Get video from database
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise ValueError(f"Video with ID {video_id} not found")
        
        # Update status to processing
        video.status = "processing"
        video.process_start_time = datetime.utcnow()
        db.commit()
        
        try:
            # Open video file
            cap = cv2.VideoCapture(video.file_path)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Update video metadata
            video.fps = fps
            video.width = width
            video.height = height
            video.duration = duration
            db.commit()
            
            # Create directories
            frames_dir = settings.PROCESSED_DIR / f"video_{video_id}_frames"
            anomaly_frames_dir = settings.PROCESSED_DIR / f"video_{video_id}_anomalies"
            frames_dir.mkdir(exist_ok=True)
            anomaly_frames_dir.mkdir(exist_ok=True)
            
            # Initialize tracking
            self.luggage_tracker = LuggageTracker()
            self.owner_detector = OwnerDetector()
            self.anomaly_detector = AnomalyDetector(fps=fps)
            
            # Process frames
            frame_count = 0
            detections_batch = []
            anomalies_batch = []
            timeline_data = {}
            saved_frames = []
            anomaly_frames = []
            all_detections_for_summary = []
            
            # Statistics
            total_anomalies = {
                'abandoned_luggage': 0,
                'suspicious_object': 0,
                'other': 0
            }
            
            # Calculate frame skip
            actual_frame_skip = max(1, int(fps / 2))  # Process 2 FPS
            
            print(f"Processing video at 2 FPS (skipping {actual_frame_skip} frames)")
            print(f"Total frames to process: {total_frames // actual_frame_skip}")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames for efficiency
                if frame_count % actual_frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Calculate timestamp
                timestamp = frame_count / fps if fps > 0 else 0
                interval_key = int(timestamp // 10) * 10  # 10-second intervals
                
                if interval_key not in timeline_data:
                    timeline_data[interval_key] = {
                        'timestamp': interval_key,
                        'unique_luggage': set(),
                        'persons': 0,
                        'abandoned_count': 0,
                        'anomalies': []
                    }
                
                # Resize frame for faster processing
                scale = 640 / width if width > 640 else 1.0
                if scale < 1.0:
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    resized_frame = cv2.resize(frame, (new_width, new_height))
                else:
                    resized_frame = frame
                    scale = 1.0
                
                # Run YOLO detection
                results = self.model(resized_frame, conf=settings.CONFIDENCE_THRESHOLD)
                
                # Prepare detections for tracking
                frame_detections = []
                luggage_detections = []
                person_detections = []

                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            cls_id = int(box.cls[0])
                            cls_name = self.model.names[cls_id]

                            if cls_name not in self.target_classes:
                                continue

                            conf = float(box.conf[0])

                            # Scale bbox back to original frame size
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            x1, y1, x2, y2 = int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)

                            detection_dict = {
                                'bbox_x1': x1,
                                'bbox_y1': y1,
                                'bbox_x2': x2,
                                'bbox_y2': y2,
                                'object_class': cls_name,
                                'confidence': conf,
                                'color': None
                            }

                            # Detect color for luggage
                            if cls_name in ['suitcase', 'backpack', 'handbag']:
                                detection_dict['color'] = self.color_detector.detect_color(frame, (x1, y1, x2, y2))
                                luggage_detections.append(detection_dict)
                            elif cls_name == 'person':
                                person_detections.append(detection_dict)

                            frame_detections.append(detection_dict)
                
                # Update luggage tracker
                tracked_luggage = self.luggage_tracker.update_tracks(luggage_detections, frame_count)

                # Update owner proximity tracking
                proximity_map = self.owner_detector.update_proximity(
                    frame_number=frame_count,
                    luggage_detections=tracked_luggage,
                    person_detections=person_detections
                )

                # Check for abandoned luggage
                abandoned_items = self.owner_detector.get_abandoned_luggage(
                    current_frame=frame_count,
                    fps=fps,
                    abandonment_threshold_seconds=settings.ABANDONMENT_THRESHOLD
                )
                
                # Process abandoned luggage anomalies
                luggage_anomalies = []

                for abandoned in abandoned_items:
                    luggage_track_id = abandoned['luggage_track_id']
                    owner_bbox = abandoned.get('owner_bbox')
                    luggage_bbox = abandoned.get('luggage_bbox')

                    # Capture owner image if available
                    owner_image_path = None
                    if owner_bbox and len(owner_bbox) == 4:
                        owner_image_path = self.owner_detector.capture_owner_image(
                            frame=frame,
                            owner_bbox=owner_bbox,
                            luggage_bbox=luggage_bbox,
                            video_id=video_id,
                            luggage_track_id=luggage_track_id
                        )

                    # Get luggage info from tracker
                    luggage_info = self.luggage_tracker.get_track_info(luggage_track_id)
                    luggage_class = luggage_info['class'] if luggage_info else 'unknown'
                    luggage_color = luggage_info.get('color', 'unknown') if luggage_info else 'unknown'
                    luggage_desc = f"{luggage_color} {luggage_class}".strip()

                    # Create anomaly record
                    anomaly = {
                        'type': 'ABANDONED_LUGGAGE',
                        'severity': 'high',  # Always high severity for security
                        'confidence': 0.9,
                        'description': f'Abandoned {luggage_desc} detected for {abandoned["seconds_unattended"]:.1f} seconds',
                        'track_ids': [luggage_track_id],
                        'owner_id': abandoned.get('owner_id'),
                        'owner_image_path': owner_image_path,
                        'luggage_description': luggage_desc,
                        'seconds_unattended': abandoned['seconds_unattended'],
                        'bbox': [
                            luggage_info['bbox'][0],
                            luggage_info['bbox'][1],
                            luggage_info['bbox'][2],
                            luggage_info['bbox'][3]
                        ] if luggage_info else None
                    }
                    luggage_anomalies.append(anomaly)
                    total_anomalies['abandoned_luggage'] += 1

                    # Update timeline
                    timeline_data[interval_key]['abandoned_count'] += 1

                # Combine all anomalies
                frame_anomalies = luggage_anomalies
                
                # Save anomalies to database and send alerts
                for anomaly in frame_anomalies:
                    # Convert all numpy types to Python native types
                    clean_anomaly = prepare_for_json(anomaly)
                    
                    # Extract track_ids safely
                    track_ids = clean_anomaly.get('track_ids', [])
                    if track_ids and not isinstance(track_ids[0], (int, str)):
                        track_ids = [int(tid) for tid in track_ids]
                    
                    # Create database record with cleaned data
                    anomaly_record = Anomaly(
                        video_id=int(video_id),
                        frame_number=int(frame_count),
                        timestamp=float(timestamp),
                        anomaly_type=str(clean_anomaly['type']),
                        severity=str(clean_anomaly['severity']),
                        confidence=float(clean_anomaly['confidence']),
                        description=str(clean_anomaly['description']),
                        track_ids=track_ids,
                        additional_data=clean_anomaly,
                        owner_image_path=clean_anomaly.get('owner_image_path'),
                        luggage_description=clean_anomaly.get('luggage_description'),
                        email_sent=False
                    )
                    anomalies_batch.append(anomaly_record)
                    
                    # Send email alert for high severity anomalies
                    if clean_anomaly['severity'] == 'high' and settings.ENABLE_EMAIL_ALERTS and self.alert_email:
                        await self._send_anomaly_alert(
                            anomaly=clean_anomaly,
                            video=video,
                            frame=frame,
                            frame_number=frame_count,
                            timestamp=timestamp,
                            anomaly_frames_dir=anomaly_frames_dir,
                            alert_email=self.alert_email
                        )
                
                # Create annotated frame
                annotated_frame = None
                save_frame = (int(timestamp) % 5 == 0) and (int(timestamp * 10) % 50 == 0)
                
                if save_frame or (frame_count == 0) or len(frame_anomalies) > 0 or (len(tracked_luggage) > 0 and len(saved_frames) < 20):
                    annotated_frame = frame.copy()

                    # Draw luggage detections
                    for detection in tracked_luggage:
                        self._draw_luggage_detection(annotated_frame, detection, proximity_map)

                    # Draw person detections
                    for person in person_detections:
                        self._draw_person_detection(annotated_frame, person)

                    # Draw anomaly indicators
                    if len(frame_anomalies) > 0:
                        self._draw_anomaly_indicators(annotated_frame, frame_anomalies, timestamp)
                
                # Save annotated frame
                if annotated_frame is not None:
                    if len(frame_anomalies) > 0:
                        # Save as anomaly frame
                        anomaly_filename = f"anomaly_{frame_count:06d}_{timestamp:.1f}s.jpg"
                        anomaly_path = anomaly_frames_dir / anomaly_filename
                        cv2.imwrite(str(anomaly_path), annotated_frame)
                        anomaly_frames.append({
                            'frame_number': frame_count,
                            'timestamp': timestamp,
                            'filename': anomaly_filename,
                            'anomalies': [a['type'] for a in frame_anomalies]
                        })
                    
                    if save_frame or len(saved_frames) < 20:
                        # Save as regular frame
                        frame_filename = f"frame_{frame_count:06d}_{timestamp:.1f}s.jpg"
                        frame_path = frames_dir / frame_filename
                        cv2.imwrite(str(frame_path), annotated_frame)
                        saved_frames.append({
                            'frame_number': frame_count,
                            'timestamp': timestamp,
                            'filename': frame_filename,
                            'luggage_count': len(tracked_luggage),
                            'person_count': len(person_detections),
                            'has_anomaly': len(frame_anomalies) > 0
                        })

                # Process tracked luggage for database
                for detection in tracked_luggage:
                    track_id = int(detection.get('track_id', -1))

                    # Update timeline data
                    timeline_data[interval_key]['unique_luggage'].add(track_id)

                    # Add anomalies to timeline
                    if len(frame_anomalies) > 0:
                        timeline_data[interval_key]['anomalies'].extend(
                            [a['type'] for a in frame_anomalies]
                        )

                    # Create detection record with native types
                    detection_record = Detection(
                        video_id=int(video_id),
                        frame_number=int(frame_count),
                        timestamp=float(timestamp),
                        object_class=str(detection['object_class']),
                        confidence=float(detection['confidence']),
                        color=str(detection.get('color')) if detection.get('color') else None,
                        track_id=track_id,
                        bbox_x1=int(detection['bbox_x1']),
                        bbox_y1=int(detection['bbox_y1']),
                        bbox_x2=int(detection['bbox_x2']),
                        bbox_y2=int(detection['bbox_y2'])
                    )
                    detections_batch.append(detection_record)

                # Count persons
                timeline_data[interval_key]['persons'] += len(person_detections)
                
                # Batch insert with error handling
                if len(detections_batch) >= 50:
                    try:
                        db.bulk_save_objects(detections_batch)
                        db.commit()
                    except Exception as e:
                        print(f"Error saving detections batch: {e}")
                        db.rollback()
                    finally:
                        detections_batch = []
                
                if len(anomalies_batch) >= 10:
                    try:
                        db.bulk_save_objects(anomalies_batch)
                        db.commit()
                    except Exception as e:
                        print(f"Error saving anomalies batch: {e}")
                        db.rollback()
                    finally:
                        anomalies_batch = []
                
                frame_count += 1
                
                # Progress update
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames, Anomalies: {sum(total_anomalies.values())}")
                    await asyncio.sleep(0)
            
            # Insert remaining records
            if detections_batch:
                try:
                    db.bulk_save_objects(detections_batch)
                    db.commit()
                except Exception as e:
                    print(f"Error saving final detections batch: {e}")
                    db.rollback()
            
            if anomalies_batch:
                try:
                    db.bulk_save_objects(anomalies_batch)
                    db.commit()
                except Exception as e:
                    print(f"Error saving final anomalies batch: {e}")
                    db.rollback()
            
            # Get luggage tracking summary
            unique_luggage_count = self.luggage_tracker.get_unique_count()
            luggage_by_class = self.luggage_tracker.get_luggage_by_class()

            # Process timeline data
            timeline_list = []
            total_abandoned = 0
            for interval, data in timeline_data.items():
                timeline_list.append({
                    'timestamp': interval,
                    'luggage': len(data['unique_luggage']),
                    'persons': data['persons'],
                    'abandoned_count': data.get('abandoned_count', 0),
                    'anomalies': list(set(data['anomalies']))  # Unique anomaly types
                })
                total_abandoned += data.get('abandoned_count', 0)

            # Create summary with luggage statistics
            summary_data = {
                'total_luggage': unique_luggage_count,
                'total_persons': sum([d['persons'] for d in timeline_list]),
                'abandoned_count': total_abandoned,
                'owner_identified_count': 0,  # Will be calculated from anomalies
                'luggage_types': dict(luggage_by_class),
                'color_distribution': {},  # Can be calculated from detections if needed
                'timeline_data': timeline_list,
                'processed_frames': frame_count // actual_frame_skip,
                'anomaly_statistics': total_anomalies,
                'total_anomalies': sum(total_anomalies.values())
            }

            summary = Summary(
                video_id=video_id,
                total_luggage=summary_data['total_luggage'],
                total_persons=summary_data['total_persons'],
                abandoned_count=summary_data['abandoned_count'],
                owner_identified_count=summary_data['owner_identified_count'],
                luggage_types=summary_data['luggage_types'],
                color_distribution=summary_data['color_distribution'],
                timeline_data=summary_data['timeline_data'],
                processed_frames=summary_data['processed_frames']
            )
            db.add(summary)
            
            # Save metadata
            frames_metadata_path = frames_dir / "metadata.json"
            with open(frames_metadata_path, 'w') as f:
                json.dump({
                    'video_id': video_id,
                    'total_frames': len(saved_frames),
                    'frames': saved_frames,
                    'anomaly_frames': anomaly_frames,
                    'anomaly_statistics': total_anomalies
                }, f, indent=2)
            
            # Update video status
            video.status = "completed"
            video.process_end_time = datetime.utcnow()
            db.commit()
            
            cap.release()

            print(f"Analysis complete:")
            print(f"  - Unique luggage items: {unique_luggage_count}")
            print(f"  - Abandoned luggage: {total_abandoned}")
            print(f"  - Total anomalies: {sum(total_anomalies.values())}")
            print(f"  - Anomaly breakdown: {total_anomalies}")

            return {
                "status": "success",
                "processed_frames": frame_count // actual_frame_skip,
                "unique_luggage": unique_luggage_count,
                "abandoned_luggage": total_abandoned,
                "total_anomalies": sum(total_anomalies.values()),
                "anomaly_breakdown": total_anomalies,
                "saved_frames": len(saved_frames),
                "anomaly_frames": len(anomaly_frames)
            }
            
        except Exception as e:
            print(f"Error during video analysis: {e}")
            video.status = "failed"
            video.error_message = str(e)
            video.process_end_time = datetime.utcnow()
            db.commit()
            raise e
    
    def _draw_luggage_detection(self, frame: np.ndarray, detection: Dict[str, Any], proximity_map: Dict):
        """Draw luggage detection on frame."""
        x1 = detection['bbox_x1']
        y1 = detection['bbox_y1']
        x2 = detection['bbox_x2']
        y2 = detection['bbox_y2']
        track_id = detection.get('track_id', -1)
        is_stationary = detection.get('is_stationary', False)

        # Get proximity status
        proximity_info = proximity_map.get(track_id, {})
        has_owner_nearby = proximity_info.get('has_owner_nearby', True)

        # Color based on status
        if not has_owner_nearby and is_stationary:
            rect_color = (0, 0, 255)  # Red for unattended stationary luggage
        elif not has_owner_nearby:
            rect_color = (0, 165, 255)  # Orange for unattended
        else:
            rect_color = (0, 255, 0)  # Green for attended

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 2)

        # Create label
        label_parts = [f"{detection['object_class']}"]
        if detection.get('color'):
            label_parts.append(detection['color'])
        label_parts.append(f"ID:{track_id}")
        if not has_owner_nearby:
            label_parts.append("UNATTENDED")
        label = " | ".join(label_parts)

        # Draw label
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    rect_color, -1)
        cv2.putText(frame, label,
                  (x1, y1 - 5),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.5, (255, 255, 255), 1)

    def _draw_person_detection(self, frame: np.ndarray, detection: Dict[str, Any]):
        """Draw person detection on frame."""
        x1 = detection['bbox_x1']
        y1 = detection['bbox_y1']
        x2 = detection['bbox_x2']
        y2 = detection['bbox_y2']

        # Draw bounding box (light blue for persons)
        rect_color = (255, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 2)

        # Draw label
        label = "Person"
        cv2.putText(frame, label,
                  (x1, y1 - 5),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.5, rect_color, 2)
    
    
    def _draw_anomaly_indicators(self, frame: np.ndarray, anomalies: List[Dict], timestamp: float):
        """Draw anomaly indicators on frame."""
        # Draw warning header
        if anomalies:
            # Find highest severity
            severities = [a['severity'] for a in anomalies]
            if 'high' in severities:
                header_color = (0, 0, 255)  # Red
                header_text = "⚠ CRITICAL ANOMALY DETECTED"
            elif 'medium' in severities:
                header_color = (0, 165, 255)  # Orange
                header_text = "⚠ WARNING: ANOMALY DETECTED"
            else:
                header_color = (0, 255, 255)  # Yellow
                header_text = "ℹ ANOMALY DETECTED"
            
            # Draw header background
            # cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), header_color, -1)
            # cv2.putText(frame, header_text,
            #           (10, 25),
            #           cv2.FONT_HERSHEY_SIMPLEX,
            #           0.8, (255, 255, 255), 2)
            
            # List anomalies
            # y_offset = 60
            # for anomaly in anomalies[:3]:  # Show top 3
            #     text = f"• {anomaly['type']}: {anomaly['description'][:50]}"
            #     cv2.putText(frame, text,
            #               (10, y_offset),
            #               cv2.FONT_HERSHEY_SIMPLEX,
            #               0.5, header_color, 1)
            #     y_offset += 20
            
            # Add timestamp
            # cv2.putText(frame, f"Time: {timestamp:.1f}s",
            #           (frame.shape[1] - 150, 25),
            #           cv2.FONT_HERSHEY_SIMPLEX,
            #           0.6, (255, 255, 255), 1)
    
    async def _send_anomaly_alert(
        self,
        anomaly: Dict,
        video: Video,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
        anomaly_frames_dir: Path,
        alert_email: str
    ):
        """Send email alert for anomaly."""
        try:
            # Check cooldown
            anomaly_type = anomaly['type']
            current_time = datetime.utcnow()
            
            if anomaly_type in self.last_email_sent:
                time_diff = (current_time - self.last_email_sent[anomaly_type]).total_seconds()
                if time_diff < self.email_cooldown:
                    return  # Skip email due to cooldown
            
            # Save frame for email
            frame_filename = f"alert_{anomaly_type}_{frame_number}.jpg"
            frame_path = anomaly_frames_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)

            # Collect image attachments - only send the annotated frame showing the luggage
            frame_images = [frame_path]
            # Note: Not sending owner_image_path as it may show empty background after owner left

            # Send email
            await self.email_service.send_anomaly_alert(
                recipient_email=alert_email,
                anomaly_type=anomaly_type,
                severity=anomaly['severity'],
                confidence=anomaly['confidence'],
                video_id=video.id,
                video_name=video.original_filename,
                frame_number=frame_number,
                timestamp=timestamp,
                description=anomaly['description'],
                luggage_description=anomaly.get('luggage_description', 'Unknown'),
                frame_images=frame_images,
                additional_info=anomaly.get('additional_info', [])
            )
            
            # Update last sent time
            self.last_email_sent[anomaly_type] = current_time
            
        except Exception as e:
            print(f"Error during video analysis: {e}")
            try:
                db.rollback()  # Rollback any pending transaction
                video.status = "failed"
                video.error_message = str(e)
                video.process_end_time = datetime.utcnow()
                db.commit()
            except:
                pass  # If even this fails, just log it
            raise e