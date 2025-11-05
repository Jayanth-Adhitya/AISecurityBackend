import os
import base64
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain.schema import HumanMessage, SystemMessage
except Exception:
    # Some installations split core LangChain into `langchain_core`.
    # Fall back to importing the message classes from there for compatibility.
    from langchain_core.messages import HumanMessage, SystemMessage
import re
from datetime import datetime, timedelta
from pathlib import Path

from models.database import Video, Detection, Summary, Anomaly
from config import settings

class QueryService:
    def __init__(self):
        if settings.GOOGLE_API_KEY:
            self.llm = ChatGoogleGenerativeAI(
                model=settings.MODEL_NAME,
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=0.3,
                convert_system_message_to_human=True
            )
        else:
            self.llm = None

    def encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """
        Encode an image file to base64 format for Gemini Vision API.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded image string with data URI prefix, or None if failed
        """
        try:
            path = Path(image_path)
            if not path.exists():
                print(f"Image file not found: {image_path}")
                return None

            # Read image file
            with open(path, "rb") as image_file:
                image_data = image_file.read()

            # Encode to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')

            # Determine image type from extension
            ext = path.suffix.lower()
            mime_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            mime_type = mime_types.get(ext, 'image/jpeg')

            # Return with data URI prefix
            return f"data:{mime_type};base64,{base64_image}"

        except Exception as e:
            print(f"Error encoding image: {e}")
            return None
    
    def extract_time_range(self, question: str) -> Optional[tuple]:
        """Extract time range from question."""
        # Pattern for HH:MM:SS or MM:SS format
        time_pattern = r'(\d{1,2}:\d{2}(?::\d{2})?)'
        times = re.findall(time_pattern, question)
        
        if len(times) >= 2:
            start_time = self._parse_time_to_seconds(times[0])
            end_time = self._parse_time_to_seconds(times[1])
            return (start_time, end_time)
        elif len(times) == 1:
            # Single time mentioned, assume a 1-minute window
            time_sec = self._parse_time_to_seconds(times[0])
            return (time_sec, time_sec + 60)
        
        return None
    
    def _parse_time_to_seconds(self, time_str: str) -> float:
        """Convert time string to seconds."""
        parts = time_str.split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return 0
    
    def extract_color(self, question: str) -> Optional[str]:
        """Extract color from question."""
        colors = ['red', 'blue', 'green', 'yellow', 'white', 'black', 'gray', 'grey', 'orange', 'purple', 'brown']
        question_lower = question.lower()
        
        for color in colors:
            if color in question_lower:
                return color if color != 'grey' else 'gray'
        
        return None
    
    def extract_object_type(self, question: str) -> Optional[str]:
        """Extract object type from question."""
        object_types = {
            'suitcase': ['suitcase', 'suitcases', 'luggage', 'baggage'],
            'backpack': ['backpack', 'backpacks', 'bag', 'bags'],
            'handbag': ['handbag', 'handbags', 'purse', 'purses'],
            'person': ['person', 'people', 'pedestrian', 'pedestrians', 'owner', 'owners']
        }

        question_lower = question.lower()

        for obj_type, keywords in object_types.items():
            for keyword in keywords:
                if keyword in question_lower:
                    return obj_type

        # Check for general luggage queries
        if any(word in question_lower for word in ['luggage', 'bag', 'abandoned', 'unattended']):
            return 'luggage'

        return None
    
    async def process_query(self, question: str, video_id: Optional[int], db: Session) -> Dict[str, Any]:
        """Process a natural language query with optional vision support."""

        # Extract query parameters
        time_range = self.extract_time_range(question)
        color = self.extract_color(question)
        object_type = self.extract_object_type(question)

        # Build base query
        query = db.query(Detection)

        if video_id:
            query = query.filter(Detection.video_id == video_id)

        # Apply filters based on extracted parameters
        if time_range:
            start_time, end_time = time_range
            query = query.filter(and_(
                Detection.timestamp >= start_time,
                Detection.timestamp <= end_time
            ))

        if color:
            query = query.filter(Detection.color == color)

        if object_type and object_type != 'luggage':
            query = query.filter(Detection.object_class == object_type)
        elif object_type == 'luggage':
            query = query.filter(Detection.object_class.in_(['suitcase', 'backpack', 'handbag']))

        # Get results
        detections = query.all()
        count = len(detections)

        # Get summary if video_id is provided
        summary = None
        unique_luggage_count = 0
        unique_track_ids = set()

        if video_id:
            summary = db.query(Summary).filter(Summary.video_id == video_id).first()

            # Calculate unique luggage count from detections (track_ids)
            for det in detections:
                if det.track_id and det.object_class in ['suitcase', 'backpack', 'handbag']:
                    unique_track_ids.add(det.track_id)
            unique_luggage_count = len(unique_track_ids)

        # Check if query is about owners or abandoned luggage - retrieve images
        owner_images = []
        question_lower = question.lower()
        is_owner_query = any(word in question_lower for word in [
            'owner', 'person', 'who', 'wearing', 'clothes', 'appearance',
            'abandoned', 'left', 'unattended', 'describe', 'color', 'luggage', 'bag'
        ])

        if is_owner_query and video_id:
            # Get anomalies - we'll use the anomaly frame images (first image in email)
            anomalies = db.query(Anomaly).filter(
                Anomaly.video_id == video_id,
                Anomaly.anomaly_type == 'ABANDONED_LUGGAGE'
            ).limit(5).all()  # Limit to 5 images to avoid token limits

            for anomaly in anomalies:
                # Construct path to anomaly frame (the good image showing luggage)
                # Path format: processed/video_{video_id}_anomalies/anomaly_{frame_number:06d}_{timestamp:.1f}s.jpg
                anomaly_frames_dir = settings.PROCESSED_DIR / f"video_{video_id}_anomalies"
                anomaly_frame_path = anomaly_frames_dir / f"anomaly_{anomaly.frame_number:06d}_{anomaly.timestamp:.1f}s.jpg"

                if anomaly_frame_path.exists():
                    owner_images.append({
                        'path': str(anomaly_frame_path),
                        'description': anomaly.description,
                        'luggage_description': anomaly.luggage_description,
                        'timestamp': anomaly.timestamp
                    })

        # Generate answer
        if self.llm:
            answer = await self._generate_llm_answer(
                question, detections, count, summary, owner_images, unique_luggage_count
            )
        else:
            answer = self._generate_simple_answer(question, detections, count, summary)

        # Prepare supporting data
        supporting_data = {
            "count": count,
            "unique_luggage_count": unique_luggage_count if video_id else None,
            "filters_applied": {
                "time_range": time_range,
                "color": color,
                "object_type": object_type
            }
        }
        
        if summary:
            supporting_data["video_summary"] = {
                "total_luggage": summary.total_luggage,
                "total_persons": summary.total_persons,
                "abandoned_count": summary.abandoned_count,
                "luggage_types": summary.luggage_types,
                "color_distribution": summary.color_distribution
            }
        
        # Get sample detections
        sample_detections = detections[:10] if detections else []

        # Compute fallback timestamps for any returned detections where timestamp is
        # missing or zero, using frame_number and the stored video's fps. Do this
        # for both the samples and for the full set when preparing the response.
        if video_id:
            try:
                video_obj = db.query(Video).filter(Video.id == video_id).first()
                fps = getattr(video_obj, 'fps', None) if video_obj else None
                if fps and fps > 0:
                    # Fix sample detections shown in the response
                    for det in sample_detections:
                        if (not det.timestamp) and getattr(det, 'frame_number', None) is not None:
                            det.timestamp = det.frame_number / fps

                    # Also fix all returned detections (relevant_detections) so the
                    # API consumer sees timestamps even if DB rows were 0.0
                    for det in detections:
                        if (not det.timestamp) and getattr(det, 'frame_number', None) is not None:
                            det.timestamp = det.frame_number / fps
            except Exception:
                # Non-fatal: if DB lookup fails, just return detections as-is
                pass
        
        return {
            "answer": answer,
            "supporting_data": supporting_data,
            "relevant_detections": sample_detections
        }
    
    async def _generate_llm_answer(
        self,
        question: str,
        detections: List,
        count: int,
        summary: Optional[Any],
        owner_images: List[Dict] = None,
        unique_luggage_count: int = 0
    ) -> str:
        """Generate answer using LLM with optional vision support."""

        context = f"""
        Question: {question}

        Query Results:
        - Unique luggage items in query: {unique_luggage_count if unique_luggage_count > 0 else 'N/A'}
        - Total detection records: {count}

        Note: Detection records include the same luggage across multiple frames.
        The "unique luggage items" count represents actual distinct luggage pieces.
        """

        if summary:
            context += f"""

        Video Summary:
        - Total luggage detected: {summary.total_luggage}
        - Abandoned luggage: {summary.abandoned_count}
        - Total persons detected: {summary.total_persons}
        - Luggage types: {summary.luggage_types}
        - Color distribution: {summary.color_distribution}
        """

        if detections and len(detections) > 0:
            # Add sample detection details
            context += "\n\nSample detections:\n"
            for det in detections[:5]:
                context += f"- {det.object_class} ({det.color or 'unknown color'}) at {det.timestamp:.1f}s\n"

        # Add information about owner images if available
        if owner_images:
            context += f"\n\nOwner images available: {len(owner_images)} scene(s) captured\n"
            for idx, img_info in enumerate(owner_images, 1):
                context += f"Image {idx}: {img_info.get('description', 'N/A')} at {img_info.get('timestamp', 0):.1f}s\n"

        # System instruction
        system_instruction = (
            "You are a luggage monitoring and security assistant. Answer questions about "
            "luggage detection, abandoned items, and owners clearly and concisely. "
            "Focus on security-relevant information. "
            "When images are provided, analyze them to describe the person and luggage details. "
            "\n\nIMPORTANT: When answering questions about 'how many luggage' or 'total luggage', "
            "ALWAYS use the 'Video Summary: Total luggage detected' count (from tracking), "
            "NOT the 'Total detection records' count (which includes duplicates across frames)."
        )

        # Build message content
        message_content = []

        # Add text content
        combined_text = f"{system_instruction}\n\n{context}\n\nProvide a clear, natural language answer to the question."
        message_content.append({
            "type": "text",
            "text": combined_text
        })

        # Add images if available (for vision queries)
        if owner_images:
            for img_info in owner_images:
                encoded_image = self.encode_image_to_base64(img_info['path'])
                if encoded_image:
                    message_content.append({
                        "type": "image_url",
                        "image_url": encoded_image
                    })

        # Create message with multimodal content
        messages = [HumanMessage(content=message_content)]

        response = await self.llm.ainvoke(messages)
        return response.content
    
    def _generate_simple_answer(self, question: str, detections: List, count: int, summary: Optional[Any]) -> str:
        """Generate simple answer without LLM."""
        
        question_lower = question.lower()
        
        if 'how many' in question_lower:
            # Extract what's being counted
            time_range = self.extract_time_range(question)
            color = self.extract_color(question)
            object_type = self.extract_object_type(question)
            
            answer_parts = []
            
            if count == 0:
                answer_parts.append("No")
            else:
                answer_parts.append(str(count))
            
            if color:
                answer_parts.append(color)
            
            if object_type:
                if count != 1:
                    # Pluralize
                    if object_type == 'person':
                        answer_parts.append('people')
                    else:
                        answer_parts.append(f"{object_type}s")
                else:
                    answer_parts.append(object_type)
            else:
                answer_parts.append("objects")
            
            answer_parts.append("were detected")
            
            if time_range:
                start, end = time_range
                answer_parts.append(f"between {self._seconds_to_time(start)} and {self._seconds_to_time(end)}")
            
            return " ".join(answer_parts) + "."
        
        elif 'total' in question_lower or 'summary' in question_lower:
            if summary:
                return f"Video summary: {summary.total_luggage} luggage items and {summary.total_persons} persons detected. {summary.abandoned_count} abandoned luggage. Luggage types: {summary.luggage_types}"
            else:
                return f"Total detections found: {count}"
        
        else:
            # Default response
            return f"Found {count} matching detections for your query."
    
    def _seconds_to_time(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
