import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from config import settings
import os
from datetime import datetime


class OwnerDetector:
    """
    Tracks relationships between people and luggage items.
    Detects when a person leaves luggage unattended.
    """

    def __init__(self):
        self.proximity_threshold = settings.OWNER_PROXIMITY_DISTANCE
        self.luggage_owners = {}  # {luggage_track_id: {"person_track_id": id, "last_seen": frame_num}}
        self.abandoned_luggage = {}  # {luggage_track_id: {"frame_abandoned": num, "owner_id": id}}
        self.alerted_luggage = set()  # Track which luggage has already been alerted

    def calculate_distance(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Euclidean distance between centers of two bounding boxes"""
        center1 = np.array([(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2])
        center2 = np.array([(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2])
        return np.linalg.norm(center1 - center2)

    def update_proximity(
        self,
        frame_number: int,
        luggage_detections: List[Dict],
        person_detections: List[Dict]
    ) -> Dict:
        """
        Update proximity relationships between people and luggage.
        Returns dictionary of luggage items with their proximity status.
        """
        proximity_map = {}

        for luggage in luggage_detections:
            luggage_id = luggage['track_id']
            luggage_box = [luggage['bbox_x1'], luggage['bbox_y1'],
                          luggage['bbox_x2'], luggage['bbox_y2']]

            # Find closest person
            closest_person = None
            min_distance = float('inf')

            for person in person_detections:
                person_box = [person['bbox_x1'], person['bbox_y1'],
                            person['bbox_x2'], person['bbox_y2']]
                distance = self.calculate_distance(luggage_box, person_box)

                if distance < min_distance:
                    min_distance = distance
                    closest_person = person

            # Check if someone is near the luggage
            if closest_person and min_distance <= self.proximity_threshold:
                # Person is near luggage - update owner relationship
                # Use track_id if available, otherwise use a hash of the bbox as identifier
                person_id = closest_person.get('track_id')
                if person_id is None:
                    # Create a simple identifier from bbox for untracked persons
                    person_bbox = [closest_person['bbox_x1'], closest_person['bbox_y1'],
                                  closest_person['bbox_x2'], closest_person['bbox_y2']]
                    person_id = hash(tuple(person_bbox))
                else:
                    person_bbox = [closest_person['bbox_x1'], closest_person['bbox_y1'],
                                  closest_person['bbox_x2'], closest_person['bbox_y2']]

                self.luggage_owners[luggage_id] = {
                    "person_track_id": person_id,
                    "last_seen": frame_number,
                    "person_bbox": person_bbox
                }
                proximity_map[luggage_id] = {
                    "has_owner_nearby": True,
                    "owner_id": person_id,
                    "distance": min_distance
                }

                # Remove from abandoned if it was previously abandoned
                if luggage_id in self.abandoned_luggage:
                    del self.abandoned_luggage[luggage_id]
                # Remove from alerted set (owner has returned)
                if luggage_id in self.alerted_luggage:
                    self.alerted_luggage.discard(luggage_id)

            else:
                # No one near the luggage
                if luggage_id in self.luggage_owners:
                    # Previously had an owner, now unattended
                    owner_info = self.luggage_owners[luggage_id]
                    proximity_map[luggage_id] = {
                        "has_owner_nearby": False,
                        "owner_id": owner_info.get("person_track_id"),
                        "frames_unattended": frame_number - owner_info["last_seen"],
                        "distance": min_distance if closest_person else None
                    }

                    # Mark as abandoned if not already marked
                    if luggage_id not in self.abandoned_luggage:
                        self.abandoned_luggage[luggage_id] = {
                            "frame_abandoned": owner_info["last_seen"],
                            "owner_id": owner_info.get("person_track_id"),
                            "owner_bbox": owner_info.get("person_bbox"),
                            "luggage_bbox": luggage_box  # Store luggage bbox too
                        }
                else:
                    # Never had an owner tracked
                    proximity_map[luggage_id] = {
                        "has_owner_nearby": False,
                        "owner_id": None,
                        "frames_unattended": None
                    }

        return proximity_map

    def get_abandoned_luggage(
        self,
        current_frame: int,
        fps: float,
        abandonment_threshold_seconds: float = None
    ) -> List[Dict]:
        """
        Get list of luggage items that have been abandoned beyond the threshold.
        Only returns NEW abandoned luggage (not already alerted).

        Args:
            current_frame: Current frame number
            fps: Frames per second of the video
            abandonment_threshold_seconds: Threshold in seconds (defaults to config)

        Returns:
            List of abandoned luggage items with details (only new ones)
        """
        if abandonment_threshold_seconds is None:
            abandonment_threshold_seconds = settings.ABANDONMENT_THRESHOLD

        threshold_frames = int(abandonment_threshold_seconds * fps)
        abandoned_items = []

        for luggage_id, info in self.abandoned_luggage.items():
            # Skip if already alerted
            if luggage_id in self.alerted_luggage:
                continue

            frames_abandoned = current_frame - info["frame_abandoned"]

            if frames_abandoned >= threshold_frames:
                # Mark as alerted to prevent duplicates
                self.alerted_luggage.add(luggage_id)

                abandoned_items.append({
                    "luggage_track_id": luggage_id,
                    "owner_id": info.get("owner_id"),
                    "owner_bbox": info.get("owner_bbox"),
                    "luggage_bbox": info.get("luggage_bbox"),  # Add luggage bbox
                    "frame_abandoned": info["frame_abandoned"],
                    "frames_unattended": frames_abandoned,
                    "seconds_unattended": frames_abandoned / fps
                })

        return abandoned_items

    def capture_owner_image(
        self,
        frame: np.ndarray,
        owner_bbox: List[int],
        luggage_bbox: List[int],
        video_id: int,
        luggage_track_id: int
    ) -> Optional[str]:
        """
        Capture and save scene image showing both owner and luggage.

        Args:
            frame: Video frame (numpy array)
            owner_bbox: Bounding box of the owner [x1, y1, x2, y2]
            luggage_bbox: Bounding box of the luggage [x1, y1, x2, y2]
            video_id: Video ID for organizing saved images
            luggage_track_id: Luggage track ID

        Returns:
            Path to saved image file, or None if failed
        """
        try:
            # Ensure processed directory exists
            owner_dir = settings.PROCESSED_DIR / str(video_id) / "owners"
            owner_dir.mkdir(parents=True, exist_ok=True)

            # Calculate bounding box that includes both owner and luggage
            if owner_bbox and luggage_bbox:
                owner_x1, owner_y1, owner_x2, owner_y2 = map(int, owner_bbox)
                luggage_x1, luggage_y1, luggage_x2, luggage_y2 = map(int, luggage_bbox)

                # Get combined bounding box
                x1 = min(owner_x1, luggage_x1)
                y1 = min(owner_y1, luggage_y1)
                x2 = max(owner_x2, luggage_x2)
                y2 = max(owner_y2, luggage_y2)
            elif owner_bbox:
                x1, y1, x2, y2 = map(int, owner_bbox)
            else:
                return None

            # Add padding (20% on each side for context)
            h, w = frame.shape[:2]
            width = x2 - x1
            height = y2 - y1
            padding_x = int(width * 0.3)
            padding_y = int(height * 0.3)

            x1 = max(0, x1 - padding_x)
            y1 = max(0, y1 - padding_y)
            x2 = min(w, x2 + padding_x)
            y2 = min(h, y2 + padding_y)

            scene_crop = frame[y1:y2, x1:x2]

            if scene_crop.size == 0:
                return None

            # Draw bounding boxes on the scene image for clarity
            scene_with_boxes = scene_crop.copy()

            if owner_bbox:
                # Adjust coordinates relative to crop
                rel_owner_x1 = max(0, int(owner_bbox[0]) - x1)
                rel_owner_y1 = max(0, int(owner_bbox[1]) - y1)
                rel_owner_x2 = min(scene_crop.shape[1], int(owner_bbox[2]) - x1)
                rel_owner_y2 = min(scene_crop.shape[0], int(owner_bbox[3]) - y1)
                cv2.rectangle(scene_with_boxes, (rel_owner_x1, rel_owner_y1),
                            (rel_owner_x2, rel_owner_y2), (0, 255, 0), 2)  # Green for owner
                cv2.putText(scene_with_boxes, "Owner", (rel_owner_x1, rel_owner_y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if luggage_bbox:
                # Adjust coordinates relative to crop
                rel_luggage_x1 = max(0, int(luggage_bbox[0]) - x1)
                rel_luggage_y1 = max(0, int(luggage_bbox[1]) - y1)
                rel_luggage_x2 = min(scene_crop.shape[1], int(luggage_bbox[2]) - x1)
                rel_luggage_y2 = min(scene_crop.shape[0], int(luggage_bbox[3]) - y1)
                cv2.rectangle(scene_with_boxes, (rel_luggage_x1, rel_luggage_y1),
                            (rel_luggage_x2, rel_luggage_y2), (0, 0, 255), 2)  # Red for luggage
                cv2.putText(scene_with_boxes, "Luggage", (rel_luggage_x1, rel_luggage_y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scene_luggage_{luggage_track_id}_{timestamp}.jpg"
            filepath = owner_dir / filename

            cv2.imwrite(str(filepath), scene_with_boxes)

            return str(filepath)

        except Exception as e:
            print(f"Error capturing scene image: {e}")
            return None

    def reset(self):
        """Reset tracking state"""
        self.luggage_owners = {}
        self.abandoned_luggage = {}
