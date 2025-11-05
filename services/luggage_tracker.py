import numpy as np
from typing import Dict, List, Tuple, Optional
from config import settings


class LuggageTracker:
    """
    Tracks luggage items across frames using IoU and centroid tracking.
    Monitors stationary status and maintains luggage state.
    """

    def __init__(self):
        self.tracks = {}  # {track_id: {bbox, centroid, color, class, last_seen, positions}}
        self.next_id = 1
        self.max_lost_frames = 30  # Keep track for 30 frames after disappearing
        self.stationary_threshold = settings.STATIONARY_THRESHOLD
        self.unique_luggage = set()  # Set of luggage IDs that have been counted

    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def calculate_centroid(self, bbox: List[float]) -> Tuple[float, float]:
        """Calculate centroid of a bounding box"""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def calculate_distance(self, centroid1: Tuple[float, float], centroid2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two centroids"""
        return np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)

    def is_stationary(self, track_id: int) -> bool:
        """
        Check if a luggage item has been stationary.
        Returns True if the item hasn't moved significantly.
        """
        if track_id not in self.tracks:
            return False

        track = self.tracks[track_id]
        positions = track.get('positions', [])

        if len(positions) < 2:
            return False

        # Check if the item has moved less than threshold over last few frames
        recent_positions = positions[-10:] if len(positions) > 10 else positions

        if len(recent_positions) < 2:
            return False

        # Calculate total movement
        total_movement = 0
        for i in range(1, len(recent_positions)):
            dist = self.calculate_distance(recent_positions[i-1], recent_positions[i])
            total_movement += dist

        avg_movement = total_movement / (len(recent_positions) - 1)
        return avg_movement < self.stationary_threshold

    def update_tracks(
        self,
        detections: List[Dict],
        frame_number: int
    ) -> List[Dict]:
        """
        Update luggage tracks with new detections.

        Args:
            detections: List of detections with bbox, class, confidence, color
            frame_number: Current frame number

        Returns:
            List of tracked detections with assigned track IDs
        """
        tracked_detections = []

        # Mark all tracks as not seen this frame
        for track_id in self.tracks:
            self.tracks[track_id]['seen_this_frame'] = False

        # Match detections to existing tracks
        for detection in detections:
            bbox = [detection['bbox_x1'], detection['bbox_y1'],
                   detection['bbox_x2'], detection['bbox_y2']]
            centroid = self.calculate_centroid(bbox)
            obj_class = detection['object_class']
            color = detection.get('color', 'unknown')

            # Find best matching track
            best_match = None
            best_iou = 0.15  # Lower threshold - more lenient matching
            best_distance = float('inf')

            for track_id, track in self.tracks.items():
                if track['class'] != obj_class:
                    continue

                # Try IoU matching first
                iou = self.calculate_iou(bbox, track['bbox'])

                if iou > best_iou:
                    best_iou = iou
                    best_match = track_id
                    best_distance = 0

                # Fallback to centroid distance - be more lenient
                if best_match is None or iou > 0.05:  # Very low overlap threshold
                    distance = self.calculate_distance(centroid, track['centroid'])
                    if distance < best_distance and distance < 200:  # Increased to 200 pixels
                        best_match = track_id
                        best_distance = distance
                        best_iou = iou  # Update to track this match

            # Assign track ID
            if best_match is not None:
                track_id = best_match
                # Update existing track
                self.tracks[track_id]['bbox'] = bbox
                self.tracks[track_id]['centroid'] = centroid
                self.tracks[track_id]['last_seen'] = frame_number
                self.tracks[track_id]['seen_this_frame'] = True
                self.tracks[track_id]['positions'].append(centroid)
                # Keep only last 30 positions
                if len(self.tracks[track_id]['positions']) > 30:
                    self.tracks[track_id]['positions'].pop(0)
                # Update color if detected
                if color != 'unknown':
                    self.tracks[track_id]['color'] = color
            else:
                # Create new track
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    'bbox': bbox,
                    'centroid': centroid,
                    'class': obj_class,
                    'color': color,
                    'first_seen': frame_number,
                    'last_seen': frame_number,
                    'seen_this_frame': True,
                    'positions': [centroid]
                }
                self.unique_luggage.add(track_id)

            # Add track ID to detection
            tracked_detection = detection.copy()
            tracked_detection['track_id'] = track_id
            tracked_detection['is_stationary'] = self.is_stationary(track_id)
            tracked_detections.append(tracked_detection)

        # Remove old tracks that haven't been seen
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if not track['seen_this_frame']:
                frames_lost = frame_number - track['last_seen']
                if frames_lost > self.max_lost_frames:
                    tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracks[track_id]

        return tracked_detections

    def get_track_info(self, track_id: int) -> Optional[Dict]:
        """Get information about a specific track"""
        return self.tracks.get(track_id)

    def get_unique_count(self) -> int:
        """Get count of unique luggage items detected"""
        return len(self.unique_luggage)

    def get_luggage_by_class(self) -> Dict[str, int]:
        """Get count of luggage by class (suitcase, backpack, handbag)"""
        counts = {}
        for track_id in self.unique_luggage:
            if track_id in self.tracks:
                cls = self.tracks[track_id]['class']
                counts[cls] = counts.get(cls, 0) + 1
        return counts

    def reset(self):
        """Reset all tracking state"""
        self.tracks = {}
        self.unique_luggage = set()
        self.next_id = 1
