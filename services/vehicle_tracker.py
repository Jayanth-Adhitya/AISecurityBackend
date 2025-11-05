import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import cv2
from scipy.spatial.distance import cdist

class VehicleTracker:
    """
    Simple vehicle tracker using IoU (Intersection over Union) and centroid tracking.
    This helps avoid counting the same vehicle multiple times across frames.
    """
    
    def __init__(self, max_lost_frames: int = 10, min_iou: float = 0.3, max_distance: int = 100):
        self.tracks = {}  # Active tracks: {track_id: Track object}
        self.next_track_id = 1
        self.max_lost_frames = max_lost_frames
        self.min_iou = min_iou
        self.max_distance = max_distance
        self.completed_tracks = []  # Tracks that have left the scene
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detection dictionaries with keys:
                       bbox (x1, y1, x2, y2), class, confidence, color
        
        Returns:
            List of detections with added track_id
        """
        # Extract bounding boxes from detections
        detection_bboxes = [d['bbox'] for d in detections]
        
        # Match existing tracks to new detections
        matched_pairs, unmatched_tracks, unmatched_detections = self._match_detections(
            list(self.tracks.values()), detection_bboxes
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched_pairs:
            track_id = list(self.tracks.keys())[track_idx]
            track = self.tracks[track_id]
            detection = detections[det_idx]
            
            track.update(
                bbox=detection['bbox'],
                object_class=detection['class'],
                confidence=detection['confidence'],
                color=detection.get('color')
            )
            detection['track_id'] = track_id
            detection['is_counted'] = track.is_counted
        
        # Mark unmatched tracks as lost
        for track_idx in unmatched_tracks:
            track_id = list(self.tracks.keys())[track_idx]
            self.tracks[track_id].mark_lost()
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            track_id = self._create_new_track(
                bbox=detection['bbox'],
                object_class=detection['class'],
                confidence=detection['confidence'],
                color=detection.get('color')
            )
            detection['track_id'] = track_id
            detection['is_counted'] = False
        
        # Remove dead tracks
        self._remove_dead_tracks()
        
        return detections
    
    def _match_detections(self, tracks: List, detection_bboxes: List) -> Tuple[List, List, List]:
        """
        Match tracks to detections using IoU and centroid distance.
        """
        if len(tracks) == 0 or len(detection_bboxes) == 0:
            return [], list(range(len(tracks))), list(range(len(detection_bboxes)))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detection_bboxes)))
        distance_matrix = np.zeros((len(tracks), len(detection_bboxes)))
        
        for t_idx, track in enumerate(tracks):
            track_bbox = track.bbox
            track_centroid = self._get_centroid(track_bbox)
            
            for d_idx, det_bbox in enumerate(detection_bboxes):
                iou = self._calculate_iou(track_bbox, det_bbox)
                iou_matrix[t_idx, d_idx] = iou
                
                det_centroid = self._get_centroid(det_bbox)
                distance = np.linalg.norm(np.array(track_centroid) - np.array(det_centroid))
                distance_matrix[t_idx, d_idx] = distance
        
        # Find matches using Hungarian algorithm simulation (greedy approach)
        matched_pairs = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_detections = set(range(len(detection_bboxes)))
        
        # Sort by IoU in descending order
        matches = []
        for t_idx in range(len(tracks)):
            for d_idx in range(len(detection_bboxes)):
                if iou_matrix[t_idx, d_idx] > self.min_iou:
                    matches.append((iou_matrix[t_idx, d_idx], t_idx, d_idx))
                elif distance_matrix[t_idx, d_idx] < self.max_distance:
                    # Use distance as fallback if IoU is low but objects are close
                    matches.append((0.1, t_idx, d_idx))  # Lower score for distance-based matches
        
        matches.sort(reverse=True)
        
        for score, t_idx, d_idx in matches:
            if t_idx in unmatched_tracks and d_idx in unmatched_detections:
                matched_pairs.append((t_idx, d_idx))
                unmatched_tracks.remove(t_idx)
                unmatched_detections.remove(d_idx)
        
        return matched_pairs, list(unmatched_tracks), list(unmatched_detections)
    
    def _calculate_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _get_centroid(self, bbox: Tuple) -> Tuple[float, float]:
        """Get centroid of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _create_new_track(self, bbox: Tuple, object_class: str, confidence: float, color: Optional[str]) -> int:
        """Create a new track."""
        track_id = self.next_track_id
        self.next_track_id += 1
        
        self.tracks[track_id] = Track(
            track_id=track_id,
            bbox=bbox,
            object_class=object_class,
            confidence=confidence,
            color=color
        )
        
        return track_id
    
    def _remove_dead_tracks(self):
        """Remove tracks that have been lost for too long."""
        dead_tracks = []
        for track_id, track in self.tracks.items():
            if track.lost_frames > self.max_lost_frames:
                dead_tracks.append(track_id)
                self.completed_tracks.append(track)
        
        for track_id in dead_tracks:
            del self.tracks[track_id]
    
    def get_count_summary(self) -> Dict:
        """Get summary of unique vehicles counted."""
        all_tracks = list(self.tracks.values()) + self.completed_tracks
        
        summary = {
            'total_unique_vehicles': 0,
            'by_class': defaultdict(int),
            'by_color': defaultdict(int)
        }
        
        for track in all_tracks:
            if track.is_counted:
                summary['total_unique_vehicles'] += 1
                summary['by_class'][track.object_class] += 1
                if track.dominant_color:
                    summary['by_color'][track.dominant_color] += 1
        
        return dict(summary)


class Track:
    """Represents a single tracked vehicle."""
    
    def __init__(self, track_id: int, bbox: Tuple, object_class: str, confidence: float, color: Optional[str]):
        self.track_id = track_id
        self.bbox = bbox
        self.object_class = object_class
        self.confidence = confidence
        self.color_history = [color] if color else []
        self.dominant_color = color
        self.lost_frames = 0
        self.age = 1
        self.is_counted = False  # Flag to ensure vehicle is counted only once
        self.positions = [self._get_centroid(bbox)]
        self.last_seen_frame = 0
        
    def update(self, bbox: Tuple, object_class: str, confidence: float, color: Optional[str]):
        """Update track with new detection."""
        self.bbox = bbox
        self.object_class = object_class
        self.confidence = confidence
        self.lost_frames = 0
        self.age += 1
        
        if color:
            self.color_history.append(color)
            # Update dominant color (most frequent in history)
            if len(self.color_history) > 0:
                color_counts = defaultdict(int)
                for c in self.color_history:
                    if c:
                        color_counts[c] += 1
                if color_counts:
                    self.dominant_color = max(color_counts, key=color_counts.get)
        
        self.positions.append(self._get_centroid(bbox))
        
        # Mark as counted after being tracked for at least 3 frames
        if self.age >= 3 and not self.is_counted:
            self.is_counted = True
    
    def mark_lost(self):
        """Mark track as lost for current frame."""
        self.lost_frames += 1
    
    def _get_centroid(self, bbox: Tuple) -> Tuple[float, float]:
        """Get centroid of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)