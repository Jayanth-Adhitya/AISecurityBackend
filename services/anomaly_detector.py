import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
import cv2

def to_native_type(value):
    """Convert numpy types to native Python types."""
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, np.bool_):
        return bool(value)
    return value

@dataclass
class VehicleState:
    """Represents the state of a vehicle for anomaly detection."""
    track_id: int
    positions: deque  # History of positions
    velocities: deque  # History of velocities
    accelerations: deque  # History of accelerations
    bbox_sizes: deque  # History of bounding box sizes
    last_update_frame: int
    is_stopped: bool = False
    stop_duration: int = 0
    max_velocity: float = 0.0
    sudden_stop_detected: bool = False
    collision_detected: bool = False

class AnomalyDetector:
    def __init__(self, fps: float = 30.0):
        """Initialize anomaly detector for vehicles and pedestrians."""
        self.fps = fps
        self.vehicle_states = {}
        
        # Thresholds for vehicle anomalies
        self.sudden_stop_threshold = 30.0  # pixels/frame^2 deceleration
        self.collision_distance_threshold = 50  # pixels
        self.stopped_duration_threshold = 60  # frames (2 seconds at 30fps)
        self.max_normal_velocity = 100  # pixels/frame
        self.min_velocity_for_moving = 2.0  # pixels/frame
        
        # History settings
        self.max_history_length = 30  # frames
        
        # Anomaly records
        self.detected_anomalies = []
        
    def update_vehicle(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        frame_number: int,
        object_class: str
    ) -> Optional[Dict[str, Any]]:
        """Update vehicle state and check for anomalies."""
        
        # Initialize state if new vehicle
        if track_id not in self.vehicle_states:
            self.vehicle_states[track_id] = VehicleState(
                track_id=track_id,
                positions=deque(maxlen=self.max_history_length),
                velocities=deque(maxlen=self.max_history_length),
                accelerations=deque(maxlen=self.max_history_length),
                bbox_sizes=deque(maxlen=self.max_history_length),
                last_update_frame=frame_number
            )
        
        state = self.vehicle_states[track_id]
        
        # Calculate center position
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        bbox_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        # Update position history
        state.positions.append(center)
        state.bbox_sizes.append(bbox_size)
        
        # Calculate velocity if we have previous position
        if len(state.positions) >= 2:
            velocity = self._calculate_velocity(
                state.positions[-2], 
                state.positions[-1],
                frame_number - state.last_update_frame
            )
            state.velocities.append(velocity)
            state.max_velocity = max(state.max_velocity, velocity)
            
            # Calculate acceleration if we have previous velocity
            if len(state.velocities) >= 2:
                acceleration = self._calculate_acceleration(
                    state.velocities[-2],
                    state.velocities[-1],
                    frame_number - state.last_update_frame
                )
                state.accelerations.append(acceleration)
        
        state.last_update_frame = frame_number
        
        # Check for anomalies
        anomaly = self._check_vehicle_anomalies(state, object_class, frame_number)
        
        return anomaly
    
    def _check_vehicle_anomalies(
        self,
        state: VehicleState,
        object_class: str,
        frame_number: int
    ) -> Optional[Dict[str, Any]]:
        """Check for various vehicle anomalies."""
        
        anomalies = []
        
        # 1. Check for sudden stop
        if len(state.accelerations) >= 2:
            recent_deceleration = min(list(state.accelerations)[-2:])
            if recent_deceleration < -self.sudden_stop_threshold and state.max_velocity > 20:
                if not state.sudden_stop_detected:
                    state.sudden_stop_detected = True
                    anomalies.append({
                        'type': 'SUDDEN_STOP',
                        'severity': 'high',
                        'confidence': 0.8,
                        'description': f'{object_class} made sudden stop',
                        'track_id': to_native_type(state.track_id), 
                        'deceleration': recent_deceleration
                    })
        
        # 2. Check for stopped vehicle in traffic
        if len(state.velocities) > 0:
            current_velocity = state.velocities[-1]
            if current_velocity < self.min_velocity_for_moving:
                state.stop_duration += 1
                if state.stop_duration > self.stopped_duration_threshold:
                    if not state.is_stopped:
                        state.is_stopped = True
                        anomalies.append({
                            'type': 'VEHICLE_STOPPED',
                            'severity': 'medium',
                            'confidence': 0.7,
                            'description': f'{object_class} stopped for extended period',
                            'track_id': to_native_type(state.track_id), 
                            'duration_frames': state.stop_duration
                        })
            else:
                state.stop_duration = 0
                state.is_stopped = False
        
        # 3. Check for erratic movement
        if len(state.velocities) >= 5:
            velocity_std = np.std(list(state.velocities)[-5:])
            if velocity_std > 20:  # High variance in velocity
                anomalies.append({
                    'type': 'ERRATIC_MOVEMENT',
                    'severity': 'medium',
                    'confidence': 0.6,
                    'description': f'{object_class} showing erratic movement',
                    'track_id': state.track_id,
                    'velocity_variance': velocity_std
                })
        
        # 4. Check for sudden size change (potential collision indicator)
        if len(state.bbox_sizes) >= 3:
            size_change_ratio = state.bbox_sizes[-1] / (state.bbox_sizes[-3] + 1e-6)
            if size_change_ratio > 1.5 or size_change_ratio < 0.6:
                anomalies.append({
                    'type': 'SUDDEN_SIZE_CHANGE',
                    'severity': 'high',
                    'confidence': 0.7,
                    'description': 'Sudden vehicle size change detected (possible collision)',
                    'track_id': state.track_id,
                    'size_change_ratio': size_change_ratio
                })
        
        # Return highest severity anomaly if any detected
        if anomalies:
            # Sort by severity
            severity_order = {'high': 0, 'medium': 1, 'low': 2}
            anomalies.sort(key=lambda x: severity_order.get(x['severity'], 3))
            return anomalies[0]
        
        return None
    
    def check_collision(
        self,
        vehicles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Check for potential collisions between vehicles."""
        
        collisions = []
        
        for i, vehicle1 in enumerate(vehicles):
            for vehicle2 in vehicles[i+1:]:
                # Calculate distance between vehicles
                bbox1 = vehicle1['bbox']
                bbox2 = vehicle2['bbox']
                
                # Check for overlap
                if self._check_bbox_overlap(bbox1, bbox2):
                    collisions.append({
                        'type': 'COLLISION',
                        'severity': 'high',
                        'confidence': 0.9,
                        'description': f"Collision detected between {vehicle1.get('class', 'vehicle')} and {vehicle2.get('class', 'vehicle')}",
                        'track_ids': [
                            to_native_type(vehicle1.get('track_id', -1)),
                            to_native_type(vehicle2.get('track_id', -1))
                        ],
                        'overlap_area': to_native_type(self._calculate_overlap_area(bbox1, bbox2))
                    })
                    continue
                
                # Check for near-miss
                distance = self._calculate_bbox_distance(bbox1, bbox2)
                if distance < self.collision_distance_threshold:
                    # Check if vehicles are moving toward each other
                    track_id1 = vehicle1.get('track_id')
                    track_id2 = vehicle2.get('track_id')
                    
                    if track_id1 in self.vehicle_states and track_id2 in self.vehicle_states:
                        state1 = self.vehicle_states[track_id1]
                        state2 = self.vehicle_states[track_id2]
                        
                        if len(state1.velocities) > 0 and len(state2.velocities) > 0:
                            # Check if converging
                            if self._are_converging(state1, state2):
                                collisions.append({
                                    'type': 'NEAR_COLLISION',
                                    'severity': 'medium',
                                    'confidence': 0.7,
                                    'description': 'Near collision detected between vehicles',
                                    'track_ids': [track_id1, track_id2],
                                    'distance': distance
                                })
        
        return collisions
    
    def check_pedestrian_vehicle_interaction(
        self,
        vehicles: List[Dict[str, Any]],
        pedestrians: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Check for dangerous interactions between vehicles and pedestrians."""
        
        interactions = []
        
        for vehicle in vehicles:
            for pedestrian in pedestrians:
                distance = self._calculate_bbox_distance(
                    vehicle['bbox'],
                    pedestrian['bbox']
                )
                
                if distance < self.collision_distance_threshold * 1.5:
                    # Check vehicle velocity
                    track_id = vehicle.get('track_id')
                    if track_id in self.vehicle_states:
                        state = self.vehicle_states[track_id]
                        if len(state.velocities) > 0:
                            velocity = state.velocities[-1]
                            
                            if velocity > 10:  # Vehicle moving toward pedestrian
                                interactions.append({
                                    'type': 'PEDESTRIAN_AT_RISK',
                                    'severity': 'high',
                                    'confidence': 0.8,
                                    'description': 'Vehicle approaching pedestrian dangerously',
                                    'vehicle_track_id': track_id,
                                    'distance': distance,
                                    'vehicle_velocity': velocity
                                })
        
        return interactions
    
    def _calculate_velocity(
        self,
        prev_pos: Tuple[float, float],
        curr_pos: Tuple[float, float],
        frame_diff: int
    ) -> float:
        """Calculate velocity in pixels per frame."""
        if frame_diff == 0:
            return 0.0
        
        distance = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))
        return distance / max(frame_diff, 1)
    
    def _calculate_acceleration(
        self,
        prev_velocity: float,
        curr_velocity: float,
        frame_diff: int
    ) -> float:
        """Calculate acceleration in pixels per frame squared."""
        if frame_diff == 0:
            return 0.0
        
        return (curr_velocity - prev_velocity) / max(frame_diff, 1)
    
    def _check_bbox_overlap(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> bool:
        """Check if two bounding boxes overlap."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        return not (x1_max < x2_min or x2_max < x1_min or 
                   y1_max < y2_min or y2_max < y1_min)
    
    def _calculate_overlap_area(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate overlap area between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        overlap_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        overlap_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        
        return overlap_x * overlap_y
    
    def _calculate_bbox_distance(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate minimum distance between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # If boxes overlap, distance is 0
        if self._check_bbox_overlap(bbox1, bbox2):
            return 0.0
        
        # Calculate distance between closest points
        dx = max(x1_min - x2_max, 0, x2_min - x1_max)
        dy = max(y1_min - y2_max, 0, y2_min - y1_max)
        
        return np.sqrt(dx**2 + dy**2)
    
    def _are_converging(
        self,
        state1: VehicleState,
        state2: VehicleState
    ) -> bool:
        """Check if two vehicles are converging."""
        if len(state1.positions) < 2 or len(state2.positions) < 2:
            return False
        
        # Calculate distance change
        prev_distance = np.linalg.norm(
            np.array(state1.positions[-2]) - np.array(state2.positions[-2])
        )
        curr_distance = np.linalg.norm(
            np.array(state1.positions[-1]) - np.array(state2.positions[-1])
        )
        
        # Converging if distance is decreasing
        return curr_distance < prev_distance * 0.9  # 10% reduction threshold
    
    def cleanup_old_tracks(self, active_track_ids: List[int], current_frame: int):
        """Remove old vehicle states that are no longer tracked."""
        tracks_to_remove = []
        
        for track_id, state in self.vehicle_states.items():
            if track_id not in active_track_ids:
                # Keep for a while in case track reappears
                if current_frame - state.last_update_frame > 60:  # 2 seconds at 30fps
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.vehicle_states[track_id]