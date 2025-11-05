import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from ultralytics import YOLO
import math

def to_native(value):
    """Convert numpy types to native Python types."""
    if hasattr(value, 'item'):  # numpy scalar
        return value.item()
    elif isinstance(value, np.ndarray):
        return value.tolist()
    return value

class PoseKeypoints(Enum):
    """COCO pose keypoints indices."""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

@dataclass
class PoseDetection:
    """Represents a detected human pose."""
    bbox: Tuple[int, int, int, int]
    keypoints: np.ndarray  # Shape: (17, 3) - x, y, confidence for each keypoint
    confidence: float
    action: Optional[str] = None
    action_confidence: Optional[float] = None
    is_anomaly: bool = False
    anomaly_type: Optional[str] = None

class PoseDetector:
    def __init__(self, model_path: str = "yolov8m-pose.pt"):
        """Initialize pose detector with YOLOv8 pose model."""
        self.model = YOLO(model_path)
        
        # Thresholds for anomaly detection
        self.fall_angle_threshold = 45  # degrees from vertical
        self.velocity_threshold = 50  # pixels per frame for sudden movement
        self.fight_proximity_threshold = 100  # pixels between people
        
        # Store previous frame poses for temporal analysis
        self.previous_poses = []
        self.pose_history = {}  # Track pose history for each person
        
    def detect_poses(self, frame: np.ndarray) -> List[PoseDetection]:
        """Detect human poses in a frame."""
        results = self.model(frame)
        pose_detections = []
        
        for r in results:
            if r.keypoints is not None and r.keypoints.data.shape[0] > 0:
                keypoints_data = r.keypoints.data
                boxes = r.boxes
                
                for i in range(keypoints_data.shape[0]):
                    keypoints = keypoints_data[i].cpu().numpy()
                    
                    # Get bounding box if available
                    if boxes is not None and i < len(boxes.xyxy):
                        bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                        confidence = float(boxes.conf[i])
                    else:
                        # Calculate bbox from keypoints
                        valid_points = keypoints[keypoints[:, 2] > 0.5][:, :2]
                        if len(valid_points) > 0:
                            x_min, y_min = valid_points.min(axis=0)
                            x_max, y_max = valid_points.max(axis=0)
                            bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
                            confidence = float(np.mean(keypoints[:, 2]))
                        else:
                            continue
                    
                    pose = PoseDetection(
                        bbox=tuple(bbox),
                        keypoints=keypoints,
                        confidence=confidence
                    )
                    
                    # Analyze pose for actions and anomalies
                    self._analyze_pose(pose)
                    pose_detections.append(pose)
        
        return pose_detections
    
    def _analyze_pose(self, pose: PoseDetection):
        """Analyze a single pose for actions and anomalies."""
        keypoints = pose.keypoints
        
        # Check for falling
        is_falling = self._detect_falling(keypoints)
        if is_falling:
            pose.is_anomaly = True
            pose.anomaly_type = "FALLING"
            pose.action = "falling"
            pose.action_confidence = is_falling
            return
        
        # Check for lying down
        is_lying = self._detect_lying_down(keypoints)
        if is_lying > 0.7:
            pose.is_anomaly = True
            pose.anomaly_type = "PERSON_DOWN"
            pose.action = "lying_down"
            pose.action_confidence = is_lying
            return
        
        # Check for running (high velocity)
        is_running = self._detect_running(pose)
        if is_running > 0.7:
            pose.action = "running"
            pose.action_confidence = is_running
            # Running might be anomaly in certain contexts
            if is_running > 0.9:
                pose.is_anomaly = True
                pose.anomaly_type = "SUSPICIOUS_RUNNING"
            return
        
        # Check for arms raised (potential distress or surrender)
        arms_raised = self._detect_arms_raised(keypoints)
        if arms_raised > 0.8:
            pose.action = "arms_raised"
            pose.action_confidence = arms_raised
            pose.is_anomaly = True
            pose.anomaly_type = "DISTRESS_SIGNAL"
            return
        
        # Normal walking or standing
        if self._is_standing(keypoints):
            pose.action = "standing"
            pose.action_confidence = 0.8
        else:
            pose.action = "walking"
            pose.action_confidence = 0.7
    
    def _detect_falling(self, keypoints: np.ndarray) -> float:
        """Detect if a person is falling based on body angle."""
        # Get key body points
        nose = keypoints[PoseKeypoints.NOSE.value]
        left_hip = keypoints[PoseKeypoints.LEFT_HIP.value]
        right_hip = keypoints[PoseKeypoints.RIGHT_HIP.value]
        left_ankle = keypoints[PoseKeypoints.LEFT_ANKLE.value]
        right_ankle = keypoints[PoseKeypoints.RIGHT_ANKLE.value]
        
        # Check if we have enough valid keypoints
        if nose[2] < 0.5 or (left_hip[2] < 0.5 and right_hip[2] < 0.5):
            return 0.0
        
        # Calculate hip center
        if left_hip[2] > 0.5 and right_hip[2] > 0.5:
            hip_center = (left_hip[:2] + right_hip[:2]) / 2
        elif left_hip[2] > 0.5:
            hip_center = left_hip[:2]
        else:
            hip_center = right_hip[:2]
        
        # Calculate body angle from vertical
        body_vector = nose[:2] - hip_center
        angle_from_vertical = abs(math.degrees(math.atan2(body_vector[0], -body_vector[1])))
        
        # Check if person is tilted beyond threshold
        if angle_from_vertical > self.fall_angle_threshold:
            # Calculate confidence based on angle
            confidence = min((angle_from_vertical - self.fall_angle_threshold) / 45.0, 1.0)
            
            # Additional check: head below hips (strong indicator of fall)
            if nose[1] > hip_center[1]:
                confidence = min(confidence + 0.3, 1.0)
            
            return confidence
        
        return 0.0
    
    def _detect_lying_down(self, keypoints: np.ndarray) -> float:
        """Detect if a person is lying down."""
        # Get vertical span of the person
        valid_points = keypoints[keypoints[:, 2] > 0.5][:, :2]
        if len(valid_points) < 5:
            return 0.0
        
        y_min, y_max = valid_points[:, 1].min(), valid_points[:, 1].max()
        x_min, x_max = valid_points[:, 0].min(), valid_points[:, 0].max()
        
        height = y_max - y_min
        width = x_max - x_min
        
        # If width > height, person is likely horizontal (lying down)
        if width > height * 1.5:
            # Check if major keypoints are at similar height
            shoulders_hips = [
                keypoints[PoseKeypoints.LEFT_SHOULDER.value],
                keypoints[PoseKeypoints.RIGHT_SHOULDER.value],
                keypoints[PoseKeypoints.LEFT_HIP.value],
                keypoints[PoseKeypoints.RIGHT_HIP.value]
            ]
            
            valid_sh = [kp for kp in shoulders_hips if kp[2] > 0.5]
            if len(valid_sh) >= 2:
                y_positions = [kp[1] for kp in valid_sh]
                y_variance = np.std(y_positions)
                
                # Low variance in y means they're at similar height (horizontal)
                if y_variance < height * 0.3:
                    return min(1.0, 1.5 - y_variance / (height * 0.3))
        
        return 0.0
    
    def _detect_running(self, pose: PoseDetection) -> float:
        """Detect if a person is running based on pose and history."""
        keypoints = pose.keypoints
        
        # Check stride length (distance between ankles)
        left_ankle = keypoints[PoseKeypoints.LEFT_ANKLE.value]
        right_ankle = keypoints[PoseKeypoints.RIGHT_ANKLE.value]
        
        if left_ankle[2] > 0.5 and right_ankle[2] > 0.5:
            stride_length = np.linalg.norm(left_ankle[:2] - right_ankle[:2])
            
            # Get body height estimate
            nose = keypoints[PoseKeypoints.NOSE.value]
            if nose[2] > 0.5:
                body_height = abs(nose[1] - min(left_ankle[1], right_ankle[1]))
                
                # Large stride relative to body height indicates running
                stride_ratio = stride_length / (body_height + 1e-6)
                if stride_ratio > 0.3:
                    return min(stride_ratio / 0.5, 1.0)
        
        return 0.0
    
    def _detect_arms_raised(self, keypoints: np.ndarray) -> float:
        """Detect if arms are raised above head."""
        left_wrist = keypoints[PoseKeypoints.LEFT_WRIST.value]
        right_wrist = keypoints[PoseKeypoints.RIGHT_WRIST.value]
        nose = keypoints[PoseKeypoints.NOSE.value]
        
        confidence = 0.0
        count = 0
        
        if left_wrist[2] > 0.5 and nose[2] > 0.5:
            if left_wrist[1] < nose[1]:  # Wrist above nose
                confidence += 0.5
                count += 1
        
        if right_wrist[2] > 0.5 and nose[2] > 0.5:
            if right_wrist[1] < nose[1]:  # Wrist above nose
                confidence += 0.5
                count += 1
        
        return confidence
    
    def _is_standing(self, keypoints: np.ndarray) -> bool:
        """Check if person is in standing position."""
        # Simple check: vertical alignment of nose, hips, and ankles
        nose = keypoints[PoseKeypoints.NOSE.value]
        left_hip = keypoints[PoseKeypoints.LEFT_HIP.value]
        right_hip = keypoints[PoseKeypoints.RIGHT_HIP.value]
        
        if nose[2] > 0.5 and (left_hip[2] > 0.5 or right_hip[2] > 0.5):
            # Check if nose is above hips (upright position)
            hip_y = left_hip[1] if left_hip[2] > 0.5 else right_hip[1]
            return nose[1] < hip_y
        
        return False
    
    def detect_group_anomalies(self, poses: List[PoseDetection]) -> List[Dict[str, Any]]:
        """Detect anomalies involving multiple people (fighting, crowding, etc.)."""
        anomalies = []
        
        # Check for fighting (people in close proximity with raised arms or aggressive poses)
        for i, pose1 in enumerate(poses):
            for j, pose2 in enumerate(poses[i+1:], i+1):
                # Calculate distance between people
                center1 = self._get_pose_center(pose1)
                center2 = self._get_pose_center(pose2)
                distance = np.linalg.norm(np.array(center1) - np.array(center2))
                
                if distance < self.fight_proximity_threshold:
                    # Check for aggressive postures
                    aggression_score = 0.0
                    
                    # Check if arms are extended toward each other
                    if self._arms_extended_toward(pose1, center2) or self._arms_extended_toward(pose2, center1):
                        aggression_score += 0.4
                    
                    # Check for raised arms
                    if pose1.action == "arms_raised" or pose2.action == "arms_raised":
                        aggression_score += 0.3
                    
                    # Check for rapid movement
                    if pose1.action == "running" or pose2.action == "running":
                        aggression_score += 0.3
                    
                    if aggression_score > 0.6:
                        anomalies.append({
                            'type': 'POTENTIAL_FIGHT',
                            'severity': 'high' if aggression_score > 0.8 else 'medium',
                            'confidence': to_native(aggression_score),  # Convert to native type
                            'involved_poses': [int(i), int(j)],  # Ensure these are native ints
                            'description': f'Potential altercation detected between two people'
                        })
        
        # Check for crowd forming
        if len(poses) > 5:
            # Calculate density of people
            centers = [self._get_pose_center(p) for p in poses]
            avg_distance = self._calculate_average_distance(centers)
            
            if avg_distance < self.fight_proximity_threshold * 1.5:
                anomalies.append({
                    'type': 'CROWD_FORMING',
                    'severity': 'low',
                    'confidence': 0.7,
                    'involved_poses': list(range(len(poses))),
                    'description': f'Crowd of {len(poses)} people detected'
                })
        
        return anomalies
    
    def _get_pose_center(self, pose: PoseDetection) -> Tuple[float, float]:
        """Get the center point of a pose."""
        x1, y1, x2, y2 = pose.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _arms_extended_toward(self, pose: PoseDetection, target_point: Tuple[float, float]) -> bool:
        """Check if arms are extended toward a target point."""
        keypoints = pose.keypoints
        left_wrist = keypoints[PoseKeypoints.LEFT_WRIST.value]
        right_wrist = keypoints[PoseKeypoints.RIGHT_WRIST.value]
        left_shoulder = keypoints[PoseKeypoints.LEFT_SHOULDER.value]
        right_shoulder = keypoints[PoseKeypoints.RIGHT_SHOULDER.value]
        
        extended = False
        
        # Check left arm
        if left_wrist[2] > 0.5 and left_shoulder[2] > 0.5:
            arm_vector = left_wrist[:2] - left_shoulder[:2]
            to_target = np.array(target_point) - left_shoulder[:2]
            
            # Check if arm points toward target
            cos_angle = np.dot(arm_vector, to_target) / (np.linalg.norm(arm_vector) * np.linalg.norm(to_target) + 1e-6)
            if cos_angle > 0.7:  # ~45 degrees
                extended = True
        
        # Check right arm
        if right_wrist[2] > 0.5 and right_shoulder[2] > 0.5:
            arm_vector = right_wrist[:2] - right_shoulder[:2]
            to_target = np.array(target_point) - right_shoulder[:2]
            
            cos_angle = np.dot(arm_vector, to_target) / (np.linalg.norm(arm_vector) * np.linalg.norm(to_target) + 1e-6)
            if cos_angle > 0.7:
                extended = True
        
        return extended
    
    def _calculate_average_distance(self, points: List[Tuple[float, float]]) -> float:
        """Calculate average distance between all points."""
        if len(points) < 2:
            return float('inf')
        
        total_distance = 0
        count = 0
        
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distance = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else float('inf')