import cv2
import numpy as np
from typing import Optional, Tuple
from sklearn.cluster import KMeans
import webcolors

class ColorDetector:
    def __init__(self):
        # Define better HSV ranges with multiple ranges for complex colors
        self.color_ranges = {
            'red': [
                [(0, 50, 50), (10, 255, 255)],      # Lower red range
                [(170, 50, 50), (180, 255, 255)]    # Upper red range
            ],
            'blue': [
                [(100, 50, 50), (130, 255, 255)]
            ],
            'green': [
                [(40, 50, 50), (80, 255, 255)]
            ],
            'yellow': [
                [(20, 50, 50), (40, 255, 255)]
            ],
            'orange': [
                [(10, 50, 50), (20, 255, 255)]
            ],
            'purple': [
                [(130, 50, 50), (160, 255, 255)]
            ],
            'pink': [
                [(160, 50, 50), (170, 255, 255)]
            ],
            'brown': [
                [(10, 30, 30), (20, 150, 150)]
            ],
            'white': [
                [(0, 0, 200), (180, 30, 255)]
            ],
            'silver': [
                [(0, 0, 150), (180, 20, 200)]
            ],
            'gray': [
                [(0, 0, 50), (180, 20, 150)]
            ],
            'black': [
                [(0, 0, 0), (180, 255, 50)]
            ]
        }
        
        # RGB values for common car colors (for fallback)
        self.rgb_colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'gray': (128, 128, 128),
            'silver': (192, 192, 192),
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'green': (0, 255, 0),
            'yellow': (255, 255, 0),
            'orange': (255, 165, 0),
            'brown': (165, 42, 42),
            'beige': (245, 245, 220),
            'gold': (255, 215, 0)
        }
    
    def detect_color(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[str]:
        """
        Detect the dominant color of a vehicle using multiple methods.
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Ensure bbox is within image bounds
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Crop the vehicle region
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size == 0:
                return None
            
            # Get center region of the vehicle (avoid wheels, windows)
            h_crop, w_crop = cropped.shape[:2]
            center_y1 = h_crop // 4
            center_y2 = 3 * h_crop // 4
            center_x1 = w_crop // 4
            center_x2 = 3 * w_crop // 4
            
            if center_y2 > center_y1 and center_x2 > center_x1:
                center_region = cropped[center_y1:center_y2, center_x1:center_x2]
            else:
                center_region = cropped
            
            # Method 1: K-means clustering for dominant color
            dominant_color_rgb = self._get_dominant_color_kmeans(center_region)
            
            # Method 2: HSV-based detection
            hsv_color = self._detect_color_hsv(center_region)
            
            # Method 3: Check brightness for black/white/gray
            brightness_color = self._detect_by_brightness(center_region, dominant_color_rgb)
            
            # Decision logic
            if brightness_color in ['black', 'white']:
                return brightness_color
            elif brightness_color == 'silver' and hsv_color == 'gray':
                return 'silver'
            elif hsv_color and hsv_color != 'gray':
                return hsv_color
            elif brightness_color:
                return brightness_color
            else:
                # Fallback to nearest named color
                return self._get_nearest_color_name(dominant_color_rgb)
                
        except Exception as e:
            print(f"Error detecting color: {e}")
            return None
    
    def _get_dominant_color_kmeans(self, image: np.ndarray, k: int = 3) -> Tuple[int, int, int]:
        """
        Use K-means clustering to find dominant colors.
        """
        try:
            # Reshape image to be a list of pixels
            pixels = image.reshape((-1, 3))
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get the color with the most pixels assigned to it
            labels = kmeans.labels_
            label_counts = np.bincount(labels)
            dominant_label = label_counts.argmax()
            
            dominant_color = kmeans.cluster_centers_[dominant_label]
            return tuple(map(int, dominant_color))
            
        except Exception:
            # Fallback to mean color
            mean_color = np.mean(image.reshape(-1, 3), axis=0)
            return tuple(map(int, mean_color))
    
    def _detect_color_hsv(self, image: np.ndarray) -> Optional[str]:
        """
        Detect color using HSV ranges.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        color_scores = {}
        total_pixels = hsv.shape[0] * hsv.shape[1]
        
        for color_name, ranges in self.color_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            for color_range in ranges:
                if len(color_range) == 2:
                    lower, upper = color_range
                    partial_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                    mask = cv2.bitwise_or(mask, partial_mask)
            
            score = cv2.countNonZero(mask) / total_pixels
            if score > 0.3:  # At least 30% of pixels match
                color_scores[color_name] = score
        
        if color_scores:
            return max(color_scores, key=color_scores.get)
        return None
    
    def _detect_by_brightness(self, image: np.ndarray, rgb_color: Tuple[int, int, int]) -> Optional[str]:
        """
        Detect black, white, gray, or silver based on brightness and saturation.
        """
        # Convert to grayscale for brightness analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Convert to HSV for saturation analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean_saturation = np.mean(hsv[:, :, 1])
        
        # Get RGB values
        b, g, r = rgb_color
        
        # Calculate color variance (how different are R, G, B values)
        color_variance = np.std([r, g, b])
        
        # Decision logic for achromatic colors
        if mean_brightness < 40:
            return 'black'
        elif mean_brightness > 220 and mean_saturation < 30:
            return 'white'
        elif mean_brightness > 160 and mean_brightness < 220 and color_variance < 10:
            return 'silver'
        elif color_variance < 15 and mean_saturation < 30:
            # Low color variance and low saturation = gray
            if mean_brightness > 100 and mean_brightness < 160:
                return 'gray'
        
        return None
    
    def _get_nearest_color_name(self, rgb_color: Tuple[int, int, int]) -> str:
        """
        Find the nearest named color based on RGB distance.
        """
        b, g, r = rgb_color
        rgb = (r, g, b)
        
        min_distance = float('inf')
        nearest_color = 'unknown'
        
        for color_name, color_rgb in self.rgb_colors.items():
            distance = sum((a - b) ** 2 for a, b in zip(rgb, color_rgb)) ** 0.5
            if distance < min_distance:
                min_distance = distance
                nearest_color = color_name
        
        return nearest_color