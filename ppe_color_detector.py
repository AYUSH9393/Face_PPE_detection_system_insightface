"""
PPE Color Detection Module
Detects and validates helmet and vest colors based on role requirements
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


class PPEColorDetector:
    """Detects colors of PPE items (helmets and vests) using HSV color space"""
    
    def __init__(self, db):
        """
        Initialize PPE Color Detector
        
        Args:
            db: Database instance for loading color configuration
        """
        self.db = db
        self.color_config = self._load_color_config()
        self.enable_color_checking = self.color_config.get("enable_color_checking", True)
        self.color_match_threshold = self.color_config.get("color_match_threshold", 30)
        
        print(f"🎨 PPE Color Detector initialized (enabled: {self.enable_color_checking})")
    
    def _load_color_config(self) -> dict:
        """Load color configuration from database"""
        config = self.db.system_config.find_one({"config_type": "ppe_color_rules"})
        if not config:
            # Return default configuration
            return {
                "enable_color_checking": False,
                "color_match_threshold": 30,
                "available_colors": {},
                "role_color_requirements": {}
            }
        return config
    
    def reload_config(self):
        """Reload color configuration from database"""
        self.color_config = self._load_color_config()
        self.enable_color_checking = self.color_config.get("enable_color_checking", True)
        self.color_match_threshold = self.color_config.get("color_match_threshold", 30)
        print("🔄 PPE color configuration reloaded")
    
    def detect_dominant_color(self, frame: np.ndarray, bbox: Dict) -> Optional[str]:
        """
        Detect the dominant color in a bounding box region
        
        Args:
            frame: Input frame (BGR format)
            bbox: Bounding box dict with keys x1, y1, x2, y2
            
        Returns:
            Detected color name or None
        """
        if not self.enable_color_checking:
            return None
        
        try:
            # Extract region of interest
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return None
            
            # Convert to HSV color space
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Get available colors from config
            available_colors = self.color_config.get("available_colors", {})
            
            color_matches = {}
            
            # Check each color
            for color_name, color_info in available_colors.items():
                hsv_range = color_info.get("hsv_range")
                if not hsv_range or len(hsv_range) != 2:
                    continue
                
                lower = np.array(hsv_range[0])
                upper = np.array(hsv_range[1])
                
                # Create mask for this color
                mask = cv2.inRange(hsv_roi, lower, upper)
                
                # Calculate percentage of pixels matching this color
                match_percentage = (np.count_nonzero(mask) / mask.size) * 100
                
                if match_percentage > self.color_match_threshold:
                    color_matches[color_name] = match_percentage
            
            # Return color with highest match percentage
            if color_matches:
                dominant_color = max(color_matches, key=color_matches.get)
                return dominant_color
            
            return None
            
        except Exception as e:
            print(f"❌ Color detection error: {e}")
            return None
    
    def validate_ppe_color(self, role: str, ppe_category: str, detected_color: Optional[str]) -> Dict:
        """
        Validate if detected PPE color is allowed for the role
        
        Args:
            role: Person's role
            ppe_category: PPE item category (e.g., 'safety_helmet', 'reflective_vest')
            detected_color: Detected color name
            
        Returns:
            Dict with validation results
        """
        if not self.enable_color_checking or detected_color is None:
            return {
                "color_valid": True,
                "detected_color": detected_color,
                "allowed_colors": [],
                "color_checking_enabled": False
            }
        
        role_requirements = self.color_config.get("role_color_requirements", {})
        role_colors = role_requirements.get(role.lower(), role_requirements.get("default", {}))
        
        allowed_colors = role_colors.get(ppe_category, [])
        
        # If no color requirements defined for this PPE item, allow any color
        if not allowed_colors:
            return {
                "color_valid": True,
                "detected_color": detected_color,
                "allowed_colors": [],
                "color_checking_enabled": True,
                "reason": "No color requirements defined"
            }
        
        color_valid = detected_color in allowed_colors
        
        return {
            "color_valid": color_valid,
            "detected_color": detected_color,
            "allowed_colors": allowed_colors,
            "color_checking_enabled": True,
            "reason": "Color match" if color_valid else f"Wrong color (expected: {', '.join(allowed_colors)})"
        }
    
    def check_ppe_with_color(self, frame: np.ndarray, ppe_detection: Dict, role: str) -> Dict:
        """
        Check PPE item and validate its color
        
        Args:
            frame: Input frame
            ppe_detection: PPE detection dict with bbox and category
            role: Person's role
            
        Returns:
            Enhanced PPE detection dict with color information
        """
        ppe_category = ppe_detection.get("category")
        
        # Only check colors for helmets and vests
        if ppe_category not in ["safety_helmet", "reflective_vest"]:
            ppe_detection["color_info"] = {
                "color_valid": True,
                "detected_color": None,
                "allowed_colors": [],
                "color_checking_enabled": False,
                "reason": "Color checking not applicable for this PPE type"
            }
            return ppe_detection
        
        # Detect color
        detected_color = self.detect_dominant_color(frame, ppe_detection["bbox"])
        
        # Validate color
        color_validation = self.validate_ppe_color(role, ppe_category, detected_color)
        
        ppe_detection["color_info"] = color_validation
        
        return ppe_detection
    
    def get_color_display_info(self, color_name: str) -> Dict:
        """
        Get display information for a color
        
        Args:
            color_name: Name of the color
            
        Returns:
            Dict with display_color hex code and name
        """
        available_colors = self.color_config.get("available_colors", {})
        color_info = available_colors.get(color_name, {})
        
        return {
            "name": color_info.get("name", color_name.capitalize()),
            "display_color": color_info.get("display_color", "#FFFFFF")
        }
    
    def get_role_color_requirements(self, role: str) -> Dict:
        """
        Get color requirements for a specific role
        
        Args:
            role: Role name
            
        Returns:
            Dict mapping PPE items to allowed colors
        """
        role_requirements = self.color_config.get("role_color_requirements", {})
        return role_requirements.get(role.lower(), role_requirements.get("default", {}))
    
    def get_all_available_colors(self) -> List[Dict]:
        """
        Get list of all available colors
        
        Returns:
            List of color dicts with name, display_color, and hsv_range
        """
        available_colors = self.color_config.get("available_colors", {})
        
        colors_list = []
        for color_id, color_info in available_colors.items():
            colors_list.append({
                "id": color_id,
                "name": color_info.get("name", color_id.capitalize()),
                "display_color": color_info.get("display_color", "#FFFFFF"),
                "hsv_range": color_info.get("hsv_range", [])
            })
        
        return colors_list
