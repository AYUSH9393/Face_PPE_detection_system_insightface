
import cv2, time, hashlib
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from threading import Lock
from collections import defaultdict, deque
import torch
from ultralytics import YOLO
import os


class OptimizedPPEDetectionSystem:
    """
    ✅ OPTIMIZED: Enhanced PPE Detection with improved accuracy
    
    Key Improvements:
    1. Multi-confidence tier system for different PPE types
    2. Enhanced spatial verification with distance-based tolerance
    3. Temporal PPE tracking to reduce false negatives
    4. Better occlusion handling
    5. Improved status reporting with detailed metrics
    """

    PPE_CLASSES = {
        'safety_helmet': ['safety_helmet', 'helmet', 'hardhat'],
        'reflective_vest': ['reflective_vest', 'safety_vest', 'vest'],
        'gloves': ['gloves'],
        'boots': ['boots'],
        'safety_goggles': ['safety_goggles', 'goggles'],
        'face_mask': ['face_mask', 'mask'],
        'welding_mask': ['welding_mask'],
        'safety_harness': ['safety_harness', 'harness'],
        'safety_jacket': ['safety_jacket', 'jacket'],
        'apron': ['apron'],
        'hearing_muff': ['hearing_muff', 'ear_muffs'],
        'suit': ['suit']
    }
    
    # ✅ NEW: PPE priority levels (higher = more critical)
    PPE_PRIORITY = {
        'safety_helmet': 10,
        'reflective_vest': 9,
        'safety_harness': 10,
        'welding_mask': 9,
        'safety_goggles': 8,
        'face_mask': 7,
        'gloves': 6,
        'boots': 5,
        'hearing_muff': 6,
        'safety_jacket': 5,
        'apron': 5,
        'suit': 6
    }

    def __init__(self, model_path: str, confidence_threshold: float, db, use_cuda: bool = True):
        print("🦺 Initializing Optimized PPE Detection System...")

        self.db = db
        self.base_confidence_threshold = confidence_threshold
        self.device = 'cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu'

        self.role_ppe_requirements = self._load_ppe_rules_from_db()

        os.environ['YOLO_VERBOSE'] = 'False'
        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.lock = Lock()
        
        # ✅ OPTIMIZED: Multi-tier confidence thresholds based on PPE criticality
        self.ppe_confidence_tiers = {
            'critical': {  # Helmet, harness, welding mask
                'safety_helmet': 0.35,
                'safety_harness': 0.35,
                'welding_mask': 0.40,
            },
            'high': {  # Vest, goggles, face mask
                'reflective_vest': 0.30,
                'safety_goggles': 0.35,
                'face_mask': 0.35,
            },
            'medium': {  # Gloves, boots, hearing protection
                'gloves': 0.25,
                'boots': 0.25,
                'hearing_muff': 0.30,
            },
            'low': {  # Jacket, apron, suit
                'safety_jacket': 0.25,
                'apron': 0.25,
                'suit': 0.25,
            }
        }
        
        # ✅ NEW: Temporal PPE tracking for stability
        self.ppe_temporal_tracking = defaultdict(lambda: defaultdict(lambda: deque(maxlen=10)))
        self.ppe_tracking_lock = Lock()
        
        # ✅ ENHANCED: Distance-adaptive spatial verification tolerances w/ STRICT ANATOMICAL CHECKS
        self.spatial_verification_params = {
            "safety_helmet": {
                "horizontal_tolerance": 1.5,    # Stricter: must be centered on head
                "vertical_range": (-3.0, -0.2), # STRICT: Must be ABOVE face center (negative detections only)
                "min_confidence_boost": 0.05,
            },
            "reflective_vest": {
                "horizontal_tolerance": 3.0,
                "vertical_range": (0.5, 8.0),   # STRICT: Must be BELOW face center (torso)
                "min_confidence_boost": 0.03,
            },
            "safety_goggles": {
                "horizontal_tolerance": 1.2,
                "vertical_range": (-0.8, 0.5),  # Strictly eye level
                "min_confidence_boost": 0.08,
            },
            "face_mask": {
                "horizontal_tolerance": 1.2,
                "vertical_range": (0.0, 1.0),   # Strictly lower face
                "min_confidence_boost": 0.08,
            },
            "welding_mask": {
                "horizontal_tolerance": 1.5,
                "vertical_range": (-1.5, 1.0),
                "min_confidence_boost": 0.07,
            },
            "gloves": {
                "horizontal_tolerance": 6.0,    # Wide tolerance for arms
                "vertical_range": (0.5, 12.0),  # Must be below face
                "min_confidence_boost": 0.02,
            },
            "boots": {
                "horizontal_tolerance": 4.0,
                "vertical_range": (4.0, 15.0),  # Far below face
                "min_confidence_boost": 0.02,
            },
            "safety_harness": {
                "horizontal_tolerance": 3.0,
                "vertical_range": (0.5, 8.0),   # Torso region like vest
                "min_confidence_boost": 0.04,
            },
        }

        print("✅ Optimized PPE Detection System initialized")
        print(f"   Multi-tier confidence system enabled")
        print(f"   Temporal tracking enabled (10-frame buffer)")

    def _load_ppe_rules_from_db(self) -> dict:
        config = self.db.system_config.find_one({"config_type": "ppe_rules"})
        if not config:
            return {
                "default": ["safety_helmet", "reflective_vest"],
                "visitor": ["safety_helmet", "reflective_vest"]
            }
        return config.get("role_rules", {})

    def reload_ppe_rules(self):
        self.role_ppe_requirements = self._load_ppe_rules_from_db()
        print("🔄 PPE rules reloaded")

    def _map_to_ppe_category(self, class_name: str) -> Optional[str]:
        for category, aliases in self.PPE_CLASSES.items():
            if class_name in aliases:
                return category
        return None
    
    def _get_ppe_confidence_threshold(self, category: str) -> float:
        """
        ✅ NEW: Get adaptive confidence threshold for PPE category
        """
        # Check each tier
        for tier_name, tier_thresholds in self.ppe_confidence_tiers.items():
            if category in tier_thresholds:
                return tier_thresholds[category]
        
        # Fallback to base threshold
        return self.base_confidence_threshold

    def detect_ppe(self, frame: np.ndarray) -> List[Dict]:
        """
        ✅ OPTIMIZED: Enhanced PPE detection with adaptive thresholds
        """
        with self.lock:
            # Use lower base confidence, we'll filter by category-specific thresholds
            results = self.model.predict(frame, conf=0.20, verbose=False)
            detections = []
            
            for result in results:
                for box in result.boxes:
                    class_name = self.model.names[int(box.cls[0])].lower()
                    category = self._map_to_ppe_category(class_name)
                    
                    if not category:
                        continue
                    
                    confidence = float(box.conf[0])
                    
                    # ✅ Apply category-specific threshold
                    min_threshold = self._get_ppe_confidence_threshold(category)
                    
                    if confidence < min_threshold:
                        continue
                    
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    detection = {
                        "category": category,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                        "center": {"x": int((x1 + x2) / 2), "y": int((y1 + y2) / 2)},
                        "area": int((x2 - x1) * (y2 - y1)),
                        "priority": self.PPE_PRIORITY.get(category, 5)
                    }
                    
                    detections.append(detection)
            
            return detections
    
    def detect_ppe_batch(self, frames: list) -> List[List[Dict]]:
        """
        ✅ OPTIMIZED: Batch PPE detection with adaptive thresholds
        """
        if not frames:
            return []
        
        resized_frames = [cv2.resize(f, (640, 640)) for f in frames]
        
        with self.lock:
            results = self.model.predict(resized_frames, conf=0.20, verbose=False)
        
        batch_detections = []
        
        for idx, result in enumerate(results):
            detections = []
            orig_h, orig_w = frames[idx].shape[:2]
            sx, sy = orig_w / 640, orig_h / 640
            
            for box in result.boxes:
                class_name = self.model.names[int(box.cls[0])].lower()
                category = self._map_to_ppe_category(class_name)
                
                if not category:
                    continue
                
                confidence = float(box.conf[0])
                
                # ✅ Apply category-specific threshold
                min_threshold = self._get_ppe_confidence_threshold(category)
                
                if confidence < min_threshold:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
                
                detection = {
                    "category": category,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "center": {"x": int((x1+x2)/2), "y": int((y1+y2)/2)},
                    "area": int((x2 - x1) * (y2 - y1)),
                    "priority": self.PPE_PRIORITY.get(category, 5)
                }
                
                detections.append(detection)
            
            batch_detections.append(detections)
        
        return batch_detections

    def verify_ppe_on_person(self, face_bbox: Tuple[int, int, int, int], 
                            ppe_detections: List[Dict], frame_height: int,
                            person_id: str = None) -> List[Dict]:
        """
        ✅ OPTIMIZED: Enhanced spatial verification with temporal tracking
        """
        if not ppe_detections:
            return []
        
        x1, y1, x2, y2 = face_bbox
        face_w, face_h = x2 - x1, y2 - y1
        face_cx, face_cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        verified = []
        
        for ppe in ppe_detections:
            px, py = ppe["center"]["x"], ppe["center"]["y"]
            dx, dy = abs(px - face_cx), py - face_cy
            cat = ppe["category"]
            
            # Get spatial verification parameters
            params = self.spatial_verification_params.get(cat, {
                "horizontal_tolerance": 5.0,
                "vertical_range": (-2.0, 10.0),
                "min_confidence_boost": 0.0
            })
            
            h_tol = params["horizontal_tolerance"]
            v_min, v_max = params["vertical_range"]
            conf_boost = params["min_confidence_boost"]
            
            # ✅ ENHANCED: Distance-based verification
            horizontal_valid = dx < face_w * h_tol
            vertical_valid = face_h * v_min < dy < face_h * v_max
            
            valid = horizontal_valid and vertical_valid
            
            # ✅ NEW: Confidence boost for very close proximity
            if valid:
                # Calculate proximity score (0-1, higher = closer)
                h_proximity = 1.0 - (dx / (face_w * h_tol))
                v_center = (v_min + v_max) / 2
                v_proximity = 1.0 - abs((dy / face_h) - v_center) / ((v_max - v_min) / 2)
                proximity_score = (h_proximity + v_proximity) / 2
                
                # Boost confidence for close proximity
                boosted_confidence = min(ppe["confidence"] + (proximity_score * conf_boost), 1.0)
                ppe["confidence"] = boosted_confidence
                ppe["proximity_score"] = proximity_score
            
            ppe["verified"] = valid
            ppe["spatial_distance"] = {
                "horizontal": dx,
                "vertical": dy,
                "horizontal_ratio": dx / face_w if face_w > 0 else 0,
                "vertical_ratio": dy / face_h if face_h > 0 else 0
            }
            
            verified.append(ppe)
        
        # ✅ NEW: Apply temporal tracking if person_id provided
        if person_id:
            verified = self._apply_temporal_ppe_tracking(person_id, verified)
        
        return verified
    
    def _apply_temporal_ppe_tracking(self, person_id: str, current_detections: List[Dict]) -> List[Dict]:
        """
        ✅ NEW: Apply temporal smoothing to PPE detections
        Reduces false negatives from momentary occlusions
        """
        with self.ppe_tracking_lock:
            # Update tracking history
            current_time = time.time()
            
            for detection in current_detections:
                if detection.get("verified", False):
                    category = detection["category"]
                    self.ppe_temporal_tracking[person_id][category].append({
                        "confidence": detection["confidence"],
                        "timestamp": current_time,
                        "verified": True
                    })
            
            # Check temporal history for each required PPE
            enhanced_detections = current_detections.copy()
            
            # For each PPE category in tracking history
            for category, history in self.ppe_temporal_tracking[person_id].items():
                if not history:
                    continue
                
                # Clean old entries (>2 seconds)
                recent_history = [h for h in history if current_time - h["timestamp"] < 2.0]
                self.ppe_temporal_tracking[person_id][category] = deque(recent_history, maxlen=10)
                
                # Check if this category is currently detected
                currently_detected = any(
                    d["category"] == category and d.get("verified", False)
                    for d in current_detections
                )
                
                # If not currently detected but was recently detected consistently
                if not currently_detected and len(recent_history) >= 5:
                    # Calculate average confidence from recent history
                    avg_confidence = np.mean([h["confidence"] for h in recent_history])
                    
                    # If confidence was consistently high, assume temporary occlusion
                    if avg_confidence > 0.5:
                        # Add a "ghost" detection with reduced confidence
                        ghost_detection = {
                            "category": category,
                            "class_name": category,
                            "confidence": avg_confidence * 0.7,  # Reduced confidence
                            "verified": True,
                            "temporal_inference": True,  # Mark as inferred
                            "bbox": {"x1": 0, "y1": 0, "x2": 0, "y2": 0},
                            "center": {"x": 0, "y": 0},
                            "area": 0,
                            "priority": self.PPE_PRIORITY.get(category, 5)
                        }
                        enhanced_detections.append(ghost_detection)
            
            return enhanced_detections

    def check_person_compliance(self, person_role: str, detected_ppe: List[Dict], 
                               face_detected: bool = True, person_id: str = None) -> Dict:
        """
        ✅ OPTIMIZED: Enhanced compliance checking with detailed metrics
        """
        role = person_role.lower()
        required_ppe = self.role_ppe_requirements.get(role, self.role_ppe_requirements.get("default", []))
        
        effective_ppe = {}  # Changed to dict to store confidence
        
        for d in detected_ppe:
            if not d.get("verified", True):
                continue
            
            detected_category = d["category"]
            
            if detected_category not in required_ppe:
                continue
            
            # Get minimum confidence threshold for this category
            min_conf = self._get_ppe_confidence_threshold(detected_category)
            
            if d["confidence"] >= min_conf:
                # Store highest confidence detection for each category
                if detected_category not in effective_ppe or d["confidence"] > effective_ppe[detected_category]["confidence"]:
                    effective_ppe[detected_category] = {
                        "confidence": d["confidence"],
                        "temporal_inference": d.get("temporal_inference", False),
                        "proximity_score": d.get("proximity_score", 0.0)
                    }
        
        missing = []
        wearing = []
        wearing_details = {}
        
        for ppe_item in required_ppe:
            if ppe_item in effective_ppe:
                wearing.append(ppe_item)
                wearing_details[ppe_item] = effective_ppe[ppe_item]
            else:
                missing.append(ppe_item)
        
        is_compliant = len(missing) == 0 if face_detected else False
        compliance_percentage = (len(wearing) / len(required_ppe) * 100 if required_ppe else 100.0) if face_detected else 0.0
        
        # ✅ NEW: Calculate compliance quality score
        if wearing:
            avg_confidence = np.mean([details["confidence"] for details in wearing_details.values()])
            avg_proximity = np.mean([details.get("proximity_score", 0.5) for details in wearing_details.values()])
            temporal_count = sum(1 for details in wearing_details.values() if details.get("temporal_inference", False))
            
            # Quality score considers confidence, proximity, and temporal inference
            quality_score = (avg_confidence * 0.5 + avg_proximity * 0.3 + 
                           (1.0 - temporal_count / max(len(wearing), 1)) * 0.2)
        else:
            quality_score = 0.0
        
        # ✅ NEW: Prioritize missing PPE by criticality
        missing_with_priority = sorted(
            [(item, self.PPE_PRIORITY.get(item, 5)) for item in missing],
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            "is_compliant": is_compliant,
            "required_ppe": required_ppe,
            "wearing_ppe": wearing,
            "wearing_details": wearing_details,  # ✅ NEW: Detailed info
            "missing_ppe": missing,
            "missing_priority": [item for item, _ in missing_with_priority],  # ✅ NEW: Prioritized list
            "compliance_percentage": compliance_percentage,
            "compliance_quality_score": quality_score,  # ✅ NEW: Quality metric
            "temporal_inference_used": any(d.get("temporal_inference", False) for d in detected_ppe if d.get("verified", False))
        }

    def draw_ppe_detections(self, frame: np.ndarray, ppe_detections: List[Dict], 
                           show_unverified: bool = False) -> np.ndarray:
        """
        ✅ ENHANCED: Draw PPE detections with priority-based coloring
        """
        annotated = frame.copy()
        
        # Priority-based colors (red = critical, yellow = high, green = medium/low)
        priority_colors = {
            10: (0, 0, 255),    # Critical - Red
            9: (0, 100, 255),   # High - Orange
            8: (0, 165, 255),   # Medium-High - Light Orange
            7: (0, 255, 255),   # Medium - Yellow
            6: (0, 255, 128),   # Medium-Low - Yellow-Green
            5: (0, 255, 0),     # Low - Green
        }
        
        for det in ppe_detections:
            if not show_unverified and not det.get('verified', True):
                continue
            
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            
            # Get color based on priority
            priority = det.get('priority', 5)
            color = priority_colors.get(priority, (255, 255, 255))
            
            # ✅ Different line style for temporal inference
            thickness = 2
            if det.get('temporal_inference', False):
                # Dashed line for temporal inference
                self._draw_dashed_rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            else:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # ✅ Enhanced label with confidence and proximity
            label = f"{det['category']}: {det['confidence']:.2f}"
            if det.get('proximity_score'):
                label += f" (P:{det['proximity_score']:.1f})"
            
            # Background for text
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1-text_height-8), (x1+text_width+4, y1), color, -1)
            cv2.putText(annotated, label, (x1+2, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def _draw_dashed_rectangle(self, img, pt1, pt2, color, thickness):
        """Draw dashed rectangle for temporal inference"""
        x1, y1 = pt1
        x2, y2 = pt2
        dash_length = 10
        
        # Top
        for x in range(x1, x2, dash_length * 2):
            cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        # Bottom
        for x in range(x1, x2, dash_length * 2):
            cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
        # Left
        for y in range(y1, y2, dash_length * 2):
            cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        # Right
        for y in range(y1, y2, dash_length * 2):
            cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
    
    def draw_face_label(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int],
                       person_name: str, compliance_info: Dict, is_unknown: bool) -> np.ndarray:
        """
        ✅ ENHANCED: Draw face label with detailed compliance info
        """
        x1, y1, x2, y2 = face_bbox
        
        missing_ppe = compliance_info.get("missing_priority", compliance_info.get("missing_ppe", []))
        quality_score = compliance_info.get("compliance_quality_score", 0.0)
        
        # Status text with quality indicator
        if missing_ppe:
            status_text = f"Missing: {', '.join(missing_ppe[:2])}"  # Show top 2
            if len(missing_ppe) > 2:
                status_text += f" +{len(missing_ppe)-2}"
            status_color = (0, 0, 255)  # Red
        else:
            status_text = f"Compliant (Q:{quality_score:.1f})"
            status_color = (0, 255, 0)  # Green
        
        name_text = "Unknown" if is_unknown else person_name
        
        # Draw with background
        cv2.rectangle(frame, (x1, y1-50), (x2, y1), (0, 0, 0), -1)
        cv2.putText(frame, name_text, (x1+5, y1-28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, status_text, (x1+5, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        # ✅ NEW: Temporal inference indicator
        if compliance_info.get("temporal_inference_used", False):
            cv2.putText(frame, "T", (x2-20, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return frame


# ✅✅✅ OPTIMIZED: ImprovedIntegratedSystem class ✅✅✅
class OptimizedIntegratedSystem:
    """
    ✅ OPTIMIZED: Enhanced integrated system with accurate status reporting
    """
    
    def __init__(self, face_system, ppe_system, db, alert_engine):
        self.face_system = face_system
        self.ppe_system = ppe_system
        self.db = db
        self.alert_engine = alert_engine
        self.unknown_trackers = {}
        self.track_id_counter = defaultdict(int)
        self.track_last_seen = {}
        self.violation_cooldown = {}
        self.VIOLATION_COOLDOWN_SEC = 30
        self.TRACKER_CLEANUP_SEC = 60
        self.last_cleanup = time.time()
        self.identity_cache = {}
        
        # ✅ NEW: Status tracking for frontend
        self.status_history = defaultdict(lambda: deque(maxlen=30))  # 30-frame history
        self.status_lock = Lock()
        
        # ✅ NEW: Attendance tracking (once per day)
        self.attendance_cache = {}  # {person_id: 'YYYY-MM-DD'}
        
        print("🔗 Optimized Integrated Face + PPE System initialized")
        print("   ✅ Enhanced status tracking enabled")
        print("   ✅ Automatic attendance marking enabled")

    def _cleanup_stale_trackers(self):
        now = time.time()
        if now - self.last_cleanup < self.TRACKER_CLEANUP_SEC:
            return
        stale_tracks = [track_id for track_id, last_seen in self.track_last_seen.items() 
                       if now - last_seen > self.TRACKER_CLEANUP_SEC]
        for track_id in stale_tracks:
            del self.track_last_seen[track_id]
            for camera_id in self.unknown_trackers:
                to_remove = [k for k, v in self.unknown_trackers[camera_id].items() if v == track_id]
                for k in to_remove:
                    del self.unknown_trackers[camera_id][k]
        self.last_cleanup = now

    def _calculate_iou(self, box1, box2):
        """Calculate IOU between two boxes (x1,y1,x2,y2)"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[2], box2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / (area1 + area2 - intersection + 1e-6)

    def _assign_track_id(self, camera_id: str, face_bbox: Tuple) -> str:
        """
        ✅ OPTIMIZED: Robust IoU Tracking
        Matches current face to previous tracks based on overlap
        """
        self._cleanup_stale_trackers()
        if camera_id not in self.unknown_trackers:
            self.unknown_trackers[camera_id] = {} # {track_id: {'bbox': bbox, 'last_seen': time}}

        # Find best match among existing tracks
        best_iou = 0
        best_track_id = None
        
        current_tracks = self.unknown_trackers[camera_id]
        
        for track_id, data in current_tracks.items():
            iou = self._calculate_iou(face_bbox, data['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_track_id = track_id
        
        # Threshold for matching (0.3 allows for movement)
        if best_iou > 0.3:
            # Update existing track
            current_tracks[best_track_id]['bbox'] = face_bbox
            current_tracks[best_track_id]['last_seen'] = time.time()
            self.track_last_seen[best_track_id] = time.time()
            return best_track_id
            
        # Create new track
        self.track_id_counter[camera_id] += 1
        new_track_id = f"visitor_{self.track_id_counter[camera_id]:02d}"
        
        current_tracks[new_track_id] = {
            'bbox': face_bbox,
            'last_seen': time.time()
        }
        self.track_last_seen[new_track_id] = time.time()
        return new_track_id

    def _should_log_violation(self, camera_id: str, person_id: str, track_id: str = None) -> bool:
        key_id = track_id if track_id and person_id.startswith("visitor_") else person_id
        cooldown_key = f"{camera_id}:{key_id}"
        now = time.time()
        last_violation = self.violation_cooldown.get(cooldown_key, 0)
        if now - last_violation < self.VIOLATION_COOLDOWN_SEC:
            return False
        self.violation_cooldown[cooldown_key] = now
        return True

    def _mark_attendance_if_needed(self, person_id: str, camera_id: str):
        """
        ✅ NEW: Mark attendance only once per day per person
        """
        if not person_id or person_id == "unknown" or person_id.startswith("visitor_"):
            return

        today_str = datetime.utcnow().strftime('%Y-%m-%d')
        
        # Check cache first (fast)
        if self.attendance_cache.get(person_id) == today_str:
            return

        # Mark in DB
        log_id = f"ATTENDANCE_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # We run this in a separate thread to not block inference?
            # For now, it's a quick DB update, so standard call is fine.
            # But improved safety: check if db call succeeds
            success = self.db.mark_attendance(person_id, camera_id, log_id)
            if success:
                self.attendance_cache[person_id] = today_str
                print(f"✅ Attendance marked for {person_id} on {today_str}")
        except Exception as e:
            print(f"❌ Failed to mark attendance: {e}")

    def process_frames_with_ppe_batch(self, frames: list, camera_ids: list, run_face: bool = True):
        """
        ✅ OPTIMIZED: Enhanced batch processing with accurate status tracking
        """
        assert len(frames) == len(camera_ids)
        timestamp = datetime.utcnow()
        face_results_batch = []
        now_ts = time.time()
        
        # Process faces
        for idx, (frame, camera_id) in enumerate(zip(frames, camera_ids)):
            try:
                faces = self.face_system.process_frame(frame, camera_id, store_logs=False)
                for fr in faces:
                    if fr["person_id"] and fr["person_id"] != "unknown":
                        person = self.db.get_person(fr["person_id"])
                        if person:
                            bbox_str = f"{fr['box'][0]}_{fr['box'][1]}"
                            self.identity_cache[bbox_str] = {
                                "person_id": fr["person_id"],
                                "person_name": fr["person_name"],
                                "role": person.get("role", "default"),
                                "last_seen": now_ts
                            }
                face_results_batch.append(faces)
            except Exception as e:
                print(f"❌ Face processing error for {camera_id}: {e}")
                face_results_batch.append([])
        
        # Detect PPE
        try:
            ppe_batch = self.ppe_system.detect_ppe_batch(frames)
        except Exception as e:
            print(f"❌ PPE batch detection error: {e}")
            ppe_batch = [[] for _ in frames]
        
        # Process compliance
        results_batch = []
        for idx in range(len(frames)):
            camera_id = camera_ids[idx]
            frame = frames[idx]
            face_results = face_results_batch[idx] if idx < len(face_results_batch) else []
            ppe_detections = ppe_batch[idx] if idx < len(ppe_batch) else []
            compliance_results = []
            
            for fr in face_results:
                face_bbox = fr["box"]
                person_id = fr.get("person_id")
                person_name = fr.get("person_name")
                
                # Check known identity cache
                bbox_str = f"{face_bbox[0]}_{face_bbox[1]}"
                cached = self.identity_cache.get(bbox_str)
                
                track_id = None
                
                if cached:
                    person_id = cached["person_id"]
                    person_name = cached["person_name"]
                    role = cached["role"]
                    is_unknown = False
                elif person_id and person_id != "unknown":
                    person = self.db.get_person(person_id)
                    role = person.get("role", "default") if person else "visitor"
                    is_unknown = False
                else:
                    # ✅ OPTIMIZED: Use robust IoU tracking for unknowns
                    track_id = self._assign_track_id(camera_id, face_bbox)
                    person_id = track_id
                    person_name = f"Unknown ({track_id})"
                    role = "visitor"
                    is_unknown = True
                
                # ✅ NEW: Mark attendance for known persons
                if not is_unknown:
                    self._mark_attendance_if_needed(person_id, camera_id)
                
                # ✅ OPTIMIZED: Enhanced PPE verification with person tracking
                verified_ppe = self.ppe_system.verify_ppe_on_person(
                    face_bbox, ppe_detections, frame.shape[0], person_id=person_id
                )
                
                # ✅ OPTIMIZED: Enhanced compliance check
                compliance = self.ppe_system.check_person_compliance(
                    role, verified_ppe, face_detected=True, person_id=person_id
                )
                
                result = {
                    "camera_id": camera_id,
                    "person_id": person_id,
                    "person_name": person_name,
                    "role": role,
                    "face_bbox": face_bbox,
                    "face_confidence": fr.get("confidence", 0.0),
                    "face_size": fr.get("face_size", (0, 0)),  # ✅ NEW
                    "detector_type": fr.get("detector", "unknown"),  # ✅ NEW
                    "ppe_detections": verified_ppe,
                    "compliance": compliance,
                    "is_violation": not compliance["is_compliant"],
                    "is_unknown": is_unknown,
                    "track_id": track_id if is_unknown else person_id,
                    "timestamp": timestamp,
                }
                
                # ✅ NEW: Update status history for frontend
                with self.status_lock:
                    status_key = f"{camera_id}:{person_id}"
                    self.status_history[status_key].append({
                        "timestamp": now_ts,
                        "is_compliant": compliance["is_compliant"],
                        "compliance_percentage": compliance["compliance_percentage"],
                        "compliance_quality": compliance.get("compliance_quality_score", 0.0),
                        "missing_ppe": compliance["missing_ppe"],
                        "wearing_ppe": compliance["wearing_ppe"]
                    })
                
                # Log violations
                if result["is_violation"]:
                    if self._should_log_violation(camera_id, result["person_id"], result.get("track_id")):
                        self._log_ppe_violation(
                            person_id=result["person_id"],
                            person_name=result["person_name"],
                            role=result["role"],
                            camera_id=camera_id,
                            compliance=compliance,
                            face_bbox=face_bbox,
                            is_unknown=is_unknown,
                            track_id=result["track_id"]
                        )
                
                compliance_results.append(result)
            
            results_batch.append({
                "face_results": face_results,
                "ppe_detections": ppe_detections,
                "compliance_results": compliance_results,
                "timestamp": timestamp,
                "frame_quality_metrics": self._calculate_frame_metrics(compliance_results)  # ✅ NEW
            })
        
        # Cleanup stale cache
        now = time.time()
        stale_keys = [k for k in list(self.identity_cache.keys()) 
                     if now - self.identity_cache[k]["last_seen"] > 30]
        for k in stale_keys:
            del self.identity_cache[k]
        
        return results_batch
    
    def _calculate_frame_metrics(self, compliance_results: List[Dict]) -> Dict:
        """
        ✅ NEW: Calculate frame-level quality metrics for status panel
        """
        if not compliance_results:
            return {
                "total_persons": 0,
                "compliant_count": 0,
                "violation_count": 0,
                "unknown_count": 0,
                "avg_compliance_percentage": 0.0,
                "avg_quality_score": 0.0
            }
        
        total = len(compliance_results)
        compliant = sum(1 for r in compliance_results if r["compliance"]["is_compliant"])
        violations = sum(1 for r in compliance_results if r["is_violation"])
        unknown = sum(1 for r in compliance_results if r["is_unknown"])
        
        avg_compliance = np.mean([r["compliance"]["compliance_percentage"] for r in compliance_results])
        avg_quality = np.mean([r["compliance"].get("compliance_quality_score", 0.0) for r in compliance_results])
        
        return {
            "total_persons": total,
            "compliant_count": compliant,
            "violation_count": violations,
            "unknown_count": unknown,
            "avg_compliance_percentage": avg_compliance,
            "avg_quality_score": avg_quality
        }
    
    def get_status_summary(self, camera_id: str = None) -> Dict:
        """
        ✅ NEW: Get comprehensive status summary for frontend
        """
        with self.status_lock:
            if camera_id:
                # Filter by camera
                relevant_keys = [k for k in self.status_history.keys() if k.startswith(f"{camera_id}:")]
            else:
                relevant_keys = list(self.status_history.keys())
            
            if not relevant_keys:
                return {
                    "active_persons": 0,
                    "compliant_persons": 0,
                    "violation_persons": 0,
                    "avg_compliance": 0.0,
                    "avg_quality": 0.0,
                    "status_details": []
                }
            
            status_details = []
            compliant_count = 0
            violation_count = 0
            
            for key in relevant_keys:
                history = self.status_history[key]
                if not history:
                    continue
                
                # Get most recent status
                recent = list(history)[-1]
                cam_id, person_id = key.split(":", 1)
                
                if recent["is_compliant"]:
                    compliant_count += 1
                else:
                    violation_count += 1
                
                status_details.append({
                    "camera_id": cam_id,
                    "person_id": person_id,
                    "is_compliant": recent["is_compliant"],
                    "compliance_percentage": recent["compliance_percentage"],
                    "compliance_quality": recent["compliance_quality"],
                    "missing_ppe": recent["missing_ppe"],
                    "wearing_ppe": recent["wearing_ppe"],
                    "last_update": recent["timestamp"]
                })
            
            avg_compliance = np.mean([s["compliance_percentage"] for s in status_details]) if status_details else 0.0
            avg_quality = np.mean([s["compliance_quality"] for s in status_details]) if status_details else 0.0
            
            return {
                "active_persons": len(status_details),
                "compliant_persons": compliant_count,
                "violation_persons": violation_count,
                "avg_compliance": avg_compliance,
                "avg_quality": avg_quality,
                "status_details": status_details
            }

    def _log_ppe_violation(self, person_id: str, person_name: str, role: str, camera_id: str,
                          compliance: Dict, face_bbox: Tuple, is_unknown: bool = False, track_id: str = None):
        """
        ✅ ENHANCED: Log violations with detailed compliance metrics
        """
        try:
            camera = self.db.get_camera(camera_id)
            violation_log = {
                'log_type': 'ppe_violation',
                'person_id': person_id,
                'person_name': person_name,
                'role': role,
                'is_unknown_person': is_unknown,
                'track_id': track_id,
                'camera_id': camera_id,
                'camera_name': camera['name'] if camera else camera_id,
                'location': camera['location'] if camera else 'Unknown',
                'missing_ppe': compliance['missing_ppe'],
                'missing_ppe_priority': compliance.get('missing_priority', compliance['missing_ppe']),  # ✅ NEW
                'required_ppe': compliance['required_ppe'],
                'wearing_ppe': compliance['wearing_ppe'],
                'wearing_details': compliance.get('wearing_details', {}),  # ✅ NEW
                'compliance_percentage': compliance['compliance_percentage'],
                'compliance_quality_score': compliance.get('compliance_quality_score', 0.0),  # ✅ NEW
                'temporal_inference_used': compliance.get('temporal_inference_used', False),  # ✅ NEW
                'face_bbox': {'x1': int(face_bbox[0]), 'y1': int(face_bbox[1]), 
                             'x2': int(face_bbox[2]), 'y2': int(face_bbox[3])},
                'timestamp': datetime.utcnow(),
                'is_alert': True,
                'alert_sent': False,
                'resolved': False
            }
            result = self.db.recognition_logs.insert_one(violation_log)
            camera_name = camera['name'] if camera else camera_id
            print(f"🚨 Violation logged: {person_name} @ {camera_name} ({camera_id}) - Quality: {compliance.get('compliance_quality_score', 0.0):.2f}")
            try:
                self.alert_engine.handle_violation(violation_log)
                self.db.recognition_logs.update_one({"_id": result.inserted_id}, {"$set": {"alert_sent": True}})
            except Exception as e:
                print(f"⚠️ Alert trigger failed: {e}")
        except Exception as e:
            print(f"❌ Failed to log violation: {e}")

    def draw_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """
        ✅ ENHANCED: Draw results with quality indicators
        """
        annotated = frame.copy()
        annotated = self.ppe_system.draw_ppe_detections(annotated, results['ppe_detections'], show_unverified=False)
        
        for cr in results['compliance_results']:
            x1, y1, x2, y2 = cr['face_bbox']
            color = (0, 0, 255) if cr['is_violation'] else ((255, 165, 0) if cr['is_unknown'] else (0, 255, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            annotated = self.ppe_system.draw_face_label(
                annotated, (x1, y1, x2, y2), 
                cr['person_name'], cr['compliance'], cr['is_unknown']
            )
        
        # ✅ NEW: Draw frame metrics
        metrics = results.get('frame_quality_metrics', {})
        h = annotated.shape[0]
        
        cv2.putText(annotated, f"Persons: {metrics.get('total_persons', 0)}", 
                   (10, h-65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, f"Compliant: {metrics.get('compliant_count', 0)}", 
                   (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        violations = metrics.get('violation_count', 0)
        cv2.putText(annotated, f"Violations: {violations}", 
                   (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if violations else (0, 255, 0), 2)
        
        return annotated

    def process_single_frame(self, frame: np.ndarray, camera_id: str) -> Dict:
        """Process single frame (for API endpoint)"""
        results = self.process_frames_with_ppe_batch([frame], [camera_id], run_face=True)
        if results:
            result = results[0]
            result['processing_time_ms'] = 0
            return result
        return None
