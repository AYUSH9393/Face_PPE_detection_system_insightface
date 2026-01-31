
import cv2, time, hashlib
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from threading import Lock
from collections import defaultdict, deque
import torch
from ultralytics import YOLO
import os
from ppe_color_detector import PPEColorDetector


class EnhancedPPEDetectionSystem:
    """
    Enhanced PPE Detection System with spatial verification
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

    def __init__(self, model_path: str, confidence_threshold: float, db, use_cuda: bool = True):
        print("🦺 Initializing Enhanced PPE Detection System...")

        self.db = db
        self.confidence_threshold = confidence_threshold
        self.device = 'cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu'

        self.role_ppe_requirements = self._load_ppe_rules_from_db()

        os.environ['YOLO_VERBOSE'] = 'False'
        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.lock = Lock()

        self.role_tolerance = {
            "default": {
                "safety_helmet": 0.30,
                "reflective_vest": 0.30,
                "gloves": 0.25,
                "boots": 0.25,
                "safety_goggles": 0.30,
                "face_mask": 0.30,
                "welding_mask": 0.35,
                "safety_harness": 0.30,
            },
            "worker": {
                "safety_helmet": 0.30,
                "reflective_vest": 0.30,
                "gloves": 0.25,
                "boots": 0.25,
            },
            "visitor": {
                "safety_helmet": 0.35,
                "reflective_vest": 0.35,
            }
        }

        # Initialize color detector
        try:
            self.color_detector = PPEColorDetector(db)
        except Exception as e:
            print(f"⚠️ Color detector initialization failed: {e}")
            self.color_detector = None

        print("✅ Enhanced PPE Detection System initialized")

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
        if self.color_detector:
            self.color_detector.reload_config()
        print("🔄 PPE rules and color configuration reloaded")

    def _map_to_ppe_category(self, class_name: str) -> Optional[str]:
        for category, aliases in self.PPE_CLASSES.items():
            if class_name in aliases:
                return category
        return None

    def detect_ppe(self, frame: np.ndarray) -> List[Dict]:
        with self.lock:
            results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False)
            detections = []
            for result in results:
                for box in result.boxes:
                    class_name = self.model.names[int(box.cls[0])].lower()
                    category = self._map_to_ppe_category(class_name)
                    if not category:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append({
                        "category": category,
                        "class_name": class_name,
                        "confidence": float(box.conf[0]),
                        "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                        "center": {"x": int((x1 + x2) / 2), "y": int((y1 + y2) / 2)}
                    })
            return detections
    
    def detect_ppe_batch(self, frames: list):
        if not frames:
            return []
        resized_frames = [cv2.resize(f, (640, 640)) for f in frames]
        with self.lock:
            results = self.model.predict(resized_frames, conf=self.confidence_threshold, verbose=False)
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
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
                detections.append({
                    "category": category,
                    "class_name": class_name,
                    "confidence": float(box.conf[0]),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "center": {"x": int((x1+x2)/2), "y": int((y1+y2)/2)}
                })
            batch_detections.append(detections)
        return batch_detections

    def verify_ppe_on_person(self, face_bbox: Tuple[int, int, int, int], 
                            ppe_detections: List[Dict], frame_height: int) -> List[Dict]:
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
            valid = False
            if cat == "safety_helmet":
                valid = dx < face_w * 3.5 and -face_h * 3.0 < dy < face_h * 2.0
            elif cat == "reflective_vest":
                valid = dx < face_w * 4.0 and -face_h * 1.0 < dy < face_h * 7.0
            elif cat in ("safety_goggles", "face_mask", "welding_mask"):
                valid = dx < face_w * 2.5 and abs(dy) < face_h * 1.5
            elif cat in ("gloves", "boots"):
                valid = dx < face_w * 5.0 and abs(dy) < face_h * 10.0
            else:
                valid = dx < face_w * 5.0 and abs(dy) < face_h * 8.0
            ppe["verified"] = valid
            verified.append(ppe)
        return verified
    
    def verify_ppe_on_person_with_color(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int],
                                       ppe_detections: List[Dict], frame_height: int, role: str) -> List[Dict]:
        """
        Verify PPE on person with color validation
        
        Args:
            frame: Input frame for color detection
            face_bbox: Face bounding box
            ppe_detections: List of detected PPE items
            frame_height: Frame height
            role: Person's role for color validation
            
        Returns:
            List of verified PPE items with color information
        """
        # First do spatial verification
        verified_ppe = self.verify_ppe_on_person(face_bbox, ppe_detections, frame_height)
        
        # Then add color checking if color detector is available
        if self.color_detector and self.color_detector.enable_color_checking:
            for ppe in verified_ppe:
                ppe = self.color_detector.check_ppe_with_color(frame, ppe, role)
        else:
            # Add default color info if color checking is disabled
            for ppe in verified_ppe:
                ppe["color_info"] = {
                    "color_valid": True,
                    "detected_color": None,
                    "allowed_colors": [],
                    "color_checking_enabled": False
                }
        
        return verified_ppe

    def check_person_compliance(self, person_role: str, detected_ppe: List[Dict], 
                               face_detected: bool = True) -> Dict:
        role = person_role.lower()
        required_ppe = self.role_ppe_requirements.get(role, self.role_ppe_requirements.get("default", []))
        effective_ppe = set()
        wrong_color_ppe = []  # Track PPE with wrong colors
        role_tol = self.role_tolerance.get(role, self.role_tolerance["default"])
        
        for d in detected_ppe:
            if not d.get("verified", True):
                continue
            detected_category = d["category"]
            if detected_category not in required_ppe:
                continue
            min_conf = role_tol.get(detected_category, 0.25)
            if d["confidence"] >= min_conf:
                # Check color validity if color info is available
                color_info = d.get("color_info", {})
                if color_info.get("color_checking_enabled", False):
                    if color_info.get("color_valid", True):
                        effective_ppe.add(detected_category)
                    else:
                        # PPE detected but wrong color
                        wrong_color_ppe.append({
                            "category": detected_category,
                            "detected_color": color_info.get("detected_color"),
                            "allowed_colors": color_info.get("allowed_colors", [])
                        })
                else:
                    # Color checking disabled, count as valid
                    effective_ppe.add(detected_category)
        
        missing = [p for p in required_ppe if p not in effective_ppe]
        wearing = [p for p in required_ppe if p in effective_ppe]
        is_compliant = len(missing) == 0 and len(wrong_color_ppe) == 0 if face_detected else False
        compliance_percentage = (len(wearing) / len(required_ppe) * 100 if required_ppe else 100.0) if face_detected else 0.0
        
        return {
            "is_compliant": is_compliant,
            "required_ppe": required_ppe,
            "wearing_ppe": wearing,
            "missing_ppe": missing,
            "wrong_color_ppe": wrong_color_ppe,
            "compliance_percentage": compliance_percentage
        }

    def draw_ppe_detections(self, frame: np.ndarray, ppe_detections: List[Dict], 
                           show_unverified: bool = False) -> np.ndarray:
        annotated = frame.copy()
        colors = {
            'safety_helmet': (255, 255, 0),
            'reflective_vest': (242, 87, 39),
            'gloves': (255, 0, 0),
            'boots': (255, 165, 0),
            'safety_goggles': (128, 0, 128),
            'face_mask': (255, 192, 203)
        }
        for det in ppe_detections:
            if not show_unverified and not det.get('verified', True):
                continue
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            color = colors.get(det['category'], (255, 255, 255))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{det['category']}: {det['confidence']:.2f}"
            cv2.putText(annotated, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return annotated
    
    def draw_face_label(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int],
                       person_name: str, missing_ppe: list, is_unknown: bool) -> np.ndarray:
        x1, y1, x2, y2 = face_bbox
        status_text = f"Missing: {', '.join(missing_ppe)}" if missing_ppe else "Compliant"
        status_color = (0, 0, 255) if missing_ppe else (0, 255, 0)
        name_text = "Unknown" if is_unknown else person_name
        cv2.putText(frame, name_text, (x1, y1-28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, status_text, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        return frame


# ✅✅✅ CRITICAL: ImprovedIntegratedSystem class ✅✅✅
class ImprovedIntegratedSystem:
    """Enhanced system with proper violation logging"""
    
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
        self.FACE_RECOGNITION_INTERVAL = 0.5
        self.last_face_recognition_time = 0.0
        
        # ✅ NEW: Attendance tracking
        self.attendance_cache = {}
        
        print("🔗 Improved Integrated Face + PPE System initialized")

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

    def _get_bbox_hash(self, bbox: Tuple[int, int, int, int]) -> str:
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2 / 50) * 50
        center_y = int((y1 + y2) / 2 / 50) * 50
        width, height = x2 - x1, y2 - y1
        hash_str = f"{center_x}:{center_y}:{int(width/20)}:{int(height/20)}"
        return hashlib.md5(hash_str.encode()).hexdigest()[:8]

    def _assign_track_id(self, camera_id: str, face_bbox: Tuple) -> str:
        self._cleanup_stale_trackers()
        if camera_id not in self.unknown_trackers:
            self.unknown_trackers[camera_id] = {}
        bbox_hash = self._get_bbox_hash(face_bbox)
        if bbox_hash in self.unknown_trackers[camera_id]:
            track_id = self.unknown_trackers[camera_id][bbox_hash]
            self.track_last_seen[track_id] = time.time()
            return track_id
        self.track_id_counter[camera_id] += 1
        track_id = f"visitor_{self.track_id_counter[camera_id]:02d}"
        self.unknown_trackers[camera_id][bbox_hash] = track_id
        self.track_last_seen[track_id] = time.time()
        return track_id

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
        """Mark attendance only once per day per person"""
        if not person_id or person_id == "unknown" or person_id.startswith("visitor_"):
            return

        today_str = datetime.utcnow().strftime('%Y-%m-%d')
        if self.attendance_cache.get(person_id) == today_str:
            return

        log_id = f"ATTENDANCE_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        try:
            success = self.db.mark_attendance(person_id, camera_id, log_id)
            if success:
                self.attendance_cache[person_id] = today_str
                print(f"✅ Attendance marked for {person_id} on {today_str}")
        except Exception as e:
            print(f"❌ Failed to mark attendance: {e}")

    def process_frames_with_ppe_batch(self, frames: list, camera_ids: list, run_face: bool = True):
        assert len(frames) == len(camera_ids)
        timestamp = datetime.utcnow()
        face_results_batch = []
        now_ts = time.time()
        
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
        
        try:
            ppe_batch = self.ppe_system.detect_ppe_batch(frames)
        except Exception as e:
            print(f"❌ PPE batch detection error: {e}")
            ppe_batch = [[] for _ in frames]
        
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
                track_id = self._get_bbox_hash(face_bbox)
                cached = self.identity_cache.get(track_id)
                
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
                    track_id = self._assign_track_id(camera_id, face_bbox)
                    person_id = track_id
                    person_name = f"Unknown ({track_id})"
                    role = "visitor"
                    is_unknown = True
                
                # ✅ NEW: Mark attendance
                if not is_unknown:
                    self._mark_attendance_if_needed(person_id, camera_id)
                
                verified_ppe = self.ppe_system.verify_ppe_on_person_with_color(frame, face_bbox, ppe_detections, frame.shape[0], role)
                compliance = self.ppe_system.check_person_compliance(role, verified_ppe, face_detected=True)
                
                result = {
                    "camera_id": camera_id,
                    "person_id": person_id,
                    "person_name": person_name,
                    "role": role,
                    "face_bbox": face_bbox,
                    "face_confidence": fr.get("confidence", 0.0),
                    "ppe_detections": verified_ppe,
                    "compliance": compliance,
                    "is_violation": not compliance["is_compliant"],
                    "is_unknown": is_unknown,
                    "track_id": track_id if is_unknown else person_id,
                    "timestamp": timestamp,
                }
                
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
                "timestamp": timestamp
            })
        
        now = time.time()
        stale_keys = [k for k in list(self.identity_cache.keys()) 
                     if now - self.identity_cache[k]["last_seen"] > 30]
        for k in stale_keys:
            del self.identity_cache[k]
        
        return results_batch

    def _log_ppe_violation(self, person_id: str, person_name: str, role: str, camera_id: str,
                          compliance: Dict, face_bbox: Tuple, is_unknown: bool = False, track_id: str = None):
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
                'required_ppe': compliance['required_ppe'],
                'wearing_ppe': compliance['wearing_ppe'],
                'compliance_percentage': compliance['compliance_percentage'],
                'face_bbox': {'x1': int(face_bbox[0]), 'y1': int(face_bbox[1]), 
                             'x2': int(face_bbox[2]), 'y2': int(face_bbox[3])},
                'timestamp': datetime.utcnow(),
                'is_alert': True,
                'alert_sent': False,
                'resolved': False
            }
            result = self.db.recognition_logs.insert_one(violation_log)
            print(f"🚨 Violation logged: {person_name} @ {camera_id}")
            try:
                self.alert_engine.handle_violation(violation_log)
                self.db.recognition_logs.update_one({"_id": result.inserted_id}, {"$set": {"alert_sent": True}})
            except Exception as e:
                print(f"⚠️ Alert trigger failed: {e}")
        except Exception as e:
            print(f"❌ Failed to log violation: {e}")

    def draw_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        annotated = frame.copy()
        annotated = self.ppe_system.draw_ppe_detections(annotated, results['ppe_detections'], show_unverified=False)
        for cr in results['compliance_results']:
            x1, y1, x2, y2 = cr['face_bbox']
            color = (0, 0, 255) if cr['is_violation'] else ((255, 165, 0) if cr['is_unknown'] else (0, 255, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            annotated = self.ppe_system.draw_face_label(annotated, (x1, y1, x2, y2), 
                                                       cr['person_name'], cr['compliance']['missing_ppe'], cr['is_unknown'])
        h = annotated.shape[0]
        cv2.putText(annotated, f"Faces: {len(results['face_results'])}", (10, h-35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        violations = sum(1 for r in results['compliance_results'] if r['is_violation'])
        cv2.putText(annotated, f"Violations: {violations}", (10, h-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if violations else (0, 255, 0), 2)
        return annotated

    def process_single_frame(self, frame: np.ndarray, camera_id: str) -> Dict:
        """Process single frame (for API endpoint)"""
        results = self.process_frames_with_ppe_batch([frame], [camera_id], run_face=True)
        if results:
            result = results[0]
            result['processing_time_ms'] = 0  # Add if needed
            return result
        return None

        return None