"""
PPE Detection System using YOLO
Integrates with Face Recognition for role-based PPE compliance checking
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
from ultralytics import YOLO
from datetime import datetime
from threading import Lock


class PPEDetectionSystem:
    """
    PPE Detection System using YOLOv8
    Detects: helmet, vest, gloves, boots, goggles, mask
    """
    
    # PPE class mapping (what to look for)
    # ✅ OPTIMIZED: Added more keyword variations for better detection
    PPE_CLASSES = {
        'helmet': [
            'helmet', 'hard hat', 'hardhat', 'hard-hat',
            'safety helmet', 'construction helmet', 'hat',
            'head protection', 'hard_hat', 'safety hat'
        ],
        'vest': [
            'vest', 'safety vest', 'reflective vest',
            'hi-vis', 'high-vis', 'visibility vest',
            'safety jacket', 'reflective jacket', 'jacket'
        ],
        'gloves': ['gloves', 'safety gloves', 'hand protection', 'glove'],
        'goggles': ['goggles', 'safety glasses', 'glasses', 'eye protection', 'eyewear'],
        'mask': ['mask', 'face mask', 'respirator', 'face protection'],
        'boots': ['boots', 'safety boots', 'footwear', 'foot protection', 'shoe', 'shoes']
    }
    
    # Role-based PPE requirements
    ROLE_PPE_REQUIREMENTS = {
        'engineer': ['helmet', 'vest'],
        'worker': ['helmet', 'vest', 'gloves'],
        'supervisor': ['helmet', 'vest'],
        'contractor': ['helmet', 'vest', 'gloves', 'boots'],
        'electrician': ['helmet', 'vest', 'gloves', 'goggles'],
        'welder': ['helmet', 'vest', 'gloves', 'goggles', 'mask'],
        'visitor': ['helmet', 'vest'],
        'default': ['helmet', 'vest']  # Default requirements
    }
    
    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.5,
                 use_cuda: bool = True):
        """
        Initialize PPE Detection System
        
        Args:
            model_path: Path to YOLO model (can be yolov8n.pt or custom trained model)
            confidence_threshold: Minimum confidence for detections
            use_cuda: Use GPU if available
        """
        print("🦺 Initializing PPE Detection System...")
        
        self.confidence_threshold = confidence_threshold
        self.device = 'cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu'
        
        print(f"   Device: {self.device}")
        print(f"   Confidence threshold: {confidence_threshold}")
        
        # Load YOLO model
        try:
            print(f"   Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)
            self.model.to(self.device)
            print(f"   ✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"   ❌ Error loading YOLO model: {e}")
            print(f"   Downloading default YOLOv8 model...")
            self.model = YOLO('yolov8n.pt')  # Will auto-download
            self.model.to(self.device)
        
        # Thread safety
        self.lock = Lock()
        
        print("✅ PPE Detection System initialized\n")
    
    def detect_ppe(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect PPE equipment in frame
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            List of detected PPE items with bounding boxes and confidence
        """
        try:
            with self.lock:
                # Run YOLO detection
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                
                ppe_detections = []
                
                for result in results:
                    boxes = result.boxes
                    
                    for box in boxes:
                        # Get detection info
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id].lower()
                        
                        # Map to PPE category
                        ppe_category = self._map_to_ppe_category(class_name)
                        
                        # ✅ DEBUG: Log what YOLO is detecting
                        if ppe_category:
                            print(f"🔍 YOLO: '{class_name}' (conf={confidence:.2f}) → mapped to '{ppe_category}'")
                        
                        if ppe_category:
                            ppe_detections.append({
                                'category': ppe_category,
                                'class_name': class_name,
                                'confidence': confidence,
                                'bbox': {
                                    'x1': int(x1),
                                    'y1': int(y1),
                                    'x2': int(x2),
                                    'y2': int(y2)
                                }
                            })
                
                return ppe_detections
        except Exception as e:
            print(f"❌ Error in PPE detection: {e}")
            return []  # Return empty list on error
    
    def _map_to_ppe_category(self, class_name: str) -> Optional[str]:
        """Map detected class name to PPE category"""
        class_name = class_name.lower()
        
        for category, keywords in self.PPE_CLASSES.items():
            for keyword in keywords:
                if keyword in class_name:
                    return category
        
        return None
    
    def check_person_compliance(self, person_role: str, detected_ppe: List[Dict]) -> Dict:
        """
        Check if person is wearing required PPE based on their role
        
        Args:
            person_role: Person's role (e.g., 'engineer', 'worker')
            detected_ppe: List of detected PPE items
            
        Returns:
            Compliance status dictionary
        """
        # Get required PPE for role
        person_role = person_role.lower()
        required_ppe = self.ROLE_PPE_REQUIREMENTS.get(
            person_role, 
            self.ROLE_PPE_REQUIREMENTS['default']
        )
        
        # Get detected PPE categories
        detected_categories = set(item['category'] for item in detected_ppe)
        
        # Check compliance
        missing_ppe = []
        wearing_ppe = []
        
        for ppe in required_ppe:
            if ppe in detected_categories:
                wearing_ppe.append(ppe)
            else:
                missing_ppe.append(ppe)
        
        is_compliant = len(missing_ppe) == 0
        
        return {
            'is_compliant': is_compliant,
            'required_ppe': required_ppe,
            'wearing_ppe': wearing_ppe,
            'missing_ppe': missing_ppe,
            'detected_ppe': detected_categories,
            'compliance_percentage': (len(wearing_ppe) / len(required_ppe) * 100) if required_ppe else 100
        }
    
    def draw_ppe_detections(self, frame: np.ndarray, ppe_detections: List[Dict]) -> np.ndarray:
        """Draw PPE detection boxes on frame"""
        annotated_frame = frame.copy()
        
        # Define colors for different PPE
        colors = {
            'helmet': (0, 255, 0),      # Green
            'vest': (0, 255, 255),      # Yellow
            'gloves': (255, 0, 0),      # Blue
            'boots': (255, 165, 0),     # Orange
            'goggles': (128, 0, 128),   # Purple
            'mask': (255, 192, 203)     # Pink
        }
        
        for detection in ppe_detections:
            bbox = detection['bbox']
            category = detection['category']
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            color = colors.get(category, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{category.upper()}: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
            
            # Background for text
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - th - 5), 
                         (x1 + tw, y1), 
                         color, cv2.FILLED)
            
            # Text
            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                       font, 0.5, (0, 0, 0), 1)
        
        return annotated_frame
    
    def draw_compliance_status(self, frame: np.ndarray, person_name: str, 
                              role: str, compliance: Dict, position: Tuple[int, int]) -> np.ndarray:
        """
        Draw compliance status on frame
        
        Args:
            frame: Input frame
            person_name: Person's name
            role: Person's role
            compliance: Compliance check result
            position: (x, y) position for status box
        """
        annotated_frame = frame.copy()
        x, y = position
        
        # Determine status color
        if compliance['is_compliant']:
            status_color = (0, 255, 0)  # Green
            status_text = "✓ COMPLIANT"
        else:
            status_color = (0, 0, 255)  # Red
            status_text = "✗ NON-COMPLIANT"
        
        # Create status box
        box_height = 150
        box_width = 300
        
        # Semi-transparent background
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
        
        # Draw border
        cv2.rectangle(annotated_frame, (x, y), (x + box_width, y + box_height), 
                     status_color, 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Person info
        cv2.putText(annotated_frame, f"Name: {person_name}", (x + 10, y + 25),
                   font, 0.5, (255, 255, 255), 1)
        
        cv2.putText(annotated_frame, f"Role: {role.upper()}", (x + 10, y + 45),
                   font, 0.5, (255, 255, 255), 1)
        
        # Status
        cv2.putText(annotated_frame, status_text, (x + 10, y + 70),
                   font, 0.6, status_color, 2)
        
        # Compliance percentage
        percentage = compliance['compliance_percentage']
        cv2.putText(annotated_frame, f"Compliance: {percentage:.0f}%", 
                   (x + 10, y + 95), font, 0.5, (255, 255, 255), 1)
        
        # Missing PPE
        if compliance['missing_ppe']:
            missing_text = f"Missing: {', '.join(compliance['missing_ppe'])}"
            cv2.putText(annotated_frame, missing_text, (x + 10, y + 120),
                       font, 0.4, (0, 0, 255), 1)
        else:
            cv2.putText(annotated_frame, "All PPE Present", (x + 10, y + 120),
                       font, 0.4, (0, 255, 0), 1)
        
        return annotated_frame
    
    @staticmethod
    def add_ppe_requirements(role: str, required_ppe: List[str]):
        """Add or update PPE requirements for a role"""
        PPEDetectionSystem.ROLE_PPE_REQUIREMENTS[role.lower()] = required_ppe
    
    @staticmethod
    def get_role_requirements(role: str) -> List[str]:
        """Get PPE requirements for a specific role"""
        return PPEDetectionSystem.ROLE_PPE_REQUIREMENTS.get(
            role.lower(),
            PPEDetectionSystem.ROLE_PPE_REQUIREMENTS['default']
        )


class IntegratedFaceAndPPESystem:
    """
    Integrated system combining Face Recognition and PPE Detection
    """
    
    def __init__(self, face_system, ppe_system, db, 
                 ppe_search_expansion: float = 2.5,
                 max_ppe_distance: float = 500.0):
        """
        Initialize integrated system
        
        Args:
            face_system: EnhancedFaceRecognitionSystem instance
            ppe_system: PPEDetectionSystem instance
            db: FaceRecognitionDB instance
            ppe_search_expansion: Factor to expand face region when searching for PPE (default: 2.5)
            max_ppe_distance: Maximum distance to associate PPE with a face (default: 500.0 pixels)
        """
        self.face_system = face_system
        self.ppe_system = ppe_system
        self.db = db
        self.ppe_search_expansion = ppe_search_expansion
        self.max_ppe_distance = max_ppe_distance
        
        print("🔗 Integrated Face Recognition + PPE Detection System initialized")
        print(f"   PPE search expansion: {ppe_search_expansion}x")
        print(f"   Max PPE distance: {max_ppe_distance} pixels")
    
    def _calculate_bbox_distance(self, bbox1: Dict, bbox2: Dict) -> float:
        """
        Calculate center-to-center distance between two bounding boxes
        
        Args:
            bbox1: First bounding box dict with keys x1, y1, x2, y2
            bbox2: Second bounding box dict with keys x1, y1, x2, y2
            
        Returns:
            Euclidean distance between bbox centers
        """
        # Calculate centers
        center1_x = (bbox1['x1'] + bbox1['x2']) / 2
        center1_y = (bbox1['y1'] + bbox1['y2']) / 2
        center2_x = (bbox2['x1'] + bbox2['x2']) / 2
        center2_y = (bbox2['y1'] + bbox2['y2']) / 2
        
        # Euclidean distance
        distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
        return distance
    
    def _expand_face_region(self, face_bbox: Tuple, frame_shape: Tuple = None) -> Dict:
        """
        Expand face bounding box to search for PPE in surrounding area
        PPE items (helmet, vest, etc.) are typically near but not overlapping the face
        
        Args:
            face_bbox: Face bounding box as (x1, y1, x2, y2)
            frame_shape: Optional frame shape (height, width) for boundary clipping
            
        Returns:
            Expanded region as dict with keys x1, y1, x2, y2
        """
        x1, y1, x2, y2 = face_bbox
        width = x2 - x1
        height = y2 - y1
        
        # Expand region (especially downward for body PPE like vest, gloves)
        expand_x = width * (self.ppe_search_expansion - 1) / 2
        expand_y_up = height * 0.5  # Less expansion upward (helmet is close to face)
        expand_y_down = height * (self.ppe_search_expansion - 1)  # More expansion downward for body PPE
        
        expanded = {
            'x1': int(x1 - expand_x),
            'y1': int(y1 - expand_y_up),
            'x2': int(x2 + expand_x),
            'y2': int(y2 + expand_y_down)
        }
        
        # Clip to frame boundaries if provided
        if frame_shape is not None:
            height, width = frame_shape[:2]
            expanded['x1'] = max(0, expanded['x1'])
            expanded['y1'] = max(0, expanded['y1'])
            expanded['x2'] = min(width, expanded['x2'])
            expanded['y2'] = min(height, expanded['y2'])
        
        return expanded
    
    def _associate_ppe_to_faces(self, face_results: List[Dict], 
                               ppe_detections: List[Dict],
                               frame_shape: Tuple = None) -> Dict[str, List[Dict]]:
        """
        Associate each PPE item with the nearest face using spatial proximity
        
        This fixes the critical bug where PPE was detected globally but checked
        against all persons equally, causing incorrect compliance results.
        
        Args:
            face_results: List of face detection results with 'person_id' and 'box'
            ppe_detections: List of PPE detections with 'bbox' and 'category'
            frame_shape: Optional frame shape for boundary clipping
            
        Returns:
            Dictionary mapping person_id to list of their associated PPE items
        """
        # Initialize empty PPE lists for each person
        person_ppe_map = {}
        for face in face_results:
            person_id = face.get('person_id')
            if person_id and person_id != 'unknown':
                person_ppe_map[person_id] = []
        
        # If no recognized persons or no PPE, return empty mapping
        if not person_ppe_map or not ppe_detections:
            return person_ppe_map
        
        # For each PPE item, find the closest face
        for ppe in ppe_detections:
            min_distance = float('inf')
            closest_person_id = None
            
            for face_result in face_results:
                person_id = face_result.get('person_id')
                
                # Skip unknown faces
                if not person_id or person_id == 'unknown':
                    continue
                
                # Expand face region to search for PPE
                search_region = self._expand_face_region(face_result['box'], frame_shape)
                
                # Calculate distance between PPE and expanded face region
                distance = self._calculate_bbox_distance(search_region, ppe['bbox'])
                
                # Track closest face
                if distance < min_distance:
                    min_distance = distance
                    closest_person_id = person_id
            
            # Assign PPE to closest person if within max distance threshold
            if closest_person_id and min_distance < self.max_ppe_distance:
                person_ppe_map[closest_person_id].append(ppe)
                # Debug logging
                # print(f"   Assigned {ppe['category']} to {closest_person_id} (distance: {min_distance:.1f}px)")
        
        return person_ppe_map
    
    def process_frame_with_ppe(self, frame: np.ndarray, camera_id: str) -> Dict:
        """
        Process frame with both face recognition and PPE detection
        
        ✅ FIXED: Now properly associates PPE items to specific persons using spatial proximity
        
        Args:
            frame: Input frame (BGR)
            camera_id: Camera identifier
            
        Returns:
            Processing results with compliance status
        """
        detection_start = datetime.now()
        
        # Step 1: Detect and recognize faces
        face_results = self.face_system.process_frame(frame, camera_id, store_logs=False)
        
        # Step 2: Detect PPE equipment globally in the frame
        ppe_detections = self.ppe_system.detect_ppe(frame)
        
        # Step 2.5: ✅ NEW - Associate PPE items to specific faces using spatial proximity
        person_ppe_map = self._associate_ppe_to_faces(
            face_results, 
            ppe_detections,
            frame_shape=frame.shape
        )
        
        # Step 3: Check compliance for each recognized person with THEIR specific PPE
        compliance_results = []
        
        for face_result in face_results:
            person_id = face_result.get('person_id')
            
            if person_id and person_id != 'unknown':
                # Get person details from database
                person = self.db.get_person(person_id)
                
                if person:
                    person_role = person.get('role', 'default')
                    
                    # ✅ FIXED: Get person-specific PPE items (not global PPE list)
                    person_ppe = person_ppe_map.get(person_id, [])
                    
                    # Check PPE compliance using only this person's PPE
                    compliance = self.ppe_system.check_person_compliance(
                        person_role, person_ppe  # ← Now using person-specific PPE!
                    )
                    
                    compliance_result = {
                        'person_id': person_id,
                        'person_name': person['name'],
                        'role': person_role,
                        'face_confidence': face_result['confidence'],
                        'face_bbox': face_result['box'],
                        'compliance': compliance,
                        'is_violation': not compliance['is_compliant'],
                        'ppe_count': len(person_ppe)  # ✅ NEW: Track PPE count per person
                    }
                    
                    compliance_results.append(compliance_result)
                    
                    # Log to database if non-compliant
                    if not compliance['is_compliant']:
                        self._log_ppe_violation(
                            person_id=person_id,
                            person_name=person['name'],
                            role=person_role,
                            camera_id=camera_id,
                            compliance=compliance,
                            face_bbox=face_result['box']
                        )
        
        detection_time = (datetime.now() - detection_start).total_seconds() * 1000
        
        return {
            'face_results': face_results,
            'ppe_detections': ppe_detections,
            'compliance_results': compliance_results,
            'processing_time_ms': detection_time,
            'timestamp': datetime.utcnow(),
            'person_ppe_map': person_ppe_map  # ✅ NEW: Include mapping for debugging
        }

    
    def _log_ppe_violation(self, person_id: str, person_name: str, role: str,
                          camera_id: str, compliance: Dict, face_bbox: Tuple):
        """Log PPE violation to database"""
        
        camera = self.db.get_camera(camera_id)
        
        violation_log = {
            'log_type': 'ppe_violation',
            'person_id': person_id,
            'person_name': person_name,
            'role': role,
            'camera_id': camera_id,
            'camera_name': camera['name'] if camera else camera_id,
            'location': camera['location'] if camera else 'Unknown',
            'missing_ppe': compliance['missing_ppe'],
            'required_ppe': compliance['required_ppe'],
            'wearing_ppe': compliance['wearing_ppe'],
            'compliance_percentage': compliance['compliance_percentage'],
            'face_bbox': {
                'x1': face_bbox[0],
                'y1': face_bbox[1],
                'x2': face_bbox[2],
                'y2': face_bbox[3]
            },
            'timestamp': datetime.utcnow(),
            'is_alert': True,
            'alert_sent': False,
            'resolved': False
        }
        
        # Insert into recognition_logs with PPE violation flag
        self.db.recognition_logs.insert_one(violation_log)
    
    def draw_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """
        Draw all detection results on frame
        
        Args:
            frame: Input frame
            results: Processing results
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw PPE detections
        annotated_frame = self.ppe_system.draw_ppe_detections(
            annotated_frame, results['ppe_detections']
        )
        
        # Draw face recognition and compliance status
        for i, compliance_result in enumerate(results['compliance_results']):
            # Draw face bounding box
            x1, y1, x2, y2 = compliance_result['face_bbox']
            
            # Color based on compliance
            if compliance_result['is_violation']:
                face_color = (0, 0, 255)  # Red for non-compliant
            else:
                face_color = (0, 255, 0)  # Green for compliant
            
            # Draw face box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), face_color, 3)
            
            # Draw person name and PPE count
            name_label = f"{compliance_result['person_name']} (PPE: {compliance_result.get('ppe_count', 0)})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(annotated_frame, name_label, (x1, y1 - 10),
                       font, 0.7, face_color, 2)
            
            # Draw compliance status box
            status_x = 10
            status_y = 10 + (i * 160)
            
            annotated_frame = self.ppe_system.draw_compliance_status(
                annotated_frame,
                compliance_result['person_name'],
                compliance_result['role'],
                compliance_result['compliance'],
                (status_x, status_y)
            )
        
        # Draw statistics
        stats_y = annotated_frame.shape[0] - 100
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(annotated_frame, 
                   f"Faces Detected: {len(results['face_results'])}", 
                   (10, stats_y), font, 0.6, (255, 255, 255), 2)
        
        cv2.putText(annotated_frame,
                   f"PPE Items: {len(results['ppe_detections'])}",
                   (10, stats_y + 25), font, 0.6, (255, 255, 255), 2)
        
        violations = sum(1 for r in results['compliance_results'] if r['is_violation'])
        cv2.putText(annotated_frame,
                   f"Violations: {violations}",
                   (10, stats_y + 50), font, 0.6, (0, 0, 255) if violations > 0 else (0, 255, 0), 2)
        
        cv2.putText(annotated_frame,
                   f"Processing: {results['processing_time_ms']:.1f}ms",
                   (10, stats_y + 75), font, 0.6, (255, 255, 255), 2)
        
        return annotated_frame


# ============================================================================
# Example Usage
# ============================================================================
# if __name__ == "__main__":
#     from mongo_db_manager import FaceRecognitionDB
#     from enhanced_face_recognition import EnhancedFaceRecognitionSystem
    
#     # Initialize systems
#     db = FaceRecognitionDB()
#     face_system = EnhancedFaceRecognitionSystem(db, threshold=0.7, use_cuda=True)
    
#     # Initialize PPE detection
#     # Option 1: Use default YOLOv8
#     ppe_system = PPEDetectionSystem(
#         model_path='yolov8n.pt',  # Will auto-download
#         confidence_threshold=0.5,
#         use_cuda=False
#     )
    
#     # Option 2: Use custom trained PPE model
#     # ppe_system = PPEDetectionSystem(
#     #     model_path='path/to/ppe_yolov8.pt',
#     #     confidence_threshold=0.6
#     # )
    
#     # Create integrated system
#     integrated_system = IntegratedFaceAndPPESystem(face_system, ppe_system, db)
    
#     # Add custom PPE requirements (optional)
#     PPEDetectionSystem.add_ppe_requirements('engineer', ['helmet', 'vest'])
#     PPEDetectionSystem.add_ppe_requirements('welder', ['helmet', 'vest', 'gloves', 'mask'])
    
#     print("✅ System ready!")
#     print("\nPPE Requirements by Role:")
#     for role in ['engineer', 'worker', 'welder']:
#         requirements = PPEDetectionSystem.get_role_requirements(role)
#         print(f"  {role}: {', '.join(requirements)}")
    
#     db.close()