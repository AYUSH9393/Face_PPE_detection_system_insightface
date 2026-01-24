"""
Improved Face Recognition + PPE Detection with All Fixes
- RTSP stream optimization (suppress HEVC warnings)
- Spatial PPE verification
- Unknown person handling as visitors
- Fixed compliance logic
"""

import cv2
import sys
import os
from datetime import datetime
import numpy as np
import threading
import time


# Suppress OpenCV HEVC warnings for RTSP
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|analyzeduration;1000000|probesize;1000000'
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
cv2.setLogLevel(0)  # Suppress all OpenCV warnings

from mongo_db_manager import FaceRecognitionDB
from enhanced_face_recognition import EnhancedFaceRecognitionSystem
from fixed_ppe_detection_system import EnhancedPPEDetectionSystem, ImprovedIntegratedSystem


class RTSPFrameGrabber:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
            'rtsp_transport;tcp|'
            'fflags;nobuffer|'
            'flags;low_delay|'
            'probesize;32|'
            'analyzeduration;0'
        )

        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.running = True
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


def configure_rtsp_capture(cap, target_width=1280, target_height=720):
    """
    Configure RTSP camera for optimal performance
    Reduces resolution and buffers to minimize lag
    """
    if cap.isOpened():
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        
        # Reduce FPS for RTSP streams
        cap.set(cv2.CAP_PROP_FPS, 15)
        
        # Minimize buffer to reduce lag (IMPORTANT for RTSP)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Set timeout
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        
        # Get actual values
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"   ✅ Camera configured: {actual_width}x{actual_height} @ {actual_fps}fps")
        return True
    return False


def display_frame_optimized(frame, window_name, max_height=720):
    """
    Display frame with automatic resizing to fit screen
    Prevents window from being too large
    """
    height, width = frame.shape[:2]
    
    if height > max_height:
        scale = max_height / height
        new_width = int(width * scale)
        new_height = max_height
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        resized = frame
    
    cv2.imshow(window_name, resized)


def main():
    print("="*70)
    print("Improved Face Recognition + PPE Detection System")
    print("With Fixes: RTSP optimization, Spatial verification, Unknown handling")
    print("="*70)
    
    # Initialize database
    print("\n🔌 Connecting to database...")
    try:
        db = FaceRecognitionDB(
            connection_string='mongodb://localhost:27017/',
            database_name='face_recognition'
        )
        print("✅ Connected to database")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # Initialize face recognition system
    print("\n🤖 Initializing face recognition system...")
    try:
        face_system = EnhancedFaceRecognitionSystem(
            db=db,
            threshold=0.7,
            use_cuda=True
        )
        print(f"✅ Face system initialized with {len(face_system.embeddings_cache)} persons")
    except Exception as e:
        print(f"❌ Failed to initialize face system: {e}")
        db.close()
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # Initialize enhanced PPE detection system
    print("\n🦺 Initializing enhanced PPE detection system...")
    try:
        ppe_system = EnhancedPPEDetectionSystem(
            model_path='models/best.pt',
            confidence_threshold=0.7,
            use_cuda=True
        )
        print("✅ Enhanced PPE detection system initialized")
    except Exception as e:
        print(f"❌ Failed to initialize PPE system: {e}")
        db.close()
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # Create improved integrated system
    print("\n🔗 Creating improved integrated system...")
    integrated_system = ImprovedIntegratedSystem(face_system, ppe_system, db)
    
    # Configure PPE requirements
    print("\n⚙️ Configuring PPE requirements...")
    
    ppe_requirements = {
        'engineer': ['helmet'],
        'worker': ['helmet', 'vest', 'gloves'],
        'supervisor': ['helmet', 'vest'],
        'contractor': ['helmet', 'vest', 'gloves', 'boots'],
        'electrician': ['helmet', 'vest', 'gloves', 'goggles'],
        'welder': ['helmet', 'vest', 'gloves', 'goggles', 'mask'],
        'visitor': ['helmet'],  # For unknown persons
        'manager': ['helmet'],
    }
    
    for role, requirements in ppe_requirements.items():
        EnhancedPPEDetectionSystem.add_ppe_requirements(role, requirements)
    
    print("✅ PPE requirements configured")
    print("\n📋 PPE Requirements by Role:")
    for role, requirements in ppe_requirements.items():
        print(f"   • {role.capitalize()}: {', '.join(requirements)}")
    
    # Get camera configuration
    print("\n📹 Loading camera configuration...")
    camera_id = "CAM_002"
    
    camera = db.get_camera(camera_id)
    if not camera:
        print(f"⚠️ Camera {camera_id} not found")
        print("Using default webcam (index 0)")
        video_source = 0
        camera_name = "Default Webcam"
        is_rtsp = False
    else:
        camera_name = camera['name']
        if camera.get('stream_index') is not None:
            video_source = camera['stream_index']
            is_rtsp = False
        else:
            video_source = camera.get('rtsp_url', 0)
            is_rtsp = True
        print(f"✅ Camera: {camera_name}")
        print(f"   Type: {'RTSP' if is_rtsp else 'USB'}")
    
    # Open video capture with RTSP optimization
    print(f"\n🎥 Opening camera...")
    
    if is_rtsp:
        print("   Configuring RTSP stream (this may take a moment)...")
        # Use TCP transport for RTSP (more reliable)
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
    
    cap = cv2.VideoCapture(video_source)
    grabber = RTSPFrameGrabber(video_source).start()
    
    
    if not cap.isOpened():
        print(f"❌ Failed to open camera")
        print("\nTroubleshooting:")
        print("  For RTSP: Check camera IP, credentials, and network")
        print("  For USB: Try different index (0, 1, 2)")
        db.close()
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    print("✅ Camera opened successfully")
    
    # Configure camera settings
    if is_rtsp:
        print("   Optimizing RTSP settings...")
        configure_rtsp_capture(cap, target_width=1280, target_height=720)
    else:
        # For USB cameras, set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print(f"   USB camera resolution: 1280x720")
    
    # Display instructions
    print("\n" + "="*70)
    print("🎬 Starting Improved PPE Detection System")
    print("="*70)
    print("\n📝 New Features:")
    print("   ✅ RTSP stream optimization (no more HEVC warnings)")
    print("   ✅ Spatial PPE verification (helmet must be on head)")
    print("   ✅ Unknown persons treated as visitors")
    print("   ✅ Fixed compliance logic (unknown ≠ 100% compliant)")
    print("\n🎮 Controls:")
    print("   • Press 'q' to quit")
    print("   • Press 's' to save screenshot")
    print("   • Press 'i' to show system info")
    print("\n" + "="*70 + "\n")
    
    input("Press Enter to start...")
    
    # Processing variables
    frame_count = 0
    skip_frames = 1  # Process every 2nd frame
    
    # Statistics
    total_faces = 0
    total_violations = 0
    total_unknown = 0
    
    # FPS calculation
    fps_start = datetime.now()
    fps_count = 0
    current_fps = 0
    
    # Create window
    window_name = f"PPE Detection - {camera_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    try:
        while True:
            frame = grabber.read()
            if frame is None:
                print("⚠️ Failed to read frame")
                if is_rtsp:
                    print("   Attempting to reconnect to RTSP stream...")
                    cap.release()
                    #cap = cv2.VideoCapture(video_source)
                    grabber = RTSPFrameGrabber(video_source).start()
                    time.sleep(1.5)
                    if not cap.isOpened():
                        print("   ❌ Reconnection failed")
                        break
                    configure_rtsp_capture(cap)
                    continue
                else:
                    break
            
            frame_count += 1
            fps_count += 1
            
            # Calculate FPS
            elapsed = (datetime.now() - fps_start).total_seconds()
            if elapsed > 1.0:
                current_fps = fps_count / elapsed
                fps_start = datetime.now()
                fps_count = 0
            
            # Process frame
            if frame_count % skip_frames == 0:
                # Process with improved integrated system
                results = integrated_system.process_frame_with_ppe(frame, camera_id)
                
                # Update statistics
                total_faces += len(results['face_results'])
                
                for compliance_result in results['compliance_results']:
                    if compliance_result['is_violation']:
                        total_violations += 1
                        
                        person_name = compliance_result['person_name']
                        role = compliance_result['role']
                        missing = ', '.join(compliance_result['compliance']['missing_ppe'])
                        
                        if compliance_result['is_unknown']:
                            total_unknown += 1
                            print(f"⚠️ UNKNOWN VISITOR - Missing PPE: {missing}")
                        else:
                            print(f"⚠️ VIOLATION: {person_name} ({role.upper()}) - Missing: {missing}")
                
                # Draw results
                annotated_frame = integrated_system.draw_results(frame, results)
                
                # Add FPS overlay
                cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", 
                           (annotated_frame.shape[1] - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                annotated_frame = frame
            
            # Display frame (with automatic resizing)
            display_frame_optimized(annotated_frame, window_name, max_height=1080)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n🛑 Stopping system...")
                break
            
            elif key == ord('s'):
                filename = f"ppe_screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"📸 Screenshot saved: {filename}")
            
            elif key == ord('i'):
                print("\n" + "="*50)
                print("System Information:")
                print(f"  Frame count: {frame_count}")
                print(f"  Current FPS: {current_fps:.1f}")
                print(f"  Total faces detected: {total_faces}")
                print(f"  Total violations: {total_violations}")
                print(f"  Unknown persons: {total_unknown}")
                print("="*50 + "\n")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        grabber.stop()


        
        # Show final statistics
        print("\n" + "="*70)
        print("📊 Session Statistics")
        print("="*70)
        print(f"   Total frames processed: {frame_count}")
        print(f"   Total faces detected: {total_faces}")
        print(f"   Total violations: {total_violations}")
        print(f"   Unknown persons detected: {total_unknown}")
        
        # Get today's violations from database
        violations_today = db.recognition_logs.count_documents({
            'log_type': 'ppe_violation',
            'timestamp': {'$gte': datetime.utcnow().replace(hour=0, minute=0, second=0)}
        })
        
        unknown_violations = db.recognition_logs.count_documents({
            'log_type': 'ppe_violation',
            'is_unknown_person': True,
            'timestamp': {'$gte': datetime.utcnow().replace(hour=0, minute=0, second=0)}
        })
        
        print(f"\n   Database - Today's violations: {violations_today}")
        print(f"   Database - Unknown violations: {unknown_violations}")
        print("="*70)
        
        db.close()
        print("\n✅ System stopped successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input("\nPress Enter to exit...")