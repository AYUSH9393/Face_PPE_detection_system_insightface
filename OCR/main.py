"""
License Plate Recognition System - Main Application
Real-time license plate detection and OCR from camera feed
"""

import cv2
import time
import argparse
import os
import json
import threading
from datetime import datetime
from collections import deque
from plate_detector import PlateDetector
from ocr_engine import OCREngine
from camera_handler import CameraHandler, FrameProcessor


class LicensePlateRecognitionSystem:
    def __init__(self, camera_index=0, save_detections=True, output_dir="detections"):
        """
        Initialize the License Plate Recognition System
        
        Args:
            camera_index: Camera device index
            save_detections: Whether to save detected plates
            output_dir: Directory to save detections
        """
        print("=" * 60)
        print("License Plate Recognition System - Optimized")
        print("=" * 60)
        
        # Initialize components
        self.camera = CameraHandler(camera_index=camera_index, width=640, height=480)  # Lower resolution for speed
        self.detector = PlateDetector()
        self.ocr = OCREngine(languages=['en'], gpu=False)
        self.processor = FrameProcessor()
        
        self.save_detections = save_detections
        self.output_dir = output_dir
        
        # Create output directory
        if self.save_detections:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Detection settings
        self.min_confidence = 0.5
        self.detection_cooldown = 2.0  # Seconds between detections
        self.last_detection_time = 0
        self.last_plate_number = ""
        
        # Statistics
        self.total_detections = 0
        self.successful_reads = 0
        
        # Best detections tracking (plate_number: {confidence, timestamp, image_path})
        self.best_detections = {}
        self.log_file = "log/detected_plates.txt"
        self.load_previous_detections()
        
        # Performance optimization settings
        self.frame_skip = 2  # Process every Nth frame (1 = all frames, 2 = every other frame)
        self.frame_count = 0
        self.process_resolution = (320, 240)  # Lower resolution for detection
        
        # Threading for OCR
        self.ocr_queue = deque(maxlen=5)  # Queue for OCR processing
        self.ocr_lock = threading.Lock()
        self.ocr_thread = None
        self.ocr_running = False
        
        # Performance monitoring
        self.processing_times = deque(maxlen=30)
        self.avg_processing_time = 0
        
    def process_frame(self, frame):
        """
        Process a single frame for plate detection and recognition (OPTIMIZED)
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Processed frame with annotations
        """
        start_time = time.time()
        display_frame = frame.copy()
        current_time = time.time()
        
        # Frame skipping for performance
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return display_frame, []
        
        # Resize frame for faster detection
        small_frame = cv2.resize(frame, self.process_resolution)
        scale_x = frame.shape[1] / self.process_resolution[0]
        scale_y = frame.shape[0] / self.process_resolution[1]
        
        # Detect plates on smaller frame
        plates = self.detector.detect(small_frame)
        
        detected_plates = []
        
        for (x, y, w, h) in plates:
            # Scale coordinates back to original frame
            x_orig = int(x * scale_x)
            y_orig = int(y * scale_y)
            w_orig = int(w * scale_x)
            h_orig = int(h * scale_y)
            
            # Draw detection rectangle
            self.processor.draw_rectangle(display_frame, x_orig, y_orig, w_orig, h_orig, 
                                        color=(0, 255, 255), thickness=2)
            
            # Extract and preprocess plate from original frame
            plate_img = self.detector.extract_plate(frame, x_orig, y_orig, w_orig, h_orig)
            processed_plate = self.detector.preprocess_plate(plate_img)
            
            # Perform OCR with confidence
            plate_number, confidence, raw_results = self.ocr.recognize_with_confidence(
                processed_plate
            )
            
            # Check if detection is valid
            if (confidence >= self.min_confidence and 
                plate_number and 
                self.ocr.validate_plate_format(plate_number)):
                
                # Check cooldown to avoid duplicate detections
                if (current_time - self.last_detection_time > self.detection_cooldown or
                    plate_number != self.last_plate_number):
                    
                    detected_plates.append({
                        'number': plate_number,
                        'confidence': confidence,
                        'bbox': (x_orig, y_orig, w_orig, h_orig),
                        'image': plate_img
                    })
                    
                    # Update tracking
                    self.last_detection_time = current_time
                    self.last_plate_number = plate_number
                    self.total_detections += 1
                    self.successful_reads += 1
                    
                    # Save detection (in background thread to avoid blocking)
                    if self.save_detections:
                        threading.Thread(
                            target=self.save_detection,
                            args=(plate_img, plate_number, confidence),
                            daemon=True
                        ).start()
                    
                    # Draw success rectangle (green)
                    self.processor.draw_rectangle(display_frame, x_orig, y_orig, w_orig, h_orig, 
                                                color=(0, 255, 0), thickness=3)
                    
                    # Display plate number
                    text = f"{plate_number} ({confidence:.2%})"
                    self.processor.draw_text_with_background(
                        display_frame, text, (x_orig, y_orig - 10),
                        text_color=(255, 255, 255), bg_color=(0, 128, 0)
                    )
                    
                    print(f"[DETECTED] Plate: {plate_number} | Confidence: {confidence:.2%}")
                
                else:
                    # Draw yellow rectangle for duplicate detection
                    self.processor.draw_rectangle(display_frame, x_orig, y_orig, w_orig, h_orig, 
                                                color=(0, 255, 255), thickness=2)
            
            else:
                # Low confidence or invalid format
                if plate_number:
                    text = f"{plate_number} ({confidence:.2%})"
                    self.processor.draw_text_with_background(
                        display_frame, text, (x_orig, y_orig - 10),
                        text_color=(255, 255, 255), bg_color=(0, 0, 255)
                    )
        
        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        
        return display_frame, detected_plates
    
    def load_previous_detections(self):
        """Load previously detected plates from log file"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    for line in f:
                        if line.strip() and not line.startswith('#'):
                            parts = line.strip().split('|')
                            if len(parts) >= 3:
                                plate_num = parts[0].strip()
                                conf = float(parts[1].strip().replace('%', '')) / 100
                                timestamp = parts[2].strip()
                                self.best_detections[plate_num] = {
                                    'confidence': conf,
                                    'timestamp': timestamp,
                                    'image_path': ''
                                }
                print(f"[LOADED] {len(self.best_detections)} previous detections from {self.log_file}")
            except Exception as e:
                print(f"[WARNING] Could not load previous detections: {e}")
    
    def update_best_detection(self, plate_number, confidence, image_path):
        """Update best detection if current confidence is higher"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Check if this is a new plate or better detection
        if plate_number not in self.best_detections or confidence > self.best_detections[plate_number]['confidence']:
            self.best_detections[plate_number] = {
                'confidence': confidence,
                'timestamp': timestamp,
                'image_path': image_path
            }
            self.save_detections_to_file()
            
            if plate_number in self.best_detections and confidence > self.best_detections[plate_number]['confidence']:
                print(f"[UPDATED] Better detection for {plate_number}: {confidence:.2%} (previous: {self.best_detections[plate_number]['confidence']:.2%})")
            else:
                print(f"[NEW] First detection of {plate_number}: {confidence:.2%}")
    
    def save_detections_to_file(self):
        """Save all best detections to text file"""
        try:
            with open(self.log_file, 'w') as f:
                # Write header
                f.write("# License Plate Detection Log\n")
                f.write("# Format: Plate Number | Confidence | Timestamp | Image Path\n")
                f.write("# " + "="*70 + "\n\n")
                
                # Sort by confidence (highest first)
                sorted_plates = sorted(
                    self.best_detections.items(),
                    key=lambda x: x[1]['confidence'],
                    reverse=True
                )
                
                # Write detections
                for plate_num, data in sorted_plates:
                    f.write(f"{plate_num} | {data['confidence']*100:.2f}% | {data['timestamp']} | {data['image_path']}\n")
                
                # Write summary
                f.write("\n" + "="*70 + "\n")
                f.write(f"Total Unique Plates Detected: {len(self.best_detections)}\n")
                
                if self.best_detections:
                    avg_conf = sum(d['confidence'] for d in self.best_detections.values()) / len(self.best_detections)
                    f.write(f"Average Confidence: {avg_conf*100:.2f}%\n")
                    
                    best_plate = max(self.best_detections.items(), key=lambda x: x[1]['confidence'])
                    f.write(f"Highest Confidence: {best_plate[0]} ({best_plate[1]['confidence']*100:.2f}%)\n")
            
            print(f"[LOG] Updated {self.log_file} with {len(self.best_detections)} unique plates")
        except Exception as e:
            print(f"[ERROR] Could not save to log file: {e}")
    
    def save_detection(self, plate_img, plate_number, confidence):
        """Save detected plate image"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/{plate_number}_{timestamp}_{confidence:.2f}.jpg"
        cv2.imwrite(filename, plate_img)
        print(f"[SAVED] {filename}")
        
        # Update best detection tracking
        self.update_best_detection(plate_number, confidence, filename)
    
    def draw_ui(self, frame):
        """Draw user interface elements on frame (OPTIMIZED)"""
        fps = self.processor.calculate_fps()
        
        # Info panel
        info_lines = [
            f"FPS: {fps:.1f} | Proc: {self.avg_processing_time*1000:.0f}ms",
            f"Detections: {self.total_detections} | Unique: {len(self.best_detections)}",
            f"Success Rate: {self.successful_reads}/{self.total_detections}" if self.total_detections > 0 else "Success Rate: 0/0",
            f"Last Plate: {self.last_plate_number}" if self.last_plate_number else "Last Plate: None",
            f"Mode: Optimized (Skip: {self.frame_skip}, Res: {self.process_resolution[0]}x{self.process_resolution[1]})"
        ]
        
        self.processor.draw_info(frame, info_lines, color=(0, 255, 0))
        
        # Instructions
        instructions = [
            "Press 'q' to quit",
            "Press 's' to save frame",
            "Press 'r' to reset stats",
            "Press '+' to increase quality (slower)",
            "Press '-' to decrease quality (faster)"
        ]
        
        y_offset = frame.shape[0] - 100
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        """Run the main application loop"""
        try:
            # Open camera
            self.camera.open()
            
            print("\n" + "=" * 60)
            print("System Running - Press 'q' to quit")
            print("=" * 60 + "\n")
            
            while True:
                # Read frame
                ret, frame = self.camera.read_frame()
                
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Process frame
                processed_frame, detections = self.process_frame(frame)
                
                # Draw UI
                self.draw_ui(processed_frame)
                
                # Display
                cv2.imshow('License Plate Recognition System', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{self.output_dir}/frame_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"[SAVED] Frame: {filename}")
                elif key == ord('r'):
                    # Reset statistics
                    self.total_detections = 0
                    self.successful_reads = 0
                    self.last_plate_number = ""
                    print("[RESET] Statistics reset")
                elif key == ord('+') or key == ord('='):
                    # Increase quality (slower)
                    if self.frame_skip > 1:
                        self.frame_skip -= 1
                        print(f"[QUALITY] Increased quality - Frame skip: {self.frame_skip}")
                    elif self.process_resolution[0] < 640:
                        self.process_resolution = (
                            min(640, self.process_resolution[0] + 160),
                            min(480, self.process_resolution[1] + 120)
                        )
                        print(f"[QUALITY] Increased resolution - {self.process_resolution[0]}x{self.process_resolution[1]}")
                    else:
                        print("[QUALITY] Already at maximum quality")
                elif key == ord('-') or key == ord('_'):
                    # Decrease quality (faster)
                    if self.process_resolution[0] > 160:
                        self.process_resolution = (
                            max(160, self.process_resolution[0] - 160),
                            max(120, self.process_resolution[1] - 120)
                        )
                        print(f"[SPEED] Decreased resolution - {self.process_resolution[0]}x{self.process_resolution[1]}")
                    elif self.frame_skip < 5:
                        self.frame_skip += 1
                        print(f"[SPEED] Increased frame skip - Frame skip: {self.frame_skip}")
                    else:
                        print("[SPEED] Already at maximum speed")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            self.camera.release()
            cv2.destroyAllWindows()
            
            print("\n" + "=" * 60)
            print("Session Summary")
            print("=" * 60)
            print(f"Total Detections: {self.total_detections}")
            print(f"Successful Reads: {self.successful_reads}")
            if self.total_detections > 0:
                success_rate = (self.successful_reads / self.total_detections) * 100
                print(f"Success Rate: {success_rate:.1f}%")
            print(f"Unique Plates: {len(self.best_detections)}")
            print(f"Log File: {self.log_file}")
            print("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='License Plate Recognition System'
    )
    parser.add_argument(
        '--camera', type=int, default=0,
        help='Camera device index (default: 0)'
    )
    parser.add_argument(
        '--no-save', action='store_true',
        help='Disable saving detected plates'
    )
    parser.add_argument(
        '--output-dir', type=str, default='detections',
        help='Output directory for saved detections (default: detections)'
    )
    
    args = parser.parse_args()
    
    # Create and run system
    system = LicensePlateRecognitionSystem(
        camera_index=args.camera,
        save_detections=not args.no_save,
        output_dir=args.output_dir
    )
    
    system.run()


if __name__ == "__main__":
    main()
