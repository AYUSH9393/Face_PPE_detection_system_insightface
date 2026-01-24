"""
Video Processor - Process videos for face recognition
"""
import cv2
import numpy as np
from typing import Optional, Callable
from face_recognizer import FaceRecognizer
from utils import draw_face_box, draw_landmarks, create_video_writer
import time


class VideoProcessor:
    """
    Process videos for face detection and recognition
    """
    
    def __init__(self, recognizer: FaceRecognizer, show_landmarks: bool = False):
        """
        Initialize video processor
        
        Args:
            recognizer: FaceRecognizer instance
            show_landmarks: Whether to draw facial landmarks
        """
        self.recognizer = recognizer
        self.show_landmarks = show_landmarks
        
        # Color scheme
        self.colors = {
            'recognized': (0, 255, 0),      # Green for recognized faces
            'unknown': (0, 0, 255),          # Red for unknown faces
            'landmarks': (0, 255, 255)       # Yellow for landmarks
        }
    
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list]:
        """
        Process a single frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (processed_frame, recognition_results)
        """
        # Recognize faces in frame
        results = self.recognizer.recognize_faces_in_image(frame)
        
        # Draw results on frame
        processed_frame = frame.copy()
        
        for result in results:
            # Choose color based on recognition status
            color = self.colors['recognized'] if result['is_recognized'] else self.colors['unknown']
            
            # Draw bounding box and label
            label = result['name']
            confidence = result['similarity']
            
            processed_frame = draw_face_box(
                processed_frame,
                result['bbox'],
                label=label,
                confidence=confidence,
                color=color
            )
            
            # Draw landmarks if enabled
            if self.show_landmarks and result['landmarks'] is not None:
                processed_frame = draw_landmarks(
                    processed_frame,
                    result['landmarks'],
                    color=self.colors['landmarks']
                )
        
        return processed_frame, results
    
    def process_video_file(self, video_path: str, output_path: Optional[str] = None,
                          display: bool = True, max_frames: Optional[int] = None,
                          log_path: Optional[str] = None) -> dict:
        """
        Process a video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save processed video (optional)
            display: Whether to display the video while processing
            max_frames: Maximum number of frames to process (None for all)
            log_path: Path to save detection log file (optional)
            
        Returns:
            Dictionary with processing statistics
        """
        print(f"\nProcessing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS, {total_frames} frames")
        
        # Create video writer if output path specified
        writer = None
        if output_path:
            writer = create_video_writer(output_path, fps, (frame_width, frame_height))
            print(f"Saving output to: {output_path}")
        
        # Create log file if log path specified
        log_file = None
        if log_path:
            import os
            os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else '.', exist_ok=True)
            log_file = open(log_path, 'w', encoding='utf-8')
            log_file.write(f"Face Recognition Log - Video: {video_path}\n")
            log_file.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"="*70 + "\n\n")
            log_file.write(f"{'Frame':<10} {'Timestamp':<12} {'Name':<20} {'Similarity':<12}\n")
            log_file.write(f"{'-'*70}\n")
            print(f"Logging detections to: {log_path}")
        
        # Processing statistics
        stats = {
            'total_frames': 0,
            'faces_detected': 0,
            'faces_recognized': 0,
            'processing_time': 0,
            'fps': 0
        }
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check max frames limit
                if max_frames and frame_count >= max_frames:
                    break
                
                # Process frame
                frame_start = time.time()
                processed_frame, results = self.process_frame(frame)
                frame_time = time.time() - frame_start
                
                # Update statistics
                stats['total_frames'] += 1
                stats['faces_detected'] += len(results)
                stats['faces_recognized'] += sum(1 for r in results if r['is_recognized'])
                
                # Log detections if log file is open
                if log_file and len(results) > 0:
                    # Calculate timestamp in video
                    timestamp_seconds = frame_count / fps if fps > 0 else 0
                    minutes = int(timestamp_seconds // 60)
                    seconds = int(timestamp_seconds % 60)
                    timestamp_str = f"{minutes:02d}:{seconds:02d}"
                    
                    # Write each detected face to log
                    for result in results:
                        log_file.write(f"{frame_count:<10} {timestamp_str:<12} {result['name']:<20} {result['similarity']:<12.3f}\n")
                    log_file.flush()  # Ensure data is written immediately
                
                # Display FPS on frame
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                cv2.putText(processed_frame, f"FPS: {current_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Faces: {len(results)}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Write frame if output specified
                if writer:
                    writer.write(processed_frame)
                
                # Display frame
                if display:
                    cv2.imshow('Face Recognition', processed_frame)
                    
                    # Check for quit key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nProcessing interrupted by user")
                        break
                    elif key == ord('s'):
                        # Save screenshot
                        screenshot_path = f"../output/screenshot_{frame_count}.jpg"
                        cv2.imwrite(screenshot_path, processed_frame)
                        print(f"Screenshot saved: {screenshot_path}")
                
                frame_count += 1
                
                # Print progress
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    print(f"Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)", end='\r')
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            if log_file:
                # Write summary to log file
                log_file.write(f"\n{'-'*70}\n")
                log_file.write(f"\nSummary:\n")
                log_file.write(f"  Total frames processed: {stats.get('total_frames', frame_count)}\n")
                log_file.write(f"  Total faces detected: {stats.get('faces_detected', 0)}\n")
                log_file.write(f"  Total faces recognized: {stats.get('faces_recognized', 0)}\n")
                log_file.close()
                print(f"\n✓ Detection log saved")
            if display:
                cv2.destroyAllWindows()
        
        # Calculate final statistics
        total_time = time.time() - start_time
        stats['processing_time'] = total_time
        stats['fps'] = stats['total_frames'] / total_time if total_time > 0 else 0
        
        # Print summary
        print(f"\n\nProcessing complete!")
        print(f"  Frames processed: {stats['total_frames']}")
        print(f"  Total faces detected: {stats['faces_detected']}")
        print(f"  Total faces recognized: {stats['faces_recognized']}")
        print(f"  Processing time: {stats['processing_time']:.2f} seconds")
        print(f"  Average FPS: {stats['fps']:.2f}")
        
        return stats
    
    def process_webcam(self, camera_id: int = 0, output_path: Optional[str] = None):
        """
        Process webcam feed in real-time
        
        Args:
            camera_id: Camera device ID (default: 0)
            output_path: Path to save recorded video (optional)
        """
        print(f"\nStarting webcam (camera {camera_id})...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Failed to open camera {camera_id}")
        
        # Get camera properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Camera resolution: {frame_width}x{frame_height}")
        
        # Create video writer if output path specified
        writer = None
        if output_path:
            writer = create_video_writer(output_path, 30.0, (frame_width, frame_height))
            print(f"Recording to: {output_path}")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Process frame
                processed_frame, results = self.process_frame(frame)
                
                # Display info
                cv2.putText(processed_frame, f"Faces: {len(results)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(processed_frame, "Press 'q' to quit", (10, frame_height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Write frame if recording
                if writer:
                    writer.write(processed_frame)
                    cv2.circle(processed_frame, (frame_width - 30, 30), 10, (0, 0, 255), -1)
                
                # Display frame
                cv2.imshow('Webcam Face Recognition', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_path = f"../output/webcam_screenshot_{frame_count}.jpg"
                    cv2.imwrite(screenshot_path, processed_frame)
                    print(f"Screenshot saved: {screenshot_path}")
                
                frame_count += 1
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        print(f"\nWebcam session ended. Processed {frame_count} frames")


if __name__ == "__main__":
    import sys
    
    # Initialize recognizer
    database_path = "../database/faces.pkl"
    
    try:
        recognizer = FaceRecognizer(database_path, similarity_threshold=0.4)
        processor = VideoProcessor(recognizer, show_landmarks=True)
        
        if len(sys.argv) > 1:
            # Process video file
            video_path = sys.argv[1]
            output_path = sys.argv[2] if len(sys.argv) > 2 else "../output/output_video.mp4"
            processor.process_video_file(video_path, output_path, display=True)
        else:
            # Process webcam
            processor.process_webcam(camera_id=0, output_path="../output/webcam_recording.mp4")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
