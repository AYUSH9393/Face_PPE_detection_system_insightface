"""
Camera Handler Module
Manages camera feed and frame capture
"""

import cv2
import time


class CameraHandler:
    def __init__(self, camera_index=0, width=1280, height=720):
        """
        Initialize camera handler
        
        Args:
            camera_index: Camera device index (0 for default camera)
            width: Frame width
            height: Frame height
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.is_opened = False
        
    def open(self):
        """Open the camera"""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise Exception(f"Could not open camera {self.camera_index}")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Set FPS
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_opened = True
        print(f"Camera {self.camera_index} opened successfully")
        
    def read_frame(self):
        """
        Read a frame from the camera
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.is_opened:
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame
    
    def release(self):
        """Release the camera"""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            print("Camera released")
    
    def get_fps(self):
        """Get current FPS"""
        if self.cap is not None:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 0
    
    def get_resolution(self):
        """Get current resolution"""
        if self.cap is not None:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        return 0, 0
    
    def __enter__(self):
        """Context manager entry"""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()


class FrameProcessor:
    def __init__(self):
        """Initialize frame processor"""
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
    def calculate_fps(self):
        """Calculate current FPS"""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time > 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        
        return self.fps
    
    def draw_info(self, frame, text_lines, color=(0, 255, 0)):
        """
        Draw information text on frame
        
        Args:
            frame: Input frame
            text_lines: List of text lines to draw
            color: Text color (BGR)
        """
        y_offset = 30
        for line in text_lines:
            cv2.putText(frame, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30
    
    def draw_rectangle(self, frame, x, y, w, h, color=(0, 255, 0), thickness=2):
        """Draw rectangle on frame"""
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    def draw_text_with_background(self, frame, text, position, 
                                  font_scale=0.7, thickness=2,
                                  text_color=(255, 255, 255), 
                                  bg_color=(0, 128, 0)):
        """
        Draw text with background rectangle
        
        Args:
            frame: Input frame
            text: Text to draw
            position: (x, y) position
            font_scale: Font scale
            thickness: Text thickness
            text_color: Text color (BGR)
            bg_color: Background color (BGR)
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        x, y = position
        
        # Draw background rectangle
        cv2.rectangle(frame, 
                     (x, y - text_height - 10), 
                     (x + text_width + 10, y + baseline),
                     bg_color, -1)
        
        # Draw text
        cv2.putText(frame, text, (x + 5, y - 5), 
                   font, font_scale, text_color, thickness)
    
    def resize_frame(self, frame, width=None, height=None):
        """
        Resize frame while maintaining aspect ratio
        
        Args:
            frame: Input frame
            width: Target width (optional)
            height: Target height (optional)
            
        Returns:
            Resized frame
        """
        if width is None and height is None:
            return frame
        
        h, w = frame.shape[:2]
        
        if width is not None:
            ratio = width / w
            new_width = width
            new_height = int(h * ratio)
        else:
            ratio = height / h
            new_height = height
            new_width = int(w * ratio)
        
        resized = cv2.resize(frame, (new_width, new_height), 
                           interpolation=cv2.INTER_AREA)
        return resized
