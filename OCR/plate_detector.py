"""
License Plate Detection Module
Handles detection of license plates in images using Haar Cascade and contour detection
"""

import cv2
import numpy as np
import imutils


class PlateDetector:
    def __init__(self):
        """Initialize the plate detector with Haar Cascade classifier"""
        # Try to load Haar Cascade for Russian license plates (works well for most plates)
        self.plate_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
        )
        
    def detect_with_cascade(self, image):
        """
        Detect license plates using Haar Cascade
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detected plate regions (x, y, w, h)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        plates = self.plate_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(25, 25)
        )
        return plates
    
    def detect_with_contours(self, image):
        """
        Detect license plates using edge detection and contour analysis
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of potential plate regions
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Edge detection
        edged = cv2.Canny(gray, 30, 200)
        
        # Find contours
        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        plate_regions = []
        
        for contour in contours:
            # Approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
            
            # Look for rectangular contours (4 corners)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                
                # License plates typically have aspect ratio between 2:1 and 5:1
                if 2.0 <= aspect_ratio <= 5.0 and w > 50 and h > 15:
                    plate_regions.append((x, y, w, h))
        
        return plate_regions
    
    def detect(self, image):
        """
        Detect license plates using both methods and combine results
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detected plate regions
        """
        # Try Haar Cascade first
        cascade_plates = self.detect_with_cascade(image)
        
        # If no plates found, try contour method
        if len(cascade_plates) == 0:
            contour_plates = self.detect_with_contours(image)
            return contour_plates
        
        return cascade_plates
    
    def extract_plate(self, image, x, y, w, h, padding=5):
        """
        Extract the plate region from image with optional padding
        
        Args:
            image: Input image
            x, y, w, h: Bounding box coordinates
            padding: Extra pixels to add around the plate
            
        Returns:
            Cropped plate image
        """
        h_img, w_img = image.shape[:2]
        
        # Add padding while staying within image bounds
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_img, x + w + padding)
        y2 = min(h_img, y + h + padding)
        
        plate_img = image[y1:y2, x1:x2]
        return plate_img
    
    def preprocess_plate(self, plate_img):
        """
        Preprocess the plate image for better OCR results
        Focus on detecting BLACK characters only
        
        Args:
            plate_img: Cropped plate image
            
        Returns:
            Preprocessed image with black characters isolated
        """
        # Resize first for consistent processing
        height = 100
        aspect_ratio = plate_img.shape[1] / plate_img.shape[0]
        width = int(height * aspect_ratio)
        resized_color = cv2.resize(plate_img, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        gray = cv2.cvtColor(resized_color, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Isolate dark/black characters using Otsu's thresholding
        # This works well for black text on light backgrounds
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Method 2: Additional filtering for very dark pixels (black characters)
        # Only keep pixels darker than a threshold (black text)
        dark_mask = cv2.inRange(gray, 0, 100)  # Pixels with intensity 0-100 (dark/black)
        
        # Combine both methods
        combined = cv2.bitwise_and(binary, dark_mask)
        
        # Remove noise using morphological operations
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small)
        
        # Connect character components
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
        connected = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_connect)
        
        # Filter out components that are too small or too large to be characters
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(connected, connectivity=8)
        
        # Create output image
        filtered = np.zeros_like(connected)
        
        # Expected character dimensions (approximate)
        min_char_height = height * 0.3  # At least 30% of plate height
        max_char_height = height * 0.9  # At most 90% of plate height
        min_char_width = width * 0.02   # At least 2% of plate width
        max_char_width = width * 0.15   # At most 15% of plate width
        min_char_area = min_char_height * min_char_width * 0.3
        
        # Keep only components that match character size
        for i in range(1, num_labels):  # Skip background (label 0)
            x, y, w, h, area = stats[i]
            
            # Check if component size matches expected character dimensions
            if (min_char_height <= h <= max_char_height and 
                min_char_width <= w <= max_char_width and
                area >= min_char_area):
                filtered[labels == i] = 255
        
        # If filtering removed too much, fall back to connected image
        if cv2.countNonZero(filtered) < cv2.countNonZero(connected) * 0.1:
            filtered = connected
        
        # Final denoising
        denoised = cv2.fastNlMeansDenoising(filtered, None, 10, 7, 21)
        
        # Invert back to black text on white background for OCR
        final = cv2.bitwise_not(denoised)
        
        return final
