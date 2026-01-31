"""
OCR Engine Module
Handles text recognition from license plate images using EasyOCR
"""

import easyocr
import re
import cv2
import numpy as np


class OCREngine:
    def __init__(self, languages=['en'], gpu=False):
        """
        Initialize the OCR engine
        
        Args:
            languages: List of languages to recognize (default: English)
            gpu: Whether to use GPU acceleration
        """
        print("Initializing OCR Engine...")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        print("OCR Engine ready!")
        
    def recognize(self, image):
        """
        Recognize text from the plate image
        
        Args:
            image: Preprocessed plate image
            
        Returns:
            Recognized text string
        """
        # Perform OCR
        results = self.reader.readtext(image)
        
        if not results:
            return ""
        
        # Extract text from results
        texts = [result[1] for result in results]
        full_text = " ".join(texts)
        
        return full_text
    
    def clean_plate_text(self, text):
        """
        Clean and format the recognized text
        
        Args:
            text: Raw OCR output
            
        Returns:
            Cleaned plate number
        """
        # Remove special characters and extra spaces
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Common OCR mistakes correction
        replacements = {
            'O': '0',  # Letter O to number 0
            'I': '1',  # Letter I to number 1
            'S': '5',  # Letter S to number 5 (in number context)
            'Z': '2',  # Letter Z to number 2 (in number context)
            'B': '8',  # Letter B to number 8 (in number context)
        }
        
        # Apply corrections intelligently
        # Typically, license plates have letters at start and numbers at end
        result = ""
        for i, char in enumerate(cleaned):
            # If it's in the latter half and looks like a number, apply corrections
            if i > len(cleaned) // 2 and char in replacements:
                result += replacements[char]
            else:
                result += char
        
        return result
    
    def recognize_and_clean(self, image):
        """
        Perform OCR and return cleaned result
        
        Args:
            image: Plate image
            
        Returns:
            Cleaned plate number
        """
        raw_text = self.recognize(image)
        cleaned_text = self.clean_plate_text(raw_text)
        return cleaned_text, raw_text
    
    def recognize_with_confidence(self, image):
        """
        Recognize text with confidence scores
        
        Args:
            image: Plate image
            
        Returns:
            Tuple of (text, confidence, raw_results)
        """
        results = self.reader.readtext(image)
        
        if not results:
            return "", 0.0, []
        
        # Calculate average confidence
        confidences = [result[2] for result in results]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Extract text
        texts = [result[1] for result in results]
        full_text = " ".join(texts)
        cleaned_text = self.clean_plate_text(full_text)
        
        return cleaned_text, avg_confidence, results
    
    def validate_plate_format(self, text):
        """
        Validate if the recognized text matches the required 10-character format
        
        Args:
            text: Recognized plate text
            
        Returns:
            Boolean indicating if format is valid (exactly 10 characters)
        """
        # Must be exactly 10 characters
        if len(text) != 10:
            return False
        
        # Indian license plate format: 2 letters + 2 digits + 2 letters + 4 digits
        # Example: UP16CV0939
        pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$'
        
        if re.match(pattern, text):
            return True
        
        # Alternative: Just check if it has mix of letters and numbers (10 chars)
        has_letter = any(c.isalpha() for c in text)
        has_digit = any(c.isdigit() for c in text)
        
        if has_letter and has_digit:
            return True
        
        return False
