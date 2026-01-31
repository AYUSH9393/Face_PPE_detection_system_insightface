"""
Test script for PPE color detection
Run this to verify color detection is working correctly
"""

from mongo_db_manager import FaceRecognitionDB
from ppe_color_detector import PPEColorDetector
import cv2
import numpy as np

def test_color_detector():
    """Test the PPE color detector"""
    
    print("=" * 70)
    print("PPE Color Detection Test")
    print("=" * 70)
    
    # Initialize database and color detector
    print("\n1. Initializing database connection...")
    try:
        db = FaceRecognitionDB()
        print("   [OK] Database connected")
    except Exception as e:
        print(f"   [ERROR] Database connection failed: {e}")
        return
    
    print("\n2. Initializing color detector...")
    try:
        color_detector = PPEColorDetector(db)
        print(f"   [OK] Color detector initialized")
        print(f"   - Color checking enabled: {color_detector.enable_color_checking}")
        print(f"   - Color match threshold: {color_detector.color_match_threshold}%")
    except Exception as e:
        print(f"   [ERROR] Color detector initialization failed: {e}")
        return
    
    # Test 1: Get available colors
    print("\n3. Testing available colors...")
    try:
        colors = color_detector.get_all_available_colors()
        print(f"   [OK] Found {len(colors)} available colors:")
        for color in colors:
            print(f"      - {color['name']} ({color['id']}): {color['display_color']}")
    except Exception as e:
        print(f"   [ERROR] Failed to get colors: {e}")
    
    # Test 2: Get role color requirements
    print("\n4. Testing role color requirements...")
    test_roles = ["worker", "engineer", "supervisor", "visitor"]
    for role in test_roles:
        try:
            requirements = color_detector.get_role_color_requirements(role)
            print(f"   [OK] {role.upper()}:")
            for ppe_item, colors in requirements.items():
                print(f"      - {ppe_item}: {', '.join(colors)}")
        except Exception as e:
            print(f"   [ERROR] Failed for role {role}: {e}")
    
    # Test 3: Create test image and detect color
    print("\n5. Testing color detection on synthetic image...")
    try:
        # Create a yellow test image (simulating a helmet)
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        test_image[:, :] = [0, 255, 255]  # BGR: Yellow
        
        bbox = {"x1": 50, "y1": 50, "x2": 150, "y2": 150}
        detected_color = color_detector.detect_dominant_color(test_image, bbox)
        
        print(f"   [OK] Test image (yellow) detected as: {detected_color}")
        
        # Test validation
        validation = color_detector.validate_ppe_color("worker", "safety_helmet", detected_color)
        print(f"   - Valid for worker helmet: {validation['color_valid']}")
        print(f"   - Allowed colors: {', '.join(validation['allowed_colors'])}")
        
    except Exception as e:
        print(f"   [ERROR] Color detection test failed: {e}")
    
    # Test 4: Test wrong color scenario
    print("\n6. Testing wrong color detection...")
    try:
        # Create a blue test image
        test_image_blue = np.zeros((200, 200, 3), dtype=np.uint8)
        test_image_blue[:, :] = [255, 0, 0]  # BGR: Blue
        
        detected_color = color_detector.detect_dominant_color(test_image_blue, bbox)
        print(f"   [OK] Test image (blue) detected as: {detected_color}")
        
        # Test validation for worker (should be invalid)
        validation = color_detector.validate_ppe_color("worker", "safety_helmet", detected_color)
        print(f"   - Valid for worker helmet: {validation['color_valid']}")
        print(f"   - Detected: {validation['detected_color']}")
        print(f"   - Expected: {', '.join(validation['allowed_colors'])}")
        
    except Exception as e:
        print(f"   [ERROR] Wrong color test failed: {e}")
    
    # Test 5: Configuration reload
    print("\n7. Testing configuration reload...")
    try:
        color_detector.reload_config()
        print("   [OK] Configuration reloaded successfully")
    except Exception as e:
        print(f"   [ERROR] Reload failed: {e}")
    
    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Check API endpoints: GET /api/settings/ppe/colors")
    print("2. Update role requirements: PUT /api/settings/ppe/colors/role/<role>")
    print("3. Enable/disable color checking: PUT /api/settings/ppe/colors")
    print("\nSee PPE_COLOR_DETECTION_GUIDE.md for more information")
    print("=" * 70)

if __name__ == "__main__":
    test_color_detector()
