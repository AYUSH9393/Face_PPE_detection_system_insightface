"""
Test face recognition with a sample image
"""
import cv2
import numpy as np
from mongo_db_manager import FaceRecognitionDB
from insightface_recognition_system import InsightFaceRecognitionSystem

print("="*70)
print("Face Recognition Test")
print("="*70)

# Initialize database
print("\n1. Connecting to database...")
db = FaceRecognitionDB('mongodb://localhost:27017/', 'face_recognition')
print("   ✅ Connected")

# Initialize InsightFace system
print("\n2. Initializing InsightFace system...")
face_system = InsightFaceRecognitionSystem(
    db=db,
    similarity_threshold=0.4,  # Default threshold
    use_cuda=True,
    model_name='buffalo_l'
)
print(f"   ✅ Initialized with {len(face_system.embeddings_cache)} persons")
print(f"   Similarity threshold: {face_system.similarity_threshold}")

# Get system stats
stats = face_system.get_stats()
print(f"\n3. System Stats:")
print(f"   Total persons: {stats['total_persons']}")
print(f"   Total embeddings: {stats['total_embeddings']}")
print(f"   GPU enabled: {stats['gpu_enabled']}")

# List all persons in cache
print(f"\n4. Persons in cache:")
for person_id, data in face_system.embeddings_cache.items():
    print(f"   - {person_id}: {data['name']} ({data['role']}) - {len(data['embeddings'])} embeddings")

# Test with a sample image if available
print(f"\n5. Testing face detection...")
print("   To test recognition, provide an image path:")
print("   Example: python test_face_recognition.py path/to/image.jpg")

import sys
if len(sys.argv) > 1:
    image_path = sys.argv[1]
    print(f"\n   Loading image: {image_path}")
    
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"   ❌ Failed to load image")
        else:
            print(f"   ✅ Image loaded: {frame.shape}")
            
            # Process frame
            print(f"\n   Processing frame...")
            results = face_system.process_frame(frame, "TEST_CAM", store_logs=False)
            
            print(f"\n   Results:")
            print(f"   Faces detected: {len(results)}")
            
            for i, result in enumerate(results):
                print(f"\n   Face {i+1}:")
                print(f"     Person ID: {result['person_id']}")
                print(f"     Name: {result['person_name']}")
                print(f"     Confidence: {result['confidence']:.3f}")
                print(f"     Recognized: {result['is_recognized']}")
                print(f"     Role: {result['role']}")
                print(f"     Detection score: {result['det_score']:.3f}")
                
                # Draw result
                x1, y1, x2, y2 = result['box']
                color = (0, 255, 0) if result['is_recognized'] else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{result['person_name']} ({result['confidence']:.2f})"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, color, 2)
            
            # Save result
            output_path = "test_recognition_result.jpg"
            cv2.imwrite(output_path, frame)
            print(f"\n   ✅ Result saved to: {output_path}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n   💡 Tip: Run with image path to test recognition")
    print("      python test_face_recognition.py path/to/image.jpg")

# Test threshold adjustment
print(f"\n6. Threshold Recommendations:")
print(f"   Current threshold: {face_system.similarity_threshold}")
print(f"   - Lower threshold (0.3): More lenient, may have false positives")
print(f"   - Current (0.4): Balanced")
print(f"   - Higher threshold (0.5-0.6): Stricter, may miss some faces")
print(f"\n   To adjust: Set FACE_THRESHOLD environment variable")
print(f"   Or modify in api_server_n.py")

db.close()
print("\n" + "="*70)
print("Test Complete")
print("="*70)
