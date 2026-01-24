"""
Quick face recognition diagnostic tool
Usage: python quick_test.py path/to/image.jpg
"""
import sys
import cv2
import numpy as np
from mongo_db_manager import FaceRecognitionDB
from insightface_recognition_system import InsightFaceRecognitionSystem

def test_recognition(image_path, threshold=0.3):
    print("="*70)
    print("Quick Face Recognition Test")
    print("="*70)
    
    # Initialize
    print("\n1. Connecting to database...")
    db = FaceRecognitionDB('mongodb://localhost:27017/', 'face_recognition')
    
    print("\n2. Initializing InsightFace...")
    face_system = InsightFaceRecognitionSystem(
        db=db,
        similarity_threshold=threshold,
        use_cuda=True,
        model_name='buffalo_l'
    )
    
    print(f"\n3. System ready!")
    print(f"   Persons loaded: {len(face_system.embeddings_cache)}")
    print(f"   Threshold: {threshold}")
    
    # List persons
    print(f"\n4. Registered persons:")
    for person_id, data in face_system.embeddings_cache.items():
        print(f"   - {person_id}: {data['name']} ({data['role']}) - {len(data['embeddings'])} embeddings")
    
    # Load and process image
    print(f"\n5. Processing image: {image_path}")
    frame = cv2.imread(image_path)
    
    if frame is None:
        print("   ❌ Failed to load image!")
        return
    
    print(f"   ✅ Image loaded: {frame.shape}")
    
    # Process
    results = face_system.process_frame(frame, "TEST", store_logs=False)
    
    print(f"\n6. Results:")
    print(f"   Faces detected: {len(results)}")
    
    if len(results) == 0:
        print("   ⚠️  No faces detected!")
        print("\n   Possible reasons:")
        print("   - Face too small")
        print("   - Poor lighting")
        print("   - Face not frontal")
        print("   - Image quality too low")
    else:
        for i, result in enumerate(results):
            print(f"\n   Face {i+1}:")
            print(f"     Person ID: {result['person_id']}")
            print(f"     Name: {result['person_name']}")
            print(f"     Confidence: {result['confidence']:.4f}")
            print(f"     Threshold: {threshold}")
            print(f"     Recognized: {'✅ YES' if result['is_recognized'] else '❌ NO'}")
            print(f"     Role: {result['role']}")
            print(f"     Detection score: {result['det_score']:.4f}")
            
            if not result['is_recognized']:
                print(f"\n     💡 Confidence ({result['confidence']:.4f}) is below threshold ({threshold})")
                print(f"        Try lowering threshold or registering more images")
            
            # Draw on image
            x1, y1, x2, y2 = result['box']
            color = (0, 255, 0) if result['is_recognized'] else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{result['person_name']} ({result['confidence']:.2f})"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Save result
    output_path = "quick_test_result.jpg"
    cv2.imwrite(output_path, frame)
    print(f"\n7. Result saved to: {output_path}")
    
    db.close()
    print("\n" + "="*70)
    print("Test Complete!")
    print("="*70)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_test.py path/to/image.jpg [threshold]")
        print("Example: python quick_test.py test.jpg 0.3")
        sys.exit(1)
    
    image_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    
    test_recognition(image_path, threshold)
