"""
Test script to verify InsightFace + PPE integration
"""
import os
import sys

def test_imports():
    """Test if all required modules can be imported"""
    print("="*70)
    print("Testing Imports...")
    print("="*70)
    
    try:
        print("\n1. Testing MongoDB connection...")
        from mongo_db_manager import FaceRecognitionDB
        print("   ✅ MongoDB manager imported")
        
        print("\n2. Testing InsightFace system...")
        from insightface_recognition_system import InsightFaceRecognitionSystem
        print("   ✅ InsightFace system imported")
        
        print("\n3. Testing face encoder...")
        from face_encoder import FaceEncoder
        print("   ✅ Face encoder imported")
        
        print("\n4. Testing face recognizer...")
        from face_recognizer import FaceRecognizer
        print("   ✅ Face recognizer imported")
        
        print("\n5. Testing PPE detection...")
        from optimized_ppe_detection import OptimizedPPEDetectionSystem, OptimizedIntegratedSystem
        print("   ✅ PPE detection system imported")
        
        print("\n6. Testing utilities...")
        from utils import load_image, draw_face_box
        print("   ✅ Utilities imported")
        
        print("\n" + "="*70)
        print("✅ All imports successful!")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mongodb_connection():
    """Test MongoDB connection"""
    print("\n" + "="*70)
    print("Testing MongoDB Connection...")
    print("="*70)
    
    try:
        import json
        from mongo_db_manager import FaceRecognitionDB
        
        # Load connection from JSON
        json_path = 'compass-connections.json'
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                connection_string = data['connections'][0]['connectionOptions']['connectionString']
                print(f"\n✅ Loaded connection from {json_path}")
                print(f"   Connection: {connection_string}")
        else:
            connection_string = 'mongodb://localhost:27017/'
            print(f"\n⚠️  JSON file not found, using default")
        
        # Test connection
        db = FaceRecognitionDB(
            connection_string=connection_string,
            database_name='face_recognition'
        )
        
        # Try to get stats
        stats = db.get_database_stats()
        print(f"\n✅ MongoDB connected successfully!")
        print(f"   Total persons: {stats.get('total_persons', 0)}")
        print(f"   Total cameras: {stats.get('total_cameras', 0)}")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"\n❌ MongoDB connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_insightface_system():
    """Test InsightFace system initialization"""
    print("\n" + "="*70)
    print("Testing InsightFace System...")
    print("="*70)
    
    try:
        import json
        from mongo_db_manager import FaceRecognitionDB
        from insightface_recognition_system import InsightFaceRecognitionSystem
        
        # Load connection
        json_path = 'compass-connections.json'
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                connection_string = data['connections'][0]['connectionOptions']['connectionString']
        else:
            connection_string = 'mongodb://localhost:27017/'
        
        # Initialize database
        db = FaceRecognitionDB(
            connection_string=connection_string,
            database_name='face_recognition'
        )
        
        # Initialize InsightFace system
        print("\n🚀 Initializing InsightFace system...")
        face_system = InsightFaceRecognitionSystem(
            db=db,
            similarity_threshold=0.4,
            use_cuda=True,
            model_name='buffalo_l'
        )
        
        # Get stats
        stats = face_system.get_stats()
        print(f"\n✅ InsightFace system initialized!")
        print(f"   Total persons: {stats['total_persons']}")
        print(f"   Total embeddings: {stats['total_embeddings']}")
        print(f"   Similarity threshold: {stats['similarity_threshold']}")
        print(f"   GPU enabled: {stats['gpu_enabled']}")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"\n❌ InsightFace initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ppe_system():
    """Test PPE detection system"""
    print("\n" + "="*70)
    print("Testing PPE Detection System...")
    print("="*70)
    
    try:
        import json
        from mongo_db_manager import FaceRecognitionDB
        from optimized_ppe_detection import OptimizedPPEDetectionSystem
        
        # Load connection
        json_path = 'compass-connections.json'
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                connection_string = data['connections'][0]['connectionOptions']['connectionString']
        else:
            connection_string = 'mongodb://localhost:27017/'
        
        # Initialize database
        db = FaceRecognitionDB(
            connection_string=connection_string,
            database_name='face_recognition'
        )
        
        # Initialize PPE system
        print("\n🦺 Initializing PPE detection system...")
        ppe_model_path = 'models/best.pt' if os.path.exists('models/best.pt') else 'yolov8n.pt'
        
        ppe_system = OptimizedPPEDetectionSystem(
            model_path=ppe_model_path,
            confidence_threshold=0.55,
            db=db,
            use_cuda=True
        )
        
        print(f"\n✅ PPE detection system initialized!")
        print(f"   Model: {ppe_model_path}")
        print(f"   Base confidence: {ppe_system.base_confidence_threshold}")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"\n❌ PPE system initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("InsightFace + PPE Integration Test Suite")
    print("="*70)
    
    results = {
        'imports': False,
        'mongodb': False,
        'insightface': False,
        'ppe': False
    }
    
    # Run tests
    results['imports'] = test_imports()
    
    if results['imports']:
        results['mongodb'] = test_mongodb_connection()
        
        if results['mongodb']:
            results['insightface'] = test_insightface_system()
            results['ppe'] = test_ppe_system()
    
    # Print summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"  Imports:     {'✅ PASS' if results['imports'] else '❌ FAIL'}")
    print(f"  MongoDB:     {'✅ PASS' if results['mongodb'] else '❌ FAIL'}")
    print(f"  InsightFace: {'✅ PASS' if results['insightface'] else '❌ FAIL'}")
    print(f"  PPE System:  {'✅ PASS' if results['ppe'] else '❌ FAIL'}")
    print("="*70)
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed! System is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    print("\nNext steps:")
    print("  1. Run 'python api_server_n.py' to start the API server")
    print("  2. Access API at http://localhost:5000")
    print("  3. Check /api/health endpoint")
    print("\n")
    
    return all_passed


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
