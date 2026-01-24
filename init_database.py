"""
Initialize Face Recognition Database
Run this script ONCE to setup the database
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure
from datetime import datetime
import sys

def initialize_database():
    """Initialize MongoDB database for face recognition system"""
    
    print("="*70)
    print("MongoDB Database Initialization for Face Recognition System")
    print("="*70)
    
    # Connect to MongoDB
    try:
        client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("✅ Connected to MongoDB successfully!")
    except ConnectionFailure as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        print("\nTroubleshooting:")
        print("1. Check if MongoDB service is running:")
        print("   net start MongoDB")
        print("2. Check if MongoDB is listening on port 27017")
        sys.exit(1)
    
    # Create/Access database
    db = client['face_recognition']
    print(f"✅ Database 'face_recognition' created/accessed")
    
    # Create collections
    print("\n📁 Creating collections...")
    
    collections = {
        'persons': 'Store person information and face embeddings',
        'cameras': 'Store camera configurations',
        'recognition_logs': 'Store face recognition events',
        'attendance': 'Store attendance records',
        'system_config': 'Store system configuration'
    }
    
    for collection_name, description in collections.items():
        if collection_name not in db.list_collection_names():
            db.create_collection(collection_name)
            print(f"   ✅ Created collection: {collection_name} - {description}")
        else:
            print(f"   ℹ️  Collection exists: {collection_name}")
    
    # Create indexes for performance
    print("\n🔍 Creating indexes...")
    
    try:
        # Persons collection indexes
        db.persons.create_index("person_id", unique=True)
        db.persons.create_index("email")
        db.persons.create_index("status")
        db.persons.create_index([("last_seen", DESCENDING)])
        print("   ✅ Persons indexes created")
        
        # Cameras collection indexes
        db.cameras.create_index("camera_id", unique=True)
        db.cameras.create_index("status")
        print("   ✅ Cameras indexes created")
        
        # Recognition logs indexes
        db.recognition_logs.create_index([("person_id", ASCENDING), ("timestamp", DESCENDING)])
        db.recognition_logs.create_index([("camera_id", ASCENDING), ("timestamp", DESCENDING)])
        db.recognition_logs.create_index([("timestamp", DESCENDING)])
        db.recognition_logs.create_index("is_alert")
        print("   ✅ Recognition logs indexes created")
        
        # Attendance indexes
        db.attendance.create_index([("person_id", ASCENDING), ("date", DESCENDING)], unique=True)
        print("   ✅ Attendance indexes created")
        
    except Exception as e:
        print(f"   ⚠️  Some indexes may already exist: {e}")
    
    # Insert default system configuration
    print("\n⚙️  Setting up default configuration...")
    
    default_config = {
        'config_type': 'global',
        'recognition_threshold': 0.7,
        'detection_threshold': 0.8,
        'max_distance': 1.0,
        'batch_size': 4,
        'queue_size': 10,
        'worker_threads': 4,
        'store_unknown_faces': True,
        'store_full_frames': False,
        'retention_days': 90,
        'enable_alerts': True,
        'alert_channels': ['console'],
        'updated_at': datetime.utcnow(),
        'updated_by': 'system'
    }
    
    if db.system_config.count_documents({'config_type': 'global'}) == 0:
        db.system_config.insert_one(default_config)
        print("   ✅ Default configuration inserted")
    else:
        print("   ℹ️  Configuration already exists")
    
    # Database statistics
    print("\n📊 Database Statistics:")
    print(f"   Database name: {db.name}")
    print(f"   Collections: {len(db.list_collection_names())}")
    print(f"   Total documents: {sum(db[col].count_documents({}) for col in db.list_collection_names())}")
    
    # GridFS setup (for storing images)
    print("\n🖼️  GridFS initialized for image storage")
    
    print("\n" + "="*70)
    print("✅ Database initialization completed successfully!")
    print("="*70)
    print("\nNext steps:")
    print("1. Register cameras: python register_cameras.py")
    print("2. Register persons: python register_persons.py")
    print("3. Start face recognition: python run_face_recognition.py")
    print("="*70)
    
    client.close()

if __name__ == "__main__":
    initialize_database()