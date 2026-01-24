"""
Check MongoDB database and person embeddings
"""
from mongo_db_manager import FaceRecognitionDB
import json

# Connect to database
db = FaceRecognitionDB('mongodb://localhost:27017/', 'face_recognition')

print("="*70)
print("MongoDB Database Check")
print("="*70)

# Get all persons
persons = db.get_all_persons()
print(f"\n✅ Total persons in database: {len(persons)}")

# Check each person
print("\nPerson Details:")
print("-"*70)
for person in persons:
    person_id = person.get('person_id', 'N/A')
    name = person.get('name', 'N/A')
    role = person.get('role', 'N/A')
    embeddings = person.get('embeddings', [])
    
    print(f"\n  Person ID: {person_id}")
    print(f"  Name: {name}")
    print(f"  Role: {role}")
    print(f"  Embeddings: {len(embeddings)}")
    
    # Check embedding structure
    if embeddings:
        first_emb = embeddings[0]
        has_vector = 'vector' in first_emb
        vector_len = len(first_emb.get('vector', [])) if has_vector else 0
        
        print(f"  Has vector: {has_vector}")
        print(f"  Vector length: {vector_len}")
        print(f"  Quality score: {first_emb.get('quality_score', 'N/A')}")
    else:
        print("  ⚠️  No embeddings found!")

# Check system config
print("\n" + "="*70)
print("PPE Rules Configuration")
print("="*70)

ppe_config = db.system_config.find_one({"config_type": "ppe_rules"})
if ppe_config:
    print("✅ PPE rules configured")
    role_rules = ppe_config.get('role_rules', {})
    print(f"   Configured roles: {', '.join(role_rules.keys())}")
else:
    print("❌ PPE rules NOT configured")
    print("   Run: python 002_add_ppe_rules.py")

# Check cameras
print("\n" + "="*70)
print("Camera Configuration")
print("="*70)

cameras = db.get_all_cameras()
print(f"✅ Total cameras: {len(cameras)}")
for cam in cameras:
    cam_id = cam.get('camera_id', 'N/A')
    name = cam.get('name', 'N/A')
    status = cam.get('status', 'N/A')
    print(f"  - {cam_id}: {name} ({status})")

db.close()

print("\n" + "="*70)
print("Summary")
print("="*70)
print(f"✅ Database connection: OK")
print(f"✅ Persons loaded: {len(persons)}")
print(f"✅ Cameras configured: {len(cameras)}")

# Check if embeddings have vectors
total_embeddings = sum(len(p.get('embeddings', [])) for p in persons)
embeddings_with_vectors = sum(
    1 for p in persons 
    for emb in p.get('embeddings', []) 
    if 'vector' in emb and len(emb.get('vector', [])) > 0
)

print(f"✅ Total embeddings: {total_embeddings}")
print(f"✅ Embeddings with vectors: {embeddings_with_vectors}")

if embeddings_with_vectors == 0:
    print("\n⚠️  WARNING: No embeddings have vectors!")
    print("   This means faces won't be recognized.")
    print("   You need to re-register persons with InsightFace.")
elif embeddings_with_vectors < total_embeddings:
    print(f"\n⚠️  WARNING: Only {embeddings_with_vectors}/{total_embeddings} embeddings have vectors")
    print("   Some persons may not be recognized.")

print("\n")
