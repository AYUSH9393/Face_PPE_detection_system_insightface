# run_init_db.py
from mongo_db_manager import FaceRecognitionDB

# Initialize database
db = FaceRecognitionDB(
    connection_string="mongodb://face_app:password@localhost:27017/",
    database_name="face_recognition"
)

# Database is now ready!
print("Database initialized successfully!")
db.close()