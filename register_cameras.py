# register_cameras.py
from mongo_db_manager import FaceRecognitionDB

db = FaceRecognitionDB()

# Register USB webcam
db.register_camera(
    camera_id="CAM_001",
    name="Main Entrance Camera",
    location="Building A - Main Entrance",
    stream_index=0,  # USB camera index
    fps=30,
    skip_frames=2,
    detection_threshold=0.8,
    recognition_threshold=0.7,
    zone="entrance",
    access_level_required=1
)

# Register IP camera (RTSP)
db.register_camera(
    camera_id="CAM_002",
    name="Exit Camera",
    location="Exit",
    rtsp_url="rtsp://admin:%40dmin%4089@192.168.1.141:554/Streaming/channels/101/",
    fps=25,
    skip_frames=3,
    zone="exit"
)

db.register_camera(
    camera_id="CAM_003",
    name="Middle  Camera",
    location="Building A - Middle Hallway",
    stream_index=1,  # USB camera index
    fps=30,
    skip_frames=2,
    detection_threshold=0.8,
    recognition_threshold=0.7,
    zone="Middle Hallway",
    access_level_required=1
)

print("Cameras registered!")
db.close()