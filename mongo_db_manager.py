"""
MongoDB Database Manager for Face Recognition System
Handles all database operations, GridFS image storage, and queries
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError, ConnectionFailure
from gridfs import GridFS
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple
import io
from PIL import Image
import cv2
from bson import ObjectId

class FaceRecognitionDB:
    """
    Database manager for face recognition system
    Handles MongoDB operations and GridFS image storage
    """
    
    def __init__(self, connection_string: str = "mongodb://localhost:27017/",
                 database_name: str = "face_recognition"):
        """
        Initialize MongoDB connection
        
        Args:
            connection_string: MongoDB connection URI
            database_name: Name of the database
        """
        try:
            self.client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            print(f"✅ Connected to MongoDB successfully")
            
            self.db = self.client[database_name]
            self.fs = GridFS(self.db)  # For storing images
            
            # Collections
            self.persons = self.db.persons
            self.cameras = self.db.cameras
            self.recognition_logs = self.db.recognition_logs
            self.attendance = self.db.attendance
            self.system_config = self.db.system_config
            self.system_audit = self.db.audit_logs
            
            # Create indexes for performance
            self._create_indexes()
            
            print(f"✅ Database '{database_name}' initialized")
            
        except ConnectionFailure as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            raise
    
    def _create_indexes(self):
        """Create database indexes for optimal query performance"""
        
        # Persons collection indexes
        try:
            self.persons.create_index("person_id", unique=True)
            self.persons.create_index("email")
            self.persons.create_index("status")
            self.persons.create_index([("last_seen", DESCENDING)])
            
            # Cameras collection indexes
            self.cameras.create_index("camera_id", unique=True)
            self.cameras.create_index("status")
            
            # Recognition logs indexes
            self.recognition_logs.create_index([("person_id", ASCENDING), 
                                                ("timestamp", DESCENDING)])
            self.recognition_logs.create_index([("camera_id", ASCENDING), 
                                                ("timestamp", DESCENDING)])
            self.recognition_logs.create_index([("timestamp", DESCENDING)])
            self.recognition_logs.create_index("is_alert")
            
            # Attendance indexes
            self.attendance.create_index([("person_id", ASCENDING), 
                                         ("date", DESCENDING)], unique=True)
            
            # Audit logs indexes
            self.db.audit_logs.create_index([("performed_at", DESCENDING)])
            self.db.audit_logs.create_index("entity")
            self.db.audit_logs.create_index("action")

            
            print("✅ Database indexes created successfully")
        except Exception as e:
            print(f"⚠️ Warning: Some indexes may already exist - {e}")
    
    # ========================================================================
    # IMAGE STORAGE OPERATIONS (GridFS)
    # ========================================================================
    
    def store_face_image(self, image: np.ndarray, metadata: Dict = None) -> ObjectId:
        """
        Store face image in GridFS
        
        Args:
            image: Face image as numpy array (BGR format)
            metadata: Optional metadata dictionary
            
        Returns:
            ObjectId of stored image
        """
        # Convert numpy array to JPEG bytes
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("Failed to encode image")
        
        image_bytes = buffer.tobytes()
        
        # Store in GridFS
        file_id = self.fs.put(
            image_bytes,
            filename=f"face_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg",
            content_type="image/jpeg",
            upload_date=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        return file_id
    
    def get_face_image(self, image_id: ObjectId) -> Optional[np.ndarray]:
        """
        Retrieve face image from GridFS
        
        Args:
            image_id: ObjectId of the image
            
        Returns:
            Image as numpy array or None if not found
        """
        try:
            grid_out = self.fs.get(image_id)
            image_bytes = grid_out.read()
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return image
        except Exception as e:
            print(f"❌ Error retrieving image {image_id}: {e}")
            return None
    
    def delete_face_image(self, image_id: ObjectId) -> bool:
        """Delete face image from GridFS"""
        try:
            self.fs.delete(image_id)
            return True
        except Exception as e:
            print(f"❌ Error deleting image {image_id}: {e}")
            return False
    
    # ========================================================================
    # PERSON OPERATIONS
    # ========================================================================
    
    def register_person(self, person_id: str, name: str, embedding: np.ndarray,
                       face_image: np.ndarray, **kwargs) -> bool:
        """
        Register a new person in the database
        
        Args:
            person_id: Unique identifier for the person
            name: Person's name
            embedding: 512-dimensional face embedding
            face_image: Face image as numpy array
            **kwargs: Additional fields (email, phone, role, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Store face image in GridFS
            image_id = self.store_face_image(face_image, 
                metadata={"person_id": person_id, "type": "registration"})
            
            # Create person document
            person_doc = {
                "person_id": person_id,
                "name": name,
                "email": kwargs.get("email", ""),
                "phone": kwargs.get("phone", ""),
                "role": kwargs.get("role", "employee"),
                "department": kwargs.get("department", ""),
                "embeddings": [
                    {
                        "embedding_id": f"emb_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        "vector": embedding.tolist(),
                        "image_id": image_id,
                        "quality_score": kwargs.get("quality_score", 0.0),
                        "created_at": datetime.utcnow(),
                        "is_primary": True
                    }
                ],
                "status": "active",
                "registered_by": kwargs.get("registered_by", "system"),
                "registered_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "last_seen": None,
                "notes": kwargs.get("notes", ""),
                "tags": kwargs.get("tags", []),
                "access_level": kwargs.get("access_level", 1),
                "total_detections": 0,
                "avg_confidence": 0.0
            }
            
            # Insert into database
            self.persons.insert_one(person_doc)
            print(f"✅ Person registered: {name} ({person_id})")
            return True
            
        except DuplicateKeyError:
            print(f"❌ Person ID '{person_id}' already exists")
            return False
        except Exception as e:
            print(f"❌ Error registering person: {e}")
            return False
    
    def add_person_embedding(self, person_id: str, embedding: np.ndarray,
                            face_image: np.ndarray, is_primary: bool = False) -> bool:
        """
        Add additional face embedding for an existing person
        
        Args:
            person_id: Person's unique identifier
            embedding: New face embedding
            face_image: Face image
            is_primary: Whether this should be the primary embedding
            
        Returns:
            True if successful
        """
        try:
            # Store face image
            image_id = self.store_face_image(face_image,
                metadata={"person_id": person_id, "type": "additional"})
            
            # If this is primary, unset other primary embeddings
            if is_primary:
                self.persons.update_one(
                    {"person_id": person_id},
                    {"$set": {"embeddings.$[].is_primary": False}}
                )
            
            # Add new embedding
            new_embedding = {
                "embedding_id": f"emb_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                "vector": embedding.tolist(),
                "image_id": image_id,
                "quality_score": 0.0,
                "created_at": datetime.utcnow(),
                "is_primary": is_primary
            }
            
            result = self.persons.update_one(
                {"person_id": person_id},
                {
                    "$push": {"embeddings": new_embedding},
                    "$set": {"updated_at": datetime.utcnow()}
                }
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            print(f"❌ Error adding embedding: {e}")
            return False
    
    def get_person(self, person_id: str) -> Optional[Dict]:
        """Get person document by person_id"""
        return self.persons.find_one({"person_id": person_id})
    
    def get_all_persons(self, status: str = "active") -> List[Dict]:
        """Get all persons with specified status"""
        return list(self.persons.find({"status": status}))
    
    def get_all_embeddings(self, status: str = "active") -> Dict[str, List[np.ndarray]]:
        """
        Get all embeddings for face recognition
        
        Returns:
            Dictionary mapping person_id to list of embeddings
        """
        embeddings_dict = {}
        
        persons = self.persons.find({"status": status})
        
        for person in persons:
            person_id = person["person_id"]
            embeddings = []
            
            for emb_data in person.get("embeddings", []):
                vector = np.array(emb_data["vector"], dtype=np.float32)
                embeddings.append(vector)
            
            if embeddings:
                embeddings_dict[person_id] = embeddings
        
        return embeddings_dict
    
    def update_person(self, person_id: str, update_fields: Dict) -> bool:
        """Update person information"""
        try:
            update_fields["updated_at"] = datetime.utcnow()
            
            result = self.persons.update_one(
                {"person_id": person_id},
                {"$set": update_fields}
            )
            
            return result.modified_count > 0
        except Exception as e:
            print(f"❌ Error updating person: {e}")
            return False
    
    def delete_person(self, person_id: str) -> bool:
        """Delete person and associated images"""
        try:
            # Get person document
            person = self.get_person(person_id)
            if not person:
                return False
            
            # Delete all associated images from GridFS
            for emb_data in person.get("embeddings", []):
                self.delete_face_image(emb_data["image_id"])
            
            # Delete person document
            self.persons.delete_one({"person_id": person_id})
            
            print(f"🗑️ Deleted person: {person_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting person: {e}")
            return False
    
    def update_last_seen(self, person_id: str, confidence: float):
        """Update last seen timestamp and statistics"""
        person = self.get_person(person_id)
        if not person:
            return
        
        total_detections = person.get("total_detections", 0) + 1
        avg_confidence = person.get("avg_confidence", 0.0)
        
        # Calculate new average confidence
        new_avg = ((avg_confidence * (total_detections - 1)) + confidence) / total_detections
        
        self.persons.update_one(
            {"person_id": person_id},
            {
                "$set": {
                    "last_seen": datetime.utcnow(),
                    "total_detections": total_detections,
                    "avg_confidence": round(new_avg, 4)
                }
            }
        )
    
    # ========================================================================
    # CAMERA OPERATIONS
    # ========================================================================
    
    def register_camera(self, camera_id: str, name: str, **kwargs) -> bool:
        """Register a new camera"""
        try:
            camera_doc = {
                "camera_id": camera_id,
                "name": name,
                "location": kwargs.get("location", ""),
                "rtsp_url": kwargs.get("rtsp_url", ""),
                "stream_index": kwargs.get("stream_index"),
                "resolution": {
                    "width": kwargs.get("width", 1920),
                    "height": kwargs.get("height", 1080)
                },
                "fps": kwargs.get("fps", 30),
                "skip_frames": kwargs.get("skip_frames", 2),
                "detection_threshold": kwargs.get("detection_threshold", 0.8),
                "recognition_threshold": kwargs.get("recognition_threshold", 0.7),
                "min_face_size": kwargs.get("min_face_size", 40),
                "status": "active",
                "is_online": False,
                "last_heartbeat": datetime.utcnow(),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "zone": kwargs.get("zone", "default"),
                "access_level_required": kwargs.get("access_level_required", 1)
            }
            
            self.cameras.insert_one(camera_doc)
            print(f"✅ Camera registered: {name} ({camera_id})")
            return True
            
        except DuplicateKeyError:
            print(f"❌ Camera ID '{camera_id}' already exists")
            return False
        except Exception as e:
            print(f"❌ Error registering camera: {e}")
            return False
    
    def get_camera(self, camera_id: str) -> Optional[Dict]:
        """Get camera configuration"""
        return self.cameras.find_one({"camera_id": camera_id})
    
    def get_all_cameras(self, status: str = "active") -> List[Dict]:
        """Get all cameras with specified status"""
        return list(self.cameras.find({"status": status}))
    
    def update_camera_heartbeat(self, camera_id: str, is_online: bool = True):
        """Update camera heartbeat timestamp"""
        self.cameras.update_one(
            {"camera_id": camera_id},
            {
                "$set": {
                    "is_online": is_online,
                    "last_heartbeat": datetime.utcnow()
                }
            }
        )
    

    def update_camera(self, camera_id: str, updates: dict) -> bool:
        """
        Update camera fields (e.g. status, name, location)
        """
        result = self.cameras.update_one(
            {'camera_id': camera_id},
            {'$set': updates}
        )
        return result.matched_count > 0

    # ========================================================================
    # RECOGNITION LOG OPERATIONS
    # ========================================================================
    
    def log_recognition(self, person_id: str, person_name: str, camera_id: str,
                       confidence: float, face_image: np.ndarray,
                       full_frame: Optional[np.ndarray] = None,
                       **kwargs) -> str:
        """
        Log a face recognition event
        
        Returns:
            log_id of the created log
        """
        try:
            # Store face image
            face_img_id = self.store_face_image(face_image,
                metadata={"person_id": person_id, "camera_id": camera_id, "type": "detection"})
            
            # Optionally store full frame
            full_frame_id = None
            if full_frame is not None and kwargs.get("store_full_frames", False):
                full_frame_id = self.store_face_image(full_frame,
                    metadata={"camera_id": camera_id, "type": "full_frame"})
            
            # Get camera info
            camera = self.get_camera(camera_id)
            camera_name = camera["name"] if camera else camera_id
            location = camera["location"] if camera else "Unknown"
            
            # Create log document
            log_id = f"LOG_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            log_doc = {
                "log_id": log_id,
                "person_id": person_id,
                "person_name": person_name,
                "confidence": confidence,
                "status": kwargs.get("status", "recognized"),
                "camera_id": camera_id,
                "camera_name": camera_name,
                "location": location,
                "bounding_box": kwargs.get("bounding_box", {}),
                "face_image_id": face_img_id,
                "full_frame_id": full_frame_id,
                "timestamp": datetime.utcnow(),
                "detection_time_ms": kwargs.get("detection_time_ms", 0.0),
                "recognition_time_ms": kwargs.get("recognition_time_ms", 0.0),
                "embedding_used": kwargs.get("embedding_used", ""),
                "distance": kwargs.get("distance", 0.0),
                "all_matches": kwargs.get("all_matches", []),
                "is_alert": kwargs.get("is_alert", False),
                "alert_reason": kwargs.get("alert_reason", None),
                "alert_sent": False,
                "indexed": True,
                "archived": False
            }
            
            self.recognition_logs.insert_one(log_doc)
            
            # Update person's last_seen if recognized
            if person_id and person_id != "unknown":
                self.update_last_seen(person_id, confidence)
            
            return log_id
            
        except Exception as e:
            print(f"❌ Error logging recognition: {e}")
            return ""
    
    def get_recent_logs(self, limit: int = 100, camera_id: Optional[str] = None,
                       person_id: Optional[str] = None) -> List[Dict]:
        """Get recent recognition logs with optional filters"""
        query = {}
        if camera_id:
            query["camera_id"] = camera_id
        if person_id:
            query["person_id"] = person_id
        
        return list(self.recognition_logs.find(query)
                   .sort("timestamp", DESCENDING)
                   .limit(limit))
    
    def get_logs_by_date_range(self, start_date: datetime, end_date: datetime,
                               **filters) -> List[Dict]:
        """Get logs within a date range"""
        query = {
            "timestamp": {
                "$gte": start_date,
                "$lte": end_date
            }
        }
        query.update(filters)
        
        return list(self.recognition_logs.find(query).sort("timestamp", DESCENDING))
    
    def get_alert_logs(self, limit: int = 50) -> List[Dict]:
        """Get recent alert logs"""
        return list(self.recognition_logs.find({"is_alert": True})
                   .sort("timestamp", DESCENDING)
                   .limit(limit))
    
    # ========================================================================
    # ATTENDANCE OPERATIONS (Optional)
    # ========================================================================
    
    def mark_attendance(self, person_id: str, camera_id: str, log_id: str,
                       event_type: str = "check_in") -> bool:
        """Mark attendance for a person"""
        try:
            # today = datetime.utcnow().date()
            today = datetime.utcnow().replace(
                hour=0, minute=0, second=0, microsecond=0
            )

            camera = self.get_camera(camera_id)
            
            event = {
                "type": event_type,
                "timestamp": datetime.utcnow(),
                "camera_id": camera_id,
                "location": camera["location"] if camera else "Unknown",
                "confidence": 0.0,  # Can be passed as parameter
                "log_id": log_id
            }
            
            # Update or insert attendance record
            self.attendance.update_one(
                {
                    "person_id": person_id,
                    "date": today
                },
                {
                    "$push": {"events": event},
                    "$setOnInsert": {
                        "person_id": person_id,
                        "date": today,
                        "status": "present"
                    }
                },
                upsert=True
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Error marking attendance: {e}")
            return False
    
  

    def get_attendance(self, person_id: str, start_date: datetime,
                    end_date: datetime) -> List[Dict]:
        """
        ✅ FIXED: Get attendance records for a person within date range
        Ensures proper UTC date comparison
        """
        # ✅ Normalize dates to start/end of day in UTC
        start = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        print(f"🔍 DB Query: person_id={person_id}, start={start}, end={end}")
        
        # Query with proper date range
        cursor = self.attendance.find({
            "person_id": person_id,
            "date": {
                "$gte": start,
                "$lte": end
            }
        }).sort("date", -1)
        
        results = list(cursor)
        
        print(f"✅ DB returned {len(results)} records")
        
        # Debug: Print first record if exists
        if results:
            first = results[0]
            print(f"   First record date: {first.get('date')}")
        
        return results

    
    # ========================================================================
    # STATISTICS & ANALYTICS
    # ========================================================================
    
    def get_person_stats(self, person_id: str, days: int = 30) -> Dict:
        start_date = datetime.utcnow() - timedelta(days=days)

        logs = list(self.recognition_logs.find({
            "person_id": person_id,
            "timestamp": {"$gte": start_date}
        }))

        if not logs:
            return {
                "detections": 0,
                "avg_confidence": 0.0,
                "cameras_seen": [],
                "first_seen": None,
                "last_seen": None
            }

        cameras_seen = set()
        confidences = []

        for log in logs:
            # Camera
            cam = log.get("camera_id")
            if cam:
                cameras_seen.add(cam)

            # Confidence (SAFE)
            conf = log.get("confidence")
            if isinstance(conf, (int, float)):
                confidences.append(conf)

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "detections": len(logs),
            "avg_confidence": round(avg_conf, 4),
            "cameras_seen": list(cameras_seen),
            "first_seen": logs[-1].get("timestamp"),
            "last_seen": logs[0].get("timestamp")
        }

    
    def get_camera_stats(self, camera_id: str, days: int = 7) -> Dict:
        """Get statistics for a camera"""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        total_detections = self.recognition_logs.count_documents({
            "camera_id": camera_id,
            "timestamp": {"$gte": start_date}
        })
        
        recognized = self.recognition_logs.count_documents({
            "camera_id": camera_id,
            "status": "recognized",
            "timestamp": {"$gte": start_date}
        })
        
        unknown = self.recognition_logs.count_documents({
            "camera_id": camera_id,
            "status": "unknown",
            "timestamp": {"$gte": start_date}
        })
        
        return {
            "total_detections": total_detections,
            "recognized": recognized,
            "unknown": unknown,
            "recognition_rate": (recognized / total_detections * 100) if total_detections > 0 else 0
        }
    
    # ========================================================================
    # UTILITY OPERATIONS
    # ========================================================================
    
    def cleanup_old_logs(self, days: int = 90) -> int:
        """Archive or delete logs older than specified days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Archive logs
        result = self.recognition_logs.update_many(
            {"timestamp": {"$lt": cutoff_date}, "archived": False},
            {"$set": {"archived": True}}
        )
        
        print(f"📦 Archived {result.modified_count} old logs")
        return result.modified_count
    
    def get_database_stats(self) -> Dict:
        """Get overall database statistics"""
        return {
            "total_persons": self.persons.count_documents({}),
            "active_persons": self.persons.count_documents({"status": "active"}),
            "total_cameras": self.cameras.count_documents({}),
            "active_cameras": self.cameras.count_documents({"status": "active"}),
            "online_cameras": self.cameras.count_documents({"is_online": True}),
            "total_logs": self.recognition_logs.count_documents({}),
            "today_logs": self.recognition_logs.count_documents({
                "timestamp": {"$gte": datetime.utcnow().replace(hour=0, minute=0, second=0)}
            }),
            "gridfs_files": self.db.fs.files.count_documents({}),
            "database_size_mb": round(self.db.command("dbstats")["dataSize"] / (1024 * 1024), 2)
        }
    
    def close(self):
        """Close database connection"""
        self.client.close()
        print("🔌 Database connection closed")

    

    # ================================
    # Audit Logs
    # ================================
    def insert_audit_log(self, doc: dict):
        return self.db.audit_logs.insert_one(doc)


    def get_audit_logs(
        self,
        action: str = None,
        entity: str = None,
        limit: int = 50,
        skip: int = 0
    ):
        query = {}

        if action:
            query["action"] = action
        if entity:
            query["entity"] = entity

        cursor = (
            self.db.audit_logs
            .find(query)
            .sort("performed_at", -1)
            .skip(skip)
            .limit(limit)
        )

        results = []
        for log in cursor:
            log["_id"] = str(log["_id"])
            results.append(log)

        total = self.db.audit_logs.count_documents(query)

        return results, total


    # ================================
    # System Config
    # ================================
    def get_config(self, config_type: str) -> Optional[Dict]:
        return self.system_config.find_one({"config_type": config_type})


    def update_config(self, config_type: str, updates: Dict) -> bool:
        updates["updated_at"] = datetime.utcnow()
        result = self.system_config.update_one(
            {"config_type": config_type},
            {"$set": updates}
        )
        return result.matched_count > 0

# ============================================================================
# Usage Example
# ============================================================================
if __name__ == "__main__":
    # Initialize database
    db = FaceRecognitionDB(
        connection_string="mongodb://localhost:27017/",
        database_name="face_recognition"
    )
    
    # Print database stats
    stats = db.get_database_stats()
    print("\n📊 Database Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    db.close()