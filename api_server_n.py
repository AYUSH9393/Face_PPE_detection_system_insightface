"""
Enhanced REST API Server for InsightFace Recognition + PPE Detection System
Provides HTTP endpoints for all system operations including PPE compliance
"""

import os
# Set OpenCV/FFMPEG environment variables to fix H.264 decoding errors and reduce log spam
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
    'rtsp_transport;tcp|'
    'fflags;nobuffer|'
    'flags;low_delay|'
    'max_delay;500000|'
    'reorder_queue_size;0|'
    'buffer_size;1024000'
)
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
import cv2
# cv2.setLogLevel(0)

from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import cv2, threading, io, base64, os, time, json
from bson import ObjectId
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps
from werkzeug.utils import secure_filename
from threading import Thread, Lock
from mongo_db_manager import FaceRecognitionDB
from insightface_recognition_system import InsightFaceRecognitionSystem
from camera_detection_manager import CameraDetectionManager
from inference_controller import InferenceController
from alert_engine import AlertEngine

# ============================================================================
# Flask App Configuration
# ============================================================================

app = Flask(__name__)
CORS(app)  # Enable CORS for web clients

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max payload size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Suppress OpenCV warnings
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|analyzeduration;1000000|probesize;1000000'
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
# cv2.setLogLevel(0)

# ============================================================================
# Load MongoDB Connection from JSON
# ============================================================================

def load_mongodb_connection():
    """Load MongoDB connection string from compass-connections.json"""
    try:
        json_path = 'compass-connections.json'
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                if 'connections' in data and len(data['connections']) > 0:
                    connection_string = data['connections'][0]['connectionOptions']['connectionString']
                    print(f"✅ Loaded MongoDB connection from {json_path}")
                    return connection_string
    except Exception as e:
        print(f"⚠️  Failed to load MongoDB connection from JSON: {e}")
    
    # Fallback to environment variable or default
    return os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')

# Initialize database with connection from JSON
mongodb_uri = load_mongodb_connection()
db = FaceRecognitionDB(
    connection_string=mongodb_uri,
    database_name=os.getenv('DATABASE_NAME', 'face_recognition')
)

# ============================================================================
# Initialize InsightFace Recognition System
# ============================================================================

print("🚀 Initializing InsightFace Recognition System...")
try:
    face_system = InsightFaceRecognitionSystem(
        db=db,
        similarity_threshold=float(os.getenv('FACE_THRESHOLD', '0.4')),  # Lowered to 0.3 for better recognition
        use_cuda=True,
        model_name='buffalo_l'
    )
    print("✅ InsightFace System Loaded Successfully")
    print(f"   Model: buffalo_l")
    print(f"   GPU: Enabled")
    print(f"   Threshold: {face_system.similarity_threshold}")
except Exception as e:
    print(f"❌ Error loading InsightFace: {e}")
    import traceback
    traceback.print_exc()
    raise

# Initialize PPE detection system
# ✅ OPTIMIZED: Use optimized PPE system with multi-tier confidence and temporal tracking
import os

ppe_model_path = os.getenv('PPE_MODEL_PATH', 'models/best.pt')

# Check if custom model exists
if not os.path.exists(ppe_model_path):
    print(f"⚠️  Custom PPE model not found: {ppe_model_path}")
    print(f"📥 Using default YOLOv8n model (will auto-download)...")
    ppe_model_path = 'yolov8n.pt'  # Fallback to default

try:
    from optimized_ppe_detection import OptimizedPPEDetectionSystem, OptimizedIntegratedSystem
    ppe_system = OptimizedPPEDetectionSystem(
        model_path=ppe_model_path,
        confidence_threshold=float(os.getenv('PPE_CONFIDENCE_THRESHOLD', '0.55')),  # Base threshold
        db=db, 
        use_cuda=True
    )
    print(f"✅ OPTIMIZED PPE Detection initialized:")
    print(f"   Model: {ppe_model_path}")
    print(f"   Base confidence: {ppe_system.base_confidence_threshold}")
    print(f"   Features: Multi-tier confidence, Temporal tracking, Priority-based reporting")
    
    # Use optimized integrated system
    IntegratedSystemClass = OptimizedIntegratedSystem
except:
    print("⚠️  Falling back to standard PPE detection...")
    from fixed_ppe_detection_system import EnhancedPPEDetectionSystem, ImprovedIntegratedSystem
    ppe_system = EnhancedPPEDetectionSystem(
        model_path=ppe_model_path,
        confidence_threshold=float(os.getenv('PPE_CONFIDENCE_THRESHOLD', '0.6')),
        db=db, 
        use_cuda=True
    )
    print(f"📊 PPE Detection initialized:")
    print(f"   Model: {ppe_model_path}")
    print(f"   Confidence: {ppe_system.confidence_threshold}")
    
    # Use standard integrated system
    IntegratedSystemClass = ImprovedIntegratedSystem

# Initialize alert engine
alert_engine = AlertEngine(db)

# Create integrated system (using selected class)
integrated_system = IntegratedSystemClass(face_system, ppe_system, db, alert_engine)


# =============================
# Camera Detection Manager
# =============================
camera_manager = CameraDetectionManager()

inference_controller = InferenceController(
    camera_manager=camera_manager,
    integrated_system=integrated_system,
)
inference_controller.start()


# def start_active_cameras():
#     cameras = db.get_all_cameras(status="active")
#     for cam in cameras:
#         if cam.get("rtsp_url"):
#             print(f"[CameraManager] Starting {cam['camera_id']}")
#             camera_manager.start_detection(
#                 cam["camera_id"],
#                 cam["rtsp_url"]
#             )

# start_active_cameras()

def start_active_cameras():
    """Start all active cameras (RTSP or USB)"""
    cameras = db.get_all_cameras(status="active")
    
    for cam in cameras:
        camera_id = cam['camera_id']
        
        # Get video source (RTSP URL or USB index)
        source = cam.get("rtsp_url")
        if not source:
            source = cam.get("stream_index", 0)
        
        if source is not None:
            print(f"[CameraManager] Starting {camera_id} (source: {source})")
            camera_manager.start_detection(camera_id, str(source))
        else:
            print(f"[CameraManager] ⚠️ Skipping {camera_id} - no video source configured")



start_active_cameras()


# Active video streams
active_streams = {}
stream_lock = Lock()


# ============================================================================
# Utility Functions
# ============================================================================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def decode_base64_image(base64_string):
    """Decode base64 image to numpy array"""
    try:
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
        
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        return None


def encode_image_base64(image):
    """Encode numpy image to base64"""
    try:
        success, buffer = cv2.imencode('.jpg', image)
        if success:
            return base64.b64encode(buffer).decode('utf-8')
        return None
    except Exception as e:
        return None


def require_admin(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        role = request.headers.get("X-User-Role", "user")
        user = request.headers.get("X-User", "unknown")

        if role.lower() != "admin":
            return jsonify({
                "success": False,
                "error": "Admin access required"
            }), 403

        request.current_user = user
        request.current_role = role
        return fn(*args, **kwargs)

    return wrapper

def _json_ok(data=None, message=None):
    resp = {"success": True}
    if data is not None:
        resp["data"] = data
    if message:
        resp["message"] = message
    return jsonify(resp)

# ============================================================================
# API Endpoints - System Status
# ============================================================================
@app.route("/api/stream/<camera_id>")
def stream_camera(camera_id):
    """✅ OPTIMIZED: Annotated MJPEG stream with improved performance"""
    
    def generate():
        last_frame_time = 0
        frame_interval = 1.0 / 60.0  # 60 FPS cap
        
        while True:
            try:
                current_time = time.time()
                
                # Frame rate limiting
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.001)
                    continue
                
                frame = inference_controller.get_latest_frame(camera_id)

                if frame is None:
                    time.sleep(0.016)
                    continue

                # ✅ OPTIMIZED: Faster JPEG encoding
                ok, jpeg = cv2.imencode(
                    ".jpg", 
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 75,
                     int(cv2.IMWRITE_JPEG_OPTIMIZE), 1]
                )
                
                if not ok:
                    continue

                jpg = jpeg.tobytes()
                
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                    + jpg
                    + b"\r\n"
                )
                
                last_frame_time = current_time

            except GeneratorExit:
                break
            except Exception as e:
                print(f"[STREAM] {camera_id} error:", e)
                time.sleep(0.1)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'Connection': 'keep-alive'
        }
    )


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '2.0.0',
        'features': {
            'face_recognition': True,
            'ppe_detection': True,
            'rtsp_streaming': True,
            'spatial_verification': True
        }
    })


@app.route('/api/stats', methods=['GET'])
def get_system_stats():
    """Get system statistics"""
    try:
        stats = db.get_database_stats()
        
        # Add PPE violation stats
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0)
        
        ppe_violations_today = db.recognition_logs.count_documents({
            'log_type': 'ppe_violation',
            'timestamp': {'$gte': today_start}
        })
        
        unknown_violations_today = db.recognition_logs.count_documents({
            'log_type': 'ppe_violation',
            'is_unknown_person': True,
            'timestamp': {'$gte': today_start}
        })
        
        stats['ppe_violations_today'] = ppe_violations_today
        stats['unknown_violations_today'] = unknown_violations_today
        stats['active_streams'] = len(active_streams)
        
        return jsonify({
            'success': True,
            'data': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# API Endpoints - PPE Detection & Analysis
# ============================================================================

@app.route('/api/ppe/detect', methods=['POST'])
def detect_ppe():
    """
    Detect PPE in uploaded image
    Returns PPE items detected with spatial verification
    """
    try:
        if 'image' not in request.files and 'image_base64' not in request.form:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        # Get image
        if 'image' in request.files:
            file = request.files['image']
            if not allowed_file(file.filename):
                return jsonify({
                    'success': False,
                    'error': 'Invalid file type'
                }), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            img = cv2.imread(filepath)
            os.remove(filepath)
        else:
            img = decode_base64_image(request.form['image_base64'])
        
        if img is None:
            return jsonify({
                'success': False,
                'error': 'Failed to decode image'
            }), 400
        
        # Detect PPE
        ppe_detections = ppe_system.detect_ppe(img)
        
        # Format results
        results = []
        for detection in ppe_detections:
            results.append({
                'category': detection['category'],
                'class_name': detection['class_name'],
                'confidence': float(detection['confidence']),
                'bounding_box': detection['bbox'],
                'center': detection['center']
            })
        
        return jsonify({
            'success': True,
            'ppe_items_detected': len(results),
            'detections': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/ppe/analyze', methods=['POST'])
def analyze_ppe_compliance():
    """
    Analyze PPE compliance in uploaded image
    Combines face recognition + PPE detection
    """
    try:
        if 'image' not in request.files and 'image_base64' not in request.form:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        camera_id = request.form.get('camera_id', 'API_UPLOAD')
        
        # Get image
        if 'image' in request.files:
            file = request.files['image']
            if not allowed_file(file.filename):
                return jsonify({
                    'success': False,
                    'error': 'Invalid file type'
                }), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            img = cv2.imread(filepath)
            os.remove(filepath)
        else:
            img = decode_base64_image(request.form['image_base64'])
        
        if img is None:
            return jsonify({
                'success': False,
                'error': 'Failed to decode image'
            }), 400
        
        # Process frame with integrated system
        results = integrated_system.process_single_frame(img, camera_id)

        if not results:
            return jsonify({"success": False, "error": "No results available"}), 404

        
        # Format response
        compliance_results = []
        for result in results['compliance_results']:
            compliance_results.append({
                'person_id': result['person_id'],
                'person_name': result['person_name'],
                'role': result['role'],
                'is_unknown': result['is_unknown'],
                'face_confidence': float(result['face_confidence']),
                'compliance': {
                    'is_compliant': result['compliance']['is_compliant'],
                    'compliance_percentage': float(result['compliance']['compliance_percentage']),
                    'required_ppe': result['compliance']['required_ppe'],
                    'wearing_ppe': result['compliance']['wearing_ppe'],
                    'missing_ppe': result['compliance']['missing_ppe']
                },
                'is_violation': result['is_violation'],
                'bounding_box': {
                    'x1': int(result['face_bbox'][0]),
                    'y1': int(result['face_bbox'][1]),
                    'x2': int(result['face_bbox'][2]),
                    'y2': int(result['face_bbox'][3])
                }
            })
        
        # Optionally return annotated image
        include_image = request.form.get('include_image', 'false').lower() == 'true'
        annotated_image = None
        
        if include_image:
            annotated_frame = integrated_system.draw_results(img, results)
            annotated_image = f'data:image/jpeg;base64,{encode_image_base64(annotated_frame)}'
        
        response = {
            'success': True,
            'faces_detected': len(results['face_results']),
            'ppe_items_detected': len(results['ppe_detections']),
            'compliance_results': compliance_results,
            'processing_time_ms': float(results['processing_time_ms']),
            'timestamp': results['timestamp'].isoformat()
        }
        
        if annotated_image:
            response['annotated_image'] = annotated_image
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# API Endpoints - PPE Violations
# ============================================================================
@app.route('/api/ppe/violations', methods=['GET'])
def get_ppe_violations():
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 20))

    skip = (page - 1) * page_size

    query = {'log_type': 'ppe_violation'}

    cursor = (
        db.recognition_logs
        .find(query)
        .sort('timestamp', -1)
        .skip(skip)
        .limit(page_size)
    )

    data = []
    for v in cursor:
        v['_id'] = str(v['_id'])
        v['timestamp'] = v['timestamp'].isoformat()
        data.append(v)

    total = db.recognition_logs.count_documents(query)

    return jsonify({
        'success': True,
        'data': data,
        'page': page,
        'page_size': page_size,
        'total': total,
        'total_pages': (total + page_size - 1) // page_size
    })



@app.route('/api/ppe/violations/summary', methods=['GET'])
def get_violations_summary():
    """Get summary of PPE violations"""
    try:
        days = int(request.args.get('days', 7))
        start_date = datetime.utcnow() - timedelta(days=days)
        
        pipeline = [
            {
                '$match': {
                    'log_type': 'ppe_violation',
                    'timestamp': {'$gte': start_date}
                }
            },
            {
                '$group': {
                    '_id': {
                        'date': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$timestamp'}},
                        'camera_id': '$camera_id'
                    },
                    'total_violations': {'$sum': 1},
                    'unknown_violations': {
                        '$sum': {'$cond': ['$is_unknown_person', 1, 0]}
                    }
                }
            },
            {'$sort': {'_id.date': -1}}
        ]
        
        summary = list(db.recognition_logs.aggregate(pipeline))
        
        # Also get violations by PPE type
        ppe_type_pipeline = [
            {
                '$match': {
                    'log_type': 'ppe_violation',
                    'timestamp': {'$gte': start_date}
                }
            },
            {'$unwind': '$missing_ppe'},
            {
                '$group': {
                    '_id': '$missing_ppe',
                    'count': {'$sum': 1}
                }
            },
            {'$sort': {'count': -1}}
        ]
        
        ppe_types = list(db.recognition_logs.aggregate(ppe_type_pipeline))
        
        return jsonify({
            'success': True,
            'period_days': days,
            'daily_summary': summary,
            'violations_by_ppe_type': ppe_types
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# API Endpoints - Person Management (Enhanced)
# ============================================================================

@app.route('/api/persons', methods=['GET'])
def get_all_persons():
    """Get all persons"""
    try:
        status = request.args.get('status', 'active')
        persons = db.get_all_persons(status=status)
        
        for person in persons:
            person['_id'] = str(person['_id'])
            person.pop('embeddings', None)
            
            # Add PPE requirements for role
            role = person.get('role', 'default')
            person['ppe_requirements'] = ppe_system.role_ppe_requirements.get(
                role.lower(),
                ppe_system.role_ppe_requirements.get("default", [])
            )

        
        return jsonify({
            'success': True,
            'data': persons,
            'count': len(persons)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/persons/<person_id>', methods=['GET'])
def get_person(person_id):
    """Get person by ID with PPE compliance stats"""
    try:
        person = db.get_person(person_id)

        if not person:
            return jsonify({
                'success': False,
                'error': 'Person not found'
            }), 404

        # -----------------------------
        # Convert Mongo _id
        # -----------------------------
        person['_id'] = str(person['_id'])

        # -----------------------------
        # Stats (handled in DB layer)
        # -----------------------------
        person['stats'] = db.get_person_stats(person_id, days=30)

        # -----------------------------
        # PPE requirements
        # -----------------------------
        role = person.get('role', 'default')
        person['ppe_requirements'] = ppe_system.role_ppe_requirements.get(
            role.lower(),
            ppe_system.role_ppe_requirements.get("default", [])
        )

        # -----------------------------
        # Recent PPE violations
        # -----------------------------
        violations = list(
            db.recognition_logs.find({
                'log_type': 'ppe_violation',
                'person_id': person_id
            })
            .sort('timestamp', -1)
            .limit(10)
        )

        for v in violations:
            v['_id'] = str(v['_id'])
            v['timestamp'] = v['timestamp'].isoformat()

        person['recent_violations'] = violations

        # -----------------------------
        # CLEAN embeddings safely
        # -----------------------------
        clean_embeddings = []

        for emb in person.get("embeddings", []):
            emb.pop("vector", None)

            img_id = emb.get("image_id")
            if isinstance(img_id, ObjectId):
                emb["image_id"] = str(img_id)

            clean_embeddings.append(emb)

        person["embeddings"] = clean_embeddings

        # -----------------------------
        # FINAL RESPONSE
        # -----------------------------
        return jsonify({
            'success': True,
            'data': person
        })

    except Exception as e:
        print(f"❌ get_person error ({person_id}):", e)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# @app.route('/api/persons/register', methods=['POST'])
# def register_person():
#     """Register a new person with role-based PPE requirements"""
#     try:
#         if 'image' not in request.files:
#             return jsonify({
#                 'success': False,
#                 'error': 'No image provided'
#             }), 400
        
#         file = request.files['image']
        
#         if file.filename == '':
#             return jsonify({
#                 'success': False,
#                 'error': 'No image selected'
#             }), 400
        
#         if not allowed_file(file.filename):
#             return jsonify({
#                 'success': False,
#                 'error': 'Invalid file type'
#             }), 400
        
#         person_id = request.form.get('person_id')
#         name = request.form.get('name')
#         role = request.form.get('role', 'employee')
        
#         if not person_id or not name:
#             return jsonify({
#                 'success': False,
#                 'error': 'person_id and name are required'
#             }), 400
        
#         # Save uploaded file
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         # Register person
#         success = face_system.register_person_from_image(
#             image_path=filepath,
#             person_id=person_id,
#             name=name,
#             email=request.form.get('email', ''),
#             phone=request.form.get('phone', ''),
#             role=role,
#             department=request.form.get('department', ''),
#             tags=request.form.getlist('tags'),
#             access_level=int(request.form.get('access_level', 1)),
#             registered_by=request.form.get('registered_by', 'api')
#         )
        
#         os.remove(filepath)
        
#         if success:
#             ppe_requirements = ppe_system.role_ppe_requirements.get(
#                 role.lower(),
#                 ppe_system.role_ppe_requirements.get("default", [])
#             )
            
#             return jsonify({
#                 'success': True,
#                 'message': f'Person {name} registered successfully',
#                 'person_id': person_id,
#                 'role': role,
#                 'ppe_requirements': ppe_requirements
#             })
#         else:
#             return jsonify({
#                 'success': False,
#                 'error': 'Failed to register person'
#             }), 500
    
#     except Exception as e:
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         }), 500

@app.route('/api/persons/register-folder', methods=['POST'])
def register_person_from_folder_api():
    """
    Register a person using multiple uploaded face images
    (UI equivalent of register_person_from_folder)
    """
    try:
        # -------------------------
        # Validate inputs
        # -------------------------
        person_id = request.form.get('person_id')
        name = request.form.get('name')
        role = request.form.get('role', 'employee')
        department = request.form.get('department', '')

        if not person_id or not name:
            return jsonify({
                "success": False,
                "error": "person_id and name are required"
            }), 400

        if 'images' not in request.files:
            return jsonify({
                "success": False,
                "error": "No images uploaded"
            }), 400

        files = request.files.getlist('images')

        if len(files) < 3:
            return jsonify({
                "success": False,
                "error": "At least 3 face images are required"
            }), 400

        # -------------------------
        # Save images temporarily
        # -------------------------
        temp_dir = Path(app.config['UPLOAD_FOLDER']) / f"reg_{person_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        saved_images = []

        for f in files:
            if not allowed_file(f.filename):
                continue

            filename = secure_filename(f.filename)
            path = temp_dir / filename
            f.save(path)
            saved_images.append(path)

        if not saved_images:
            return jsonify({
                "success": False,
                "error": "No valid images found"
            }), 400

        # -------------------------
        # 🔥 PAUSE inference (CRITICAL)
        # -------------------------
        print("⏸ Pausing inference for registration...")
        inference_controller.pause()

        try:
            success = face_system.register_person_from_folder(
                folder_path=str(temp_dir),
                person_id=person_id,
                name=name,
                role=role,
                department=department,
                registered_by="ui"
            )
        finally:
            # -------------------------
            # ▶ RESUME inference (ALWAYS)
            # -------------------------
            inference_controller.resume()
            print("▶ Inference resumed")


        # Cleanup temp files
        import shutil

        # Cleanup temp directory safely (Windows-safe)
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"⚠️ Temp cleanup warning: {e}")


        if not success:
            return jsonify({
                "success": False,
                "error": "Face registration failed (no valid faces detected)"
            }), 500

        # ✅ Optimized: Embeddings are now reloaded internally by the registration function
        # print("🔄 Reloading embeddings into cache...")
        # try:
        #     face_system.reload_embeddings()
        #     print("✅ Embeddings reloaded successfully")
        # except Exception as e:
        #     print(f"⚠️ Warning: Error reloading embeddings: {e}")
        #     # Don't fail the registration if reload fails


        return jsonify({
            "success": True,
            "message": f"Person {name} registered successfully",
            "person_id": person_id,
            "images_used": len(saved_images),
            "role": role
        })


    except Exception as e:
        # Make sure to resume inference even if there's an error
        try:
            inference_controller.resume()
        except:
            pass
        print("❌ Register-folder error:", e)
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/persons/<person_id>', methods=['PUT'])
def update_person(person_id):
    """Update person information"""
    try:
        data = request.json
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        data.pop('person_id', None)
        data.pop('embeddings', None)
        data.pop('_id', None)
        
        success = db.update_person(person_id, data)
        
        if success:
            # If role changed, return new PPE requirements
            new_role = data.get('role')
            response = {
                'success': True,
                'message': 'Person updated successfully'
            }
            
            if new_role:
                response['ppe_requirements'] = ppe_system.role_ppe_requirements.get(
                    new_role.lower(),
                    ppe_system.role_ppe_requirements.get("default", [])
                )

            
            return jsonify(response)
        else:
            return jsonify({
                'success': False,
                'error': 'Person not found or no changes made'
            }), 404
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/persons/<person_id>', methods=['DELETE'])
def delete_person(person_id):
    """Delete a person"""
    try:
        success = db.delete_person(person_id)
        
        if success:
            face_system.reload_embeddings()
            
            return jsonify({
                'success': True,
                'message': 'Person deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Person not found'
            }), 404
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# API Endpoints - Camera Management
# ============================================================================

@app.route('/api/cameras', methods=['GET'])
def get_all_cameras():
    """Get all cameras"""
    try:
        status = request.args.get('status', 'active')
        cameras = db.get_all_cameras(status=status)
        
        for camera in cameras:
            camera['_id'] = str(camera['_id'])
            camera['is_streaming'] = camera['camera_id'] in active_streams
        
        return jsonify({
            'success': True,
            'data': cameras,
            'count': len(cameras)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/cameras/<camera_id>', methods=['GET'])
def get_camera(camera_id):
    """Get camera by ID"""
    try:
        camera = db.get_camera(camera_id)
        
        if not camera:
            return jsonify({
                'success': False,
                'error': 'Camera not found'
            }), 404
        
        camera['_id'] = str(camera['_id'])
        camera['is_streaming'] = camera_id in active_streams
        
        stats = db.get_camera_stats(camera_id, days=7)
        camera['stats'] = stats
        
        return jsonify({
            'success': True,
            'data': camera
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/cameras/register', methods=['POST'])
def register_camera():
    """Register a new camera"""
    try:
        data = request.json
        
        required_fields = ['camera_id', 'name']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'{field} is required'
                }), 400
        
        success = db.register_camera(**data)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Camera registered successfully',
                'camera_id': data['camera_id']
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to register camera'
            }), 500
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/cameras/<camera_id>', methods=['DELETE'])
def remove_camera(camera_id):
    try:
        # 1️⃣ Check camera exists
        camera = db.get_camera(camera_id)
        if not camera:
            return jsonify({
                'success': False,
                'error': 'Camera not found'
            }), 404

        # 2️⃣ Stop detection / streaming
        try:
            camera_manager.stop_detection(camera_id)
        except Exception:
            pass

        # 3️⃣ Remove from active streams
        with stream_lock:
            active_streams.pop(camera_id, None)

        # 4️⃣ HARD DELETE FROM DB
        result = db.cameras.delete_one({'camera_id': camera_id})

        if result.deleted_count == 0:
            return jsonify({
                'success': False,
                'error': 'Failed to delete camera from database'
            }), 500

        return jsonify({
            'success': True,
            'message': f'Camera {camera_id} deleted permanently'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# API Endpoints - Recognition Logs
# ============================================================================
@app.route('/api/logs', methods=['GET'])
def get_recognition_logs():
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('limit', 50))

    skip = (page - 1) * page_size

    query = {}

    camera_id = request.args.get('camera_id')
    person_id = request.args.get('person_id')

    if camera_id:
        query['camera_id'] = camera_id
    if person_id:
        query['person_id'] = person_id

    cursor = (
        db.recognition_logs
        .find(query)
        .sort('timestamp', -1)
        .skip(skip)
        .limit(page_size)
    )

    data = []
    for log in cursor:
        log['_id'] = str(log['_id'])
        log['timestamp'] = log['timestamp'].isoformat()
        data.append(log)

    total = db.recognition_logs.count_documents(query)

    return jsonify({
        'success': True,
        'data': data,
        'total': total,
        'page': page,
        'page_size': page_size,
        'total_pages': (total + page_size - 1) // page_size
    })


@app.route('/api/logs/image/<image_id>', methods=['GET'])
def get_log_image(image_id):
    """Get face image from log"""
    try:
        from bson import ObjectId
        
        image = db.get_face_image(ObjectId(image_id))
        
        if image is None:
            return jsonify({
                'success': False,
                'error': 'Image not found'
            }), 404
        
        base64_image = encode_image_base64(image)
        
        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{base64_image}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# API Endpoints - Face Recognition
# ============================================================================

@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    """Recognize face from uploaded image"""
    try:
        if 'image' not in request.files and 'image_base64' not in request.form:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        # Get image
        if 'image' in request.files:
            file = request.files['image']
            if not allowed_file(file.filename):
                return jsonify({
                    'success': False,
                    'error': 'Invalid file type'
                }), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            img = cv2.imread(filepath)
            os.remove(filepath)
        else:
            img = decode_base64_image(request.form['image_base64'])
        
        if img is None:
            return jsonify({
                'success': False,
                'error': 'Failed to decode image'
            }), 400
        
        # Detect faces
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, probs = face_system.detector.detect(rgb_img)
        
        if boxes is None or len(boxes) == 0:
            return jsonify({
                'success': True,
                'faces_detected': 0,
                'results': []
            })
        
        results = []
        
        for box, prob in zip(boxes, probs):
            x1, y1, x2, y2 = [int(b) for b in box]
            
            padding = 20
            fx1 = max(0, x1 - padding)
            fy1 = max(0, y1 - padding)
            fx2 = min(img.shape[1], x2 + padding)
            fy2 = min(img.shape[0], y2 + padding)
            
            face_img = img[fy1:fy2, fx1:fx2]
            
            if face_img.size == 0:
                continue
            
            # Recognize
            person_id, confidence, distance = face_system.recognize_face(face_img)
            if prob < 0.75:
                continue
            
            result = {
                'bounding_box': {
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                },
                'detection_confidence': float(prob),
                'person_id': person_id if person_id != "unknown" else None,
                'confidence': float(confidence),
                'distance': float(distance),
                'recognized': person_id != "unknown"
            }
            
            # Get person info if recognized
            if person_id and person_id != "unknown":
                person = db.get_person(person_id)
                if person:
                    role = person.get('role', 'default')
                    result['person'] = {
                        'name': person['name'],
                        'email': person.get('email'),
                        'role': role,
                        'department': person.get('department'),
                        'ppe_requirements': ppe_system.role_ppe_requirements.get(
                            role.lower(),
                            ppe_system.role_ppe_requirements.get("default", [])
                        )
                    }
            
            results.append(result)
        
        return jsonify({
            'success': True,
            'faces_detected': len(results),
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route("/api/cameras/performance")
def camera_performance():
    return jsonify({
        "decode_fps": camera_manager.get_decode_fps(),
        "inference_fps": inference_controller.get_inference_fps()
    })

#system_config
@app.route("/api/settings/ppe", methods=["GET"])
def get_ppe_settings():
    try:
        config = db.system_config.find_one(
            {"config_type": "ppe_rules"},
            {"_id": 0}
        )

        if not config:
            return jsonify({
                "success": False,
                "error": "PPE rules not configured"
            }), 404

        return jsonify({
            "success": True,
            "data": config
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/settings/ppe", methods=["PUT"])
@require_admin
def update_ppe_settings():
    try:
        payload = request.json
        if not payload or "role_rules" not in payload:
            return jsonify({
                "success": False,
                "error": "role_rules is required"
            }), 400

        # 1️⃣ Load existing config
        config = db.system_config.find_one(
            {"config_type": "ppe_rules"}
        ) or {}

        existing_rules = config.get("role_rules", {})

        # 2️⃣ Merge (DO NOT REPLACE)
        for role, ppe_list in payload["role_rules"].items():
            existing_rules[role.lower()] = ppe_list

        # 3️⃣ Persist merged rules
        db.system_config.update_one(
            {"config_type": "ppe_rules"},
            {
                "$set": {
                    "config_type": "ppe_rules",
                    "role_rules": existing_rules
                }
            },
            upsert=True
        )

        # 4️⃣ Hot reload PPE rules
        ppe_system.reload_ppe_rules()

        return jsonify({
            "success": True,
            "message": "PPE rules updated successfully",
            "data": existing_rules
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ======================================================
# Settings – Audit Logs
# ======================================================
@app.route("/api/settings/audit", methods=["GET"])
@require_admin
def get_settings_audit_logs():
    """
    Returns audit logs for system settings changes
    """
    try:
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 50))
        action = request.args.get("action")
        entity = request.args.get("entity")

        skip = (page - 1) * limit

        logs, total = db.get_audit_logs(
            action=action,
            entity=entity,
            limit=limit,
            skip=skip
        )

        return jsonify({
            "success": True,
            "data": logs,
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": (total + limit - 1) // limit
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def get_ppe_requirements_for_role(role: str):
    return ppe_system.role_ppe_requirements.get(
        role.lower(),
        ppe_system.role_ppe_requirements.get("default", [])
    )


# Add these endpoints to api_server_n.py

# ==================================================
# ALERTS CONFIGURATION ENDPOINTS
# ==================================================

@app.route("/api/settings/alerts", methods=["GET"])
def get_alerts_settings():
    """Get alert configuration"""
    try:
        config = db.system_config.find_one(
            {"config_type": "alerts"},
            {"_id": 0}
        )

        if not config:
            # Return default configuration
            default_config = {
                "config_type": "alerts",
                "enable_alerts": True,
                "alert_channels": ["console"],
                "cooldown_seconds": 30,
                "whatsapp": {},
                "buzzer": {"sound": "alert.wav"}
            }
            return jsonify({
                "success": True,
                "data": default_config
            })

        return jsonify({
            "success": True,
            "data": config
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# Add these endpoints to api_server_n.py

@app.route("/api/settings/alerts", methods=["PUT"])
@require_admin
def update_alerts_settings():
    """Update alert configuration with WhatsApp settings"""
    try:
        payload = request.json
        if not payload:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400

        # Update configuration with full WhatsApp settings
        db.system_config.update_one(
            {"config_type": "alerts"},
            {
                "$set": {
                    "config_type": "alerts",
                    "enable_alerts": payload.get("enable_alerts", True),
                    "alert_channels": payload.get("alert_channels", []),
                    "cooldown_seconds": payload.get("cooldown_seconds", 30),
                    "whatsapp": {
                        "sid": payload.get("whatsapp", {}).get("sid", ""),
                        "token": payload.get("whatsapp", {}).get("token", ""),
                        "from": payload.get("whatsapp", {}).get("from", ""),
                        "to": payload.get("whatsapp", {}).get("to", "")
                    },
                    "buzzer": {
                        "sound": payload.get("buzzer", {}).get("sound", "alert.wav")
                    },
                    "updated_at": datetime.utcnow(),
                    "updated_by": request.current_user
                }
            },
            upsert=True
        )

        # Reload alert engine configuration
        alert_engine.reload()

        # Log audit
        db.insert_audit_log({
            "action": "update_alerts",
            "entity": "alerts_config",
            "performed_by": request.current_user,
            "performed_at": datetime.utcnow(),
            "details": {
                "enable_alerts": payload.get("enable_alerts"),
                "channels": payload.get("alert_channels"),
                "cooldown": payload.get("cooldown_seconds")
            }
        })

        return jsonify({
            "success": True,
            "message": "Alert settings updated successfully"
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/cameras/<camera_id>/live_status")
def live_status(camera_id):
    return jsonify({
        "success": True,
        "data": inference_controller.get_live_status(camera_id)
    })


@app.route("/api/stream/raw/<camera_id>")
def stream_raw(camera_id):
    """
    🚀 ULTRA-OPTIMIZED: Raw MJPEG stream with maximum performance
    - Zero latency design
    - No frame rate limiting (client-side handles it)
    - Fastest JPEG encoding
    - Minimal lock contention
    """
    def generate():
        last_frame_hash = None  # Track frame changes to avoid sending duplicates
        
        while True:
            try:
                # ✅ CRITICAL: Minimal lock time - grab frame reference quickly
                with camera_manager.lock:
                    frame = camera_manager.latest_frames.get(camera_id)

                if frame is None:
                    time.sleep(0.01)  # 10ms wait if no frame available
                    continue

                # ✅ OPTIMIZATION: Skip duplicate frames to reduce bandwidth
                # Create a simple hash of frame data
                frame_hash = hash(frame.tobytes()[:1000])  # Hash first 1KB for speed
                if frame_hash == last_frame_hash:
                    time.sleep(0.005)  # 5ms wait before checking again
                    continue
                last_frame_hash = frame_hash

                # ✅ RAW STREAM: No resizing, maintain original quality
                # But use fast encoding settings to prevent lag
                success, buffer = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 75,      # Good quality (75 is standard)
                     int(cv2.IMWRITE_JPEG_OPTIMIZE), 0,      # Disable optimization for speed
                     int(cv2.IMWRITE_JPEG_PROGRESSIVE), 0]   # Disable progressive
                )

                if not success:
                    continue

                jpg = buffer.tobytes()

                # ✅ Streamlined response format
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
                    jpg +
                    b"\r\n"
                )

            except GeneratorExit:
                # Client disconnected
                break
            except Exception as e:
                print(f"[RAW STREAM] {camera_id} error:", e)
                time.sleep(0.05)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'  # Disable nginx buffering if behind proxy
        }
    )

@app.route("/api/settings/alerts/test", methods=["POST"])
@require_admin
def test_whatsapp_alert():
    """Send test WhatsApp message"""
    try:
        payload = request.json
        
        # Validate required fields
        required = ["sid", "token", "from", "to"]
        for field in required:
            if not payload.get(field):
                return jsonify({
                    "success": False,
                    "error": f"Missing required field: {field}"
                }), 400

        # Try to import Twilio
        try:
            from twilio.rest import Client
        except ImportError:
            return jsonify({
                "success": False,
                "error": "Twilio library not installed. Run: pip install twilio"
            }), 500

        # Send test message
        try:
            client = Client(payload["sid"], payload["token"])
            
            message_body = payload.get(
                "message",
                "🔔 Test Alert from PPE Detection System\n\n"
                "If you received this, WhatsApp alerts are working correctly!"
            )
            
            message = client.messages.create(
                body=message_body,
                from_=f"whatsapp:{payload['from']}",
                to=f"whatsapp:{payload['to']}"
            )
            
            return jsonify({
                "success": True,
                "message": "Test WhatsApp sent successfully",
                "message_sid": message.sid,
                "status": message.status
            })
            
        except Exception as twilio_error:
            error_msg = str(twilio_error)
            
            # Provide helpful error messages
            if "authenticate" in error_msg.lower():
                error_msg = "Authentication failed. Check your Account SID and Auth Token."
            elif "not a valid phone number" in error_msg.lower():
                error_msg = "Invalid phone number format. Include country code (e.g., +919876543210)"
            elif "sandbox" in error_msg.lower():
                error_msg = "WhatsApp Sandbox not configured. Join sandbox or get approved number."
            
            return jsonify({
                "success": False,
                "error": f"Twilio error: {error_msg}"
            }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/settings/alerts/status", methods=["GET"])
def get_alert_status():
    """Get current alert system status"""
    try:
        config = db.system_config.find_one(
            {"config_type": "alerts"},
            {"_id": 0}
        )
        
        if not config:
            return jsonify({
                "success": True,
                "data": {
                    "configured": False,
                    "enabled": False,
                    "channels": []
                }
            })
        
        # Check if WhatsApp is properly configured
        whatsapp_config = config.get("whatsapp", {})
        whatsapp_configured = all([
            whatsapp_config.get("sid"),
            whatsapp_config.get("token"),
            whatsapp_config.get("from"),
            whatsapp_config.get("to")
        ])
        
        # Check if Twilio is installed
        twilio_available = False
        try:
            import twilio
            twilio_available = True
        except ImportError:
            pass
        
        return jsonify({
            "success": True,
            "data": {
                "configured": True,
                "enabled": config.get("enable_alerts", False),
                "channels": config.get("alert_channels", []),
                "cooldown_seconds": config.get("cooldown_seconds", 30),
                "whatsapp": {
                    "configured": whatsapp_configured,
                    "library_available": twilio_available,
                    "has_sid": bool(whatsapp_config.get("sid")),
                    "has_token": bool(whatsapp_config.get("token")),
                    "from_number": whatsapp_config.get("from", ""),
                    "to_number": whatsapp_config.get("to", "")
                },
                "buzzer": {
                    "sound_file": config.get("buzzer", {}).get("sound", "alert.wav")
                }
            }
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    
# ==================================================
# ROLE DELETION ENDPOINT
# ==================================================

@app.route("/api/settings/ppe/<role>", methods=["DELETE"])
@require_admin
def delete_ppe_role(role):
    """Delete a PPE role"""
    try:
        role = role.lower()

        # Prevent deletion of default roles
        if role in ["default", "visitor"]:
            return jsonify({
                "success": False,
                "error": f"Cannot delete default role '{role}'"
            }), 400

        # Load current config
        config = db.system_config.find_one(
            {"config_type": "ppe_rules"}
        )

        if not config or role not in config.get("role_rules", {}):
            return jsonify({
                "success": False,
                "error": f"Role '{role}' not found"
            }), 404

        # Remove role
        role_rules = config.get("role_rules", {})
        del role_rules[role]

        # Update database
        db.system_config.update_one(
            {"config_type": "ppe_rules"},
            {
                "$set": {
                    "role_rules": role_rules,
                    "updated_at": datetime.utcnow(),
                    "updated_by": request.current_user
                }
            }
        )

        # Reload PPE system
        ppe_system.reload_ppe_rules()

        # Log audit
        db.insert_audit_log({
            "action": "delete_role",
            "entity": f"role:{role}",
            "performed_by": request.current_user,
            "performed_at": datetime.utcnow()
        })

        return jsonify({
            "success": True,
            "message": f"Role '{role}' deleted successfully"
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Attendance
@app.route("/api/persons/<person_id>/attendance")
def attendance(person_id):
    """
    ✅ FIXED: Proper date range filtering for attendance
    Returns attendance records for specified period
    """
    try:
        days = int(request.args.get("days", 30))
        
        # Calculate date range
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        
        # ✅ CRITICAL: Normalize to start/end of day in UTC
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end = end.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        print(f"📅 Fetching attendance for {person_id}: {start} to {end}")
        
        # Query attendance collection with proper date filtering
        data = db.get_attendance(person_id, start, end)
        
        # Convert MongoDB documents to JSON-serializable format
        result = []
        for record in data:
            # Convert ObjectId
            record['_id'] = str(record['_id'])
            
            # ✅ CRITICAL: Convert date to ISO string
            if 'date' in record and isinstance(record['date'], datetime):
                record['date'] = record['date'].isoformat()
            
            # Convert event timestamps
            for event in record.get("events", []):
                if 'timestamp' in event and isinstance(event['timestamp'], datetime):
                    event['timestamp'] = event['timestamp'].isoformat()
            
            result.append(record)
        
        print(f"✅ Returning {len(result)} attendance records")
        
        return _json_ok(result)
    
    except Exception as e:
        print(f"❌ Attendance endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    




@app.route("/api/performance/fps")
def get_fps_metrics():
    return jsonify({
        "decode_fps": camera_manager.get_decode_fps(),
        "inference_fps": inference_controller.get_inference_fps()
    })


@app.route("/api/cameras/fps")
def camera_inference_fps():
    """
    Frontend-compatible endpoint
    Returns INFERENCE FPS per camera
    """
    return jsonify(inference_controller.get_inference_fps(per_camera=True))


# Add these endpoints to api_server_n.py

@app.route("/api/cameras/health", methods=["GET"])
def get_cameras_health():
    """
    Get health status for all cameras
    Useful for monitoring and diagnostics
    """
    try:
        health_data = camera_manager.get_all_camera_status()
        
        return jsonify({
            "success": True,
            "data": health_data,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/cameras/<camera_id>/health", methods=["GET"])
def get_camera_health(camera_id):
    """Get health status for specific camera"""
    try:
        health = camera_manager.get_stream_health(camera_id)
        
        return jsonify({
            "success": True,
            "camera_id": camera_id,
            "health": health,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/cameras/<camera_id>/restart", methods=["POST"])
@require_admin
def restart_camera(camera_id):
    """
    Manually restart a camera
    Useful when a camera is stuck
    """
    try:
        success = camera_manager.restart_camera(camera_id)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Camera {camera_id} restarted"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to restart camera"
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/cameras/test-rtsp", methods=["POST"])
@require_admin
def test_rtsp_url():
    """
    Test an RTSP URL without adding it to the system
    Useful for validating camera URLs before registration
    """
    try:
        data = request.json
        rtsp_url = data.get("rtsp_url")
        
        if not rtsp_url:
            return jsonify({
                "success": False,
                "error": "rtsp_url is required"
            }), 400
        
        # Try to connect
        import cv2
        print(f"🧪 Testing RTSP URL: {rtsp_url}")
        
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5s timeout
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        
        if not cap.isOpened():
            cap.release()
            return jsonify({
                "success": False,
                "error": "Failed to open RTSP stream",
                "details": "Could not connect to the stream. Check URL, network, and camera status."
            }), 400
        
        # Try to read a frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({
                "success": False,
                "error": "Failed to read frame from stream",
                "details": "Connected but could not retrieve video frames."
            }), 400
        
        # Get frame info
        height, width = frame.shape[:2]
        
        return jsonify({
            "success": True,
            "message": "RTSP stream is accessible",
            "resolution": {
                "width": width,
                "height": height
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/cameras/status")
def camera_status():
    cameras = db.get_all_cameras(status="active")
    status = []

    for cam in cameras:
        has_alert = db.recognition_logs.find_one({
            "camera_id": cam["camera_id"],
            "log_type": "ppe_violation",
            "resolved": False
        }) is not None

        status.append({
            "camera_id": cam["camera_id"],
            "has_alert": has_alert
        })

    return jsonify({"success": True, "data": status})


"""
Add these diagnostic endpoints to api_server_n.py
to help debug detection and logging issues
"""

# Add these imports at the top if not already present:
from datetime import datetime, timedelta

# Add these endpoints to your Flask app:

@app.route("/api/diagnostics/detection", methods=["GET"])
def diagnostics_detection():
    """
    Get detection system status
    Helps debug why detections might not be working
    """
    try:
        camera_id = request.args.get("camera_id")
        
        # Check inference controller status
        inference_running = inference_controller.running
        
        # Get latest results
        with inference_controller.lock:
            if camera_id:
                latest_result = inference_controller.latest_results.get(camera_id)
                violations = inference_controller.live_violations.get(camera_id, [])
            else:
                latest_result = dict(inference_controller.latest_results)
                violations = dict(inference_controller.live_violations)
        
        # Check camera manager status
        with camera_manager.lock:
            frames = {k: v is not None for k, v in camera_manager.latest_frames.items()}
        
        # Check face recognition system
        face_system_status = {
            "embeddings_loaded": len(face_system.embeddings_cache),
            "persons_registered": len(db.get_all_persons("active")),
            "threshold": face_system.threshold
        }
        
        # Check PPE system
        ppe_system_status = {
            "model_loaded": ppe_system.model is not None,
            "confidence_threshold": ppe_system.confidence_threshold,
            "role_rules_loaded": len(ppe_system.role_ppe_requirements)
        }
        
        return jsonify({
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "inference_controller": {
                "running": inference_running,
                "fps": inference_controller.get_inference_fps(),
                "per_camera_fps": inference_controller.get_inference_fps(per_camera=True)
            },
            "camera_manager": {
                "active_cameras": list(frames.keys()),
                "frames_available": frames,
                "decode_fps": camera_manager.get_decode_fps()
            },
            "face_recognition": face_system_status,
            "ppe_detection": ppe_system_status,
            "latest_results": latest_result,
            "recent_violations": violations
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route("/api/diagnostics/violations", methods=["GET"])
def diagnostics_violations():
    """
    Check violation logging status
    Shows recent violations and logging stats
    """
    try:
        # Get violations from last 24 hours
        yesterday = datetime.utcnow() - timedelta(days=1)
        
        recent_violations = list(
            db.recognition_logs.find({
                "log_type": "ppe_violation",
                "timestamp": {"$gte": yesterday}
            })
            .sort("timestamp", -1)
            .limit(20)
        )
        
        # Convert ObjectId to string
        for v in recent_violations:
            v["_id"] = str(v["_id"])
            v["timestamp"] = v["timestamp"].isoformat()
        
        # Get stats
        total_violations = db.recognition_logs.count_documents({
            "log_type": "ppe_violation"
        })
        
        violations_today = db.recognition_logs.count_documents({
            "log_type": "ppe_violation",
            "timestamp": {"$gte": datetime.utcnow().replace(hour=0, minute=0, second=0)}
        })
        
        violations_last_hour = db.recognition_logs.count_documents({
            "log_type": "ppe_violation",
            "timestamp": {"$gte": datetime.utcnow() - timedelta(hours=1)}
        })
        
        # Check cooldown status
        cooldown_status = {}
        for key, timestamp in integrated_system.violation_cooldown.items():
            time_left = integrated_system.VIOLATION_COOLDOWN_SEC - (time.time() - timestamp)
            if time_left > 0:
                cooldown_status[key] = f"{time_left:.1f}s remaining"
        
        return jsonify({
            "success": True,
            "statistics": {
                "total_violations": total_violations,
                "violations_today": violations_today,
                "violations_last_hour": violations_last_hour
            },
            "recent_violations": recent_violations,
            "cooldown_status": cooldown_status,
            "cooldown_seconds": integrated_system.VIOLATION_COOLDOWN_SEC
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route("/api/diagnostics/reset-cooldown", methods=["POST"])
def reset_violation_cooldown():
    """
    Reset violation cooldown (for testing)
    Use this if you want to force re-logging of violations
    """
    try:
        integrated_system.violation_cooldown.clear()
        
        return jsonify({
            "success": True,
            "message": "Violation cooldown reset. All violations will be logged again."
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/diagnostics/ppe-rules", methods=["GET"])
def diagnostics_ppe_rules():
    """
    Check current PPE rules configuration
    """
    try:
        # Get from PPE system
        rules = ppe_system.role_ppe_requirements
        
        # Get from database
        db_config = db.system_config.find_one({"config_type": "ppe_rules"})
        
        return jsonify({
            "success": True,
            "loaded_rules": rules,
            "database_config": db_config.get("role_rules") if db_config else None,
            "rules_match": rules == (db_config.get("role_rules") if db_config else {})
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    print(f"""
    {'='*70}
    🚀 Enhanced Face Recognition + PPE Detection API Server
    {'='*70}
    Server running on: http://{HOST}:{PORT}
    Database: {db.db.name}
    Registered persons: {len(face_system.embeddings_cache)}
    
    ✨ New Features:
    - PPE Detection & Compliance Analysis
    - Spatial PPE Verification (helmet on head, not in hand)
    - Unknown Person Handling (treated as visitors)
    - Role-based PPE Requirements
    - Real-time Violation Logging
    
    📋 Main API Endpoints:
    
    System:
    - GET  /api/health
    - GET  /api/stats
    
    PPE:
    - GET  /api/ppe/requirements
    - GET  /api/ppe/requirements/<role>
    - PUT  /api/ppe/requirements/<role>
    - POST /api/ppe/detect
    - POST /api/ppe/analyze
    - GET  /api/ppe/violations
    - GET  /api/ppe/violations/summary
    
    Persons:
    - GET  /api/persons
    - GET  /api/persons/<person_id>
    - POST /api/persons/register
    - PUT  /api/persons/<person_id>
    - DELETE /api/persons/<person_id>
    
    Cameras:
    - GET  /api/cameras
    - GET  /api/cameras/<camera_id>
    - POST /api/cameras/register
    
    Recognition:
    - POST /api/recognize
    - GET  /api/logs
    - GET  /api/logs/image/<image_id>
    
    Press Ctrl+C to stop
    {'='*70}
    """)
    
    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)