import time
import threading
from datetime import datetime
from collections import deque, defaultdict
from typing import Dict, List
import numpy as np
from threading import Lock, Thread
from queue import Queue, Empty


class InferenceController(Thread):
    """
    ✅ FIXED: Run face recognition on EVERY frame
    ✅ HIGH PERFORMANCE: Decoupled Inference & Result Processing
    No throttling - maximum recognition speed
    """

    def __init__(self, camera_manager, integrated_system, max_batch_size=8):
        super().__init__(daemon=True)

        self.camera_manager = camera_manager
        self.system = integrated_system
        self.max_batch_size = max_batch_size
        self.paused = False

        self.running = True
        self.lock = Lock()
        self.live_status = {}

        # FPS tracking
        self.inference_times = deque(maxlen=60)
        self.per_camera_times = defaultdict(lambda: deque(maxlen=30))
        
        self.latest_results = {}
        self.live_violations = {}
        self.annotated_frames = {}
        
        # ✅ Decoupled Result Processing
        self.result_queue = Queue(maxsize=30)
        self.processor_thread = Thread(target=self._result_processing_loop, daemon=True)
        self.processor_thread.start()

        print(
            f"🧠 InferenceController READY | "
            f"batch={self.max_batch_size} | "
            f"mode=CONTINUOUS | "
            f"threads=GPU+CPU_POST_PROCESS"
        )

    def run(self):
        print("🚀 InferenceController started - CONTINUOUS RECOGNITION MODE")

        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            try:
                # Get latest frames
                with self.camera_manager.lock:
                    frames_snapshot = dict(self.camera_manager.latest_frames)

                if not frames_snapshot:
                    time.sleep(0.01)
                    continue

                batch_frames = []
                batch_camera_ids = []

                for cam_id, frame in frames_snapshot.items():
                    if frame is None:
                        continue
                    batch_frames.append(frame)
                    batch_camera_ids.append(cam_id)

                if not batch_frames:
                    time.sleep(0.01)
                    continue

                # ✅ ALWAYS run face recognition
                run_face = True
                
                # Run inference (GPU Heavy)
                inference_start = time.time()
                results = self.system.process_frames_with_ppe_batch(
                    batch_frames,
                    batch_camera_ids,
                    run_face=run_face
                )
                inference_end = time.time()

                # Update FPS tracking (Inference Only)
                with self.lock:
                    self.inference_times.append(inference_end)
                    for cam_id in batch_camera_ids:
                        self.per_camera_times[cam_id].append(inference_end)

                # Push to Queue for Side-Effect Processing (CPU Heavy)
                if not self.result_queue.full():
                    self.result_queue.put((results, batch_frames, batch_camera_ids))
                else:
                    # If queue full, we skip post-processing to keep up with live feed
                    pass
                    
                # Very small sleep to yield GIL
                time.sleep(0.001)

            except Exception as e:
                print(f"❌ InferenceController error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.05)

    def _result_processing_loop(self):
        """Separate thread to handle drawing and status updates"""
        while self.running:
            try:
                item = self.result_queue.get(timeout=1)
            except Empty:
                continue
                
            if item is None: break
            
            results, frames, camera_ids = item
            
            try:
                # Store results and update live_status
                with self.lock:
                    for idx, cam_id in enumerate(camera_ids):
                        result = results[idx]
                        frame = frames[idx]

                        # Store full results
                        self.latest_results[cam_id] = result

                        # Create annotated frame
                        has_violation = any(
                            not r["compliance"]["is_compliant"]
                            for r in result.get("compliance_results", [])
                        )

                        if has_violation:
                            annotated = self.system.draw_results(frame, result)
                        else:
                            annotated = frame

                        self.annotated_frames[cam_id] = annotated

                        # Update live_status for status panel
                        if cam_id not in self.live_status:
                            self.live_status[cam_id] = {}

                        status_map = self.live_status[cam_id]
                        current_time = datetime.utcnow()

                        # Add/update all detected persons
                        for r in result.get("compliance_results", []):
                            track_id = r.get("track_id") or r["person_id"]

                            status_map[track_id] = {
                                "track_id": track_id,
                                "person_id": r["person_id"],
                                "person_name": r["person_name"],
                                "is_unknown": r["is_unknown"],
                                "missing_ppe": r["compliance"]["missing_ppe"],
                                "is_compliant": r["compliance"]["is_compliant"],
                                "timestamp": current_time.isoformat(),
                                "last_seen": time.time()
                            }

                        # TTL cleanup - remove entries older than 5 seconds
                        now = time.time()
                        stale_keys = [
                            k for k, v in status_map.items()
                            if now - v.get("last_seen", 0) > 5
                        ]
                        for k in stale_keys:
                            del status_map[k]
                            
            except Exception as e:
                print(f"❌ Result processing error: {e}")
            finally:
                self.result_queue.task_done()

    def get_latest_frame(self, camera_id: str):
        """Get annotated frame for /api/stream endpoint"""
        with self.lock:
            return self.annotated_frames.get(camera_id)

    def get_inference_fps(self, per_camera=False):
        """Calculate inference FPS"""
        with self.lock:
            if per_camera:
                fps = {}
                for cam_id, dq in self.per_camera_times.items():
                    if len(dq) >= 2:
                        time_span = dq[-1] - dq[0]
                        fps[cam_id] = round((len(dq) - 1) / time_span, 2) if time_span > 0 else 0.0
                    else:
                        fps[cam_id] = 0.0
                return fps

            if len(self.inference_times) < 2:
                return 0.0

            time_span = self.inference_times[-1] - self.inference_times[0]
            count = len(self.inference_times) - 1
            
            return round(count / time_span, 2) if time_span > 0 else 0.0

    
    def get_live_status(self, camera_id: str):
        """Return live detection status for UI"""
        with self.lock:
            status_dict = self.live_status.get(camera_id, {})
            
            # Convert dict to list, sorted by most recent
            status_list = sorted(
                status_dict.values(),
                key=lambda x: x.get("last_seen", 0),
                reverse=True
            )
            
            # Return top 5 most recent detections
            return status_list[:5]

    def stop(self):
        self.running = False
        print("🛑 InferenceController stopped")
    
    def pause(self):
        self.paused = True
        print("⏸ InferenceController paused")

    def resume(self):
        self.paused = False
        print("▶ InferenceController resumed")