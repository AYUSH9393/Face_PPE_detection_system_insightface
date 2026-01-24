# SiteSecureVision - AI-Powered PPE Compliance & Face Recognition System

**SiteSecureVision** is a real-time computer vision system designed for construction sites and industrial environments. It strictly enforces Personal Protective Equipment (PPE) compliance while monitoring authorized personnel access through facial recognition.

## 🚀 Key Features

*   **Real-time Face Recognition**: Uses InsightFace (Buffalo_L) for high-accuracy identification.
*   **PPE Violation Detection**: YOLOv8-based detection for Helmets, Vests, Goggles, etc.
*   **Strict Anatomical Validation**: Ensures PPE is worn correctly (e.g., helmet on head, not in hand).
*   **GPU Acceleration**: Optimized for NVIDIA RTX GPUs using CUDA 11.8.
*   **Live Monitoring Dashboard**: Modern PyQt6 desktop interface with multi-camera support.
*   **Video Streaming**: Low-latency MJPEG streaming optimized for minimal bandwidth.
*   **Attendance Tracking**: Automated daily attendance logging.

## 🛠️ System Requirements

*   **OS**: Windows 10/11 or Linux
*   **Python**: 3.10+
*   **GPU**: NVIDIA GPU with CUDA support (Recommended: RTX 3050 or higher)
*   **CUDA Toolkit**: Version 11.8

## 📦 Installation Guide

### 1. Prerequisite: Install CUDA Toolkit 11.8
Ensure you have the NVIDIA CUDA Toolkit 11.8 installed on your system.
[Download Here](https://developer.nvidia.com/cuda-11-8-0-download-archive)

### 2. Clone the Repository
```bash
git clone <your-repo-url>
cd SiteSecureVision
```

### 3. Create a Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 4. Install Dependencies (GPU Version)
It is critical to install the GPU-enabled versions of PyTorch and ONNX Runtime.

```bash
# 1. Install GPU-enabled PyTorch (for YOLO)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2. Install GPU-enabled ONNX Runtime (for InsightFace)
pip install onnxruntime-gpu==1.16.3

# 3. Install strictly compatible NumPy version
pip install "numpy<2"

# 4. Install remaining dependencies
pip install -r requirements.txt
```

*(Note: If `requirements.txt` is missing, install manually: `ultralytics opencv-python-headless insightface flask flask-cors pymongo pyqt6 requests`)*

## 🚦 Usage

### 1. Start the Backend Server
The server handles inference, database connections, and video streaming.
```bash
python api_server_n.py
```
*Wait until you see "Rocket Initializing InsightFace..."*

### 2. Start the Desktop Dashboard
Open a new terminal window and run:
```bash
cd desktop_ui
python main.py
```

## ⚙️ Configuration

### Adding Cameras
1.  Click **"Add New Camera"** in the dashboard.
2.  **ID**: Unique identifier (e.g., `CAM_01`).
3.  **RTSP/Index**:
    *   For USB Webcam: Enter `0` (or `1`, `2`...).
    *   For IP Camera: Enter the RTSP URL (e.g., `rtsp://user:pass@192.168.1.100:554/stream`).

### PPE Rules
Default rules require **Safety Helmet** and **Reflective Vest**.
You can modify role-based rules in the MongoDB database (`system_config` collection).

## 🔧 Troubleshooting

*   **"Error loading cublasLt64_12.dll"**: You installed the wrong ONNX Runtime. Run `pip install onnxruntime-gpu==1.16.3`.
*   **Laggy Stream**: Ensure you are running on the GPU. The backend console should show `GPU Speedup` metrics.
*   **Camera not connecting**: Verify the RTSP link in VLC Media Player first.

## 📝 License
Proprietary - Internal Use Only
