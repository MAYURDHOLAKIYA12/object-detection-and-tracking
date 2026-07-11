# YOLOv8 + SORT Object Detection and Tracking System

A professional, production-ready Object Detection and Tracking System that integrates a YOLOv8 detector with a custom, pure Python implementation of the SORT (Simple Online Realtime Tracking) algorithm. Built with a responsive dark-themed Streamlit dashboard.

---

## 🌟 Features

- **YOLOv8 Detection:** Leverages PyTorch-based YOLOv8n (nano) for high-frequency real-time object classification and localization.
- **Custom SORT Tracker:** Features a pure Python implementation of the SORT algorithm utilizing:
  - Linear assignment matching (Hungarian algorithm) via `scipy.optimize.linear_sum_assignment`.
  - Motion estimation via Kalman Filtering with the `filterpy` library.
  - Consistent ID tracking across frame occlusions and short-term exits.
- **Multiclass Support:** Specifically tuned to filter and track the following COCO classes:
  - 👤 Person (ID 0)
  - 🚗 Car (ID 2)
  - 🏍️ Motorcycle (ID 3)
  - 🚌 Bus (ID 5)
  - 🚚 Truck (ID 7)
- **Interactive Metrics & Visuals:**
  - Real-time rolling average FPS calculations (over a 30-frame sliding window).
  - Track count metrics with change-deltas from the previous frame.
  - Per-class breakdown distribution table updated dynamically.
  - Class-specific BGR OpenCV bounding boxes and IDs.
- **Flexible Stream Inputs:** Webcam capture or local video file upload (`.mp4`, `.avi`, `.mov`) with execution progress bar.
- **Export Capabilities:** Option to write and export processed streams as `.mp4` video files.

---

## 📋 Prerequisites

Ensure your environment meets the following baseline requirements:
- **Python 3.8+**
- **pip** (Python package installer)
- Operating System: macOS, Linux, or Windows (highly optimized for Mac)

---

## 🛠️ Step-by-Step Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd Object-Detection-Tracking
   ```

2. **Create and activate a virtual environment (Recommended):**
   - **macOS / Linux:**
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```
   - **Windows:**
     ```cmd
     python -m venv venv
     venv\Scripts\activate
     ```

3. **Install the required dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Application

To launch the Streamlit dashboard server, execute the following command:
```bash
streamlit run app.py
```
This automatically opens the interface in your default web browser (typically at `http://localhost:8501`).

### 🎥 Tracking from Webcam
1. Set the **Source Selector** in the sidebar to **"Webcam"**.
2. Adjust your **Confidence Threshold** slider if necessary.
3. Check/uncheck visualization toggles to customize overlays.
4. Click **START** to run. Click **STOP** to end and view session metrics.

### 📁 Tracking from Video File
1. Set the **Source Selector** in the sidebar to **"Upload Video File"**.
2. Click the file uploader and select an `.mp4`, `.avi`, or `.mov` file.
3. Click **START**. A progress bar will track completion percentage.
4. The video will stop automatically at the final frame, outputting a complete session stats summary.

---

## 🎛️ Control Panel Parameters

- **Source Selector:** Toggles between active camera stream (Webcam) or uploaded file path.
- **Upload Video File:** Drag-and-drop or select local video streams for processing.
- **Confidence Threshold (0.1 - 0.9):** Detections with confidence values lower than this threshold are filtered out. Default value is **0.40**.
- **Show Bounding Boxes:** Toggles OpenCV-annotated bounding boxes and classification percentages on the stream.
- **Show Tracking IDs:** Toggles displaying unique track identifiers mapped to the object.
- **Save Output Video:** Automatically writes and saves the processed video to the output folder.

---

## 💾 Output Location

- **YOLOv8 Model Weights:** Saved automatically on first execution inside `models/` (specifically `models/yolov8n.pt`).
- **Exported Videos:** Saved as `outputs/output_TIMESTAMP.mp4` when **Save Output Video** is enabled.

---

## 🔧 Troubleshooting Section

### 1. PyTorch / CUDA Configuration Errors
- **Issue:** The application runs slowly, or fails on GPU device setup.
- **Solution:** `detector.py` uses PyTorch to auto-detect CUDA capabilities. If GPU support is unavailable or PyTorch is configured incorrectly, it falls back gracefully to CPU. For specific GPU acceleration, ensure matching `torch` + `cuda` library configurations.

### 2. Streamlit Port Already in Use
- **Issue:** Streamlit warns that port `8501` is already in use.
- **Solution:** Streamlit automatically finds the next available port (e.g., `8502`). You can also specify an exact port using:
  ```bash
  streamlit run app.py --server.port 8080
  ```

### 3. OpenCV Webcam Connection Failures
- **Issue:** Web camera fails to start, showing standard error messages.
- **Solution:** Verify that your webcam is connected to the machine and that your browser/OS has permitted terminal camera access. Close other applications (Zoom, Teams, etc.) that may be holding locks on camera device index `0`.

---

## 📸 Sample Usage Screenshots

Below is a visual layout placeholder of the dashboard interface:

```
+-------------------------------------------------------------+
| 🎯 Object Detection and Tracking System                      |
|                                                             |
| +-----------------------------------+ +-------------------+ |
| |                                   | | 📊 Real-Time      | |
| |                                   | |   Metrics         | |
| |                                   | |                   | |
| |            LIVE VIDEO             | |   👤 Persons: 3   | |
| |               FEED                | |   🚗 Vehicles: 5  | |
| |             (640x480)             | |   📊 FPS: 24.3    | |
| |                                   | |   🎯 IDs: 12      | |
| |                                   | +-------------------+ |
| |                                   | | Breakdown Table   | |
| |                                   | | Class | Count     | |
| |                                   | | ----- | -----     | |
| |                                   | | Car   | 4         | |
| +-----------------------------------+ +-------------------+ |
+-------------------------------------------------------------+
```
*(Upload your own screenshot to this repository to document visual performance)*
