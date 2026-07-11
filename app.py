import os
import time
from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from detector import YOLODetector, CLASS_NAMES
from tracker import Sort

# Set page config
st.set_page_config(
    page_title="Object Detection & Tracking Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional dark-themed UI using custom CSS
st.markdown(
    """
    <style>
    /* Dark mode override */
    .stApp {
        background-color: #0e1117;
        color: #f0f2f6;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161920;
        border-right: 1px solid #2d3139;
    }
    
    /* Sidebar text colors */
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
    }

    /* Metric card wrapper */
    div[data-testid="stMetric"] {
        background-color: #1a1d24;
        border: 1px solid #2d3139;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        margin-bottom: 15px;
        transition: transform 0.2s, border-color 0.2s;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #4b5563;
    }
    
    /* Customize metric values */
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }

    /* Customize metric labels */
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }

    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }

    /* Style dataframe tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid #2d3139;
        border-radius: 8px;
        background-color: #161920;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def resize_letterbox(img: np.ndarray, target_width: int = 640, target_height: int = 480) -> np.ndarray:
    """
    Resizes an image to target dimensions while maintaining aspect ratio by adding black padding.
    Prevents squishing and distortion, greatly improving object detection confidence.
    """
    h, w = img.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

def main() -> None:
    """
    Main entry point for the Streamlit dashboard application.
    Sets up the dashboard, parses sidebar parameters, and runs the video processing loop.
    """
    st.title("🎯 Object Detection and Tracking System")
    st.markdown("A real-time analytics platform featuring **YOLOv8** and a custom **SORT** tracker.")
    
    # Ensure standard directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # ------------------ SIDEBAR CONTROLS ------------------
    st.sidebar.title("⚙️ Control Panel")
    
    # Source selector
    source: str = st.sidebar.selectbox(
        "Source Selector",
        options=["Webcam", "Upload Video File"],
        index=1
    )
    
    # Video file uploader
    uploaded_file = None
    if source == "Upload Video File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Video File",
            type=["mp4", "avi", "mov"],
            help="Supported formats: mp4, avi, mov"
        )
        
    # YOLO Model selector
    model_option: str = st.sidebar.selectbox(
        "YOLOv8 Model Size",
        options=["yolov8n.pt (Nano - Faster)", "yolov8s.pt (Small - Balanced)", "yolov8m.pt (Medium - More Accurate)"],
        index=1,
        help="Select yolov8s.pt (Small) or yolov8m.pt (Medium) to improve person and vehicle detection accuracy."
    )
    model_name: str = model_option.split(" ")[0]

    # Confidence threshold slider
    conf_thresh: float = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.4,
        step=0.05
    )
    
    # Toggles
    show_boxes: bool = st.sidebar.toggle("Show Bounding Boxes", value=True)
    show_ids: bool = st.sidebar.toggle("Show Tracking IDs", value=True)
    save_output: bool = st.sidebar.toggle("Save Output Video", value=False)
    
    st.sidebar.markdown("---")
    
    # Initialize start/stop state
    if "running" not in st.session_state:
        st.session_state.running = False
    if "summary" not in st.session_state:
        st.session_state.summary = None
        
    # Start/Stop buttons in sidebar
    col_start, col_stop = st.sidebar.columns(2)
    start_btn = col_start.button("START", use_container_width=True, disabled=st.session_state.running)
    stop_btn = col_stop.button("STOP", use_container_width=True, disabled=not st.session_state.running)
    
    if start_btn:
        st.session_state.running = True
        st.session_state.summary = None
        st.rerun()
        
    if stop_btn:
        st.session_state.running = False
        st.rerun()
        
    # ------------------ MAIN LAYOUT ------------------
    # Left Column (70%) - Live Video Feed
    # Right Column (30%) - Real-time Metrics Dashboard
    col_main, col_metrics = st.columns([7, 3])
    
    # Video element placeholders
    with col_main:
        video_placeholder = st.empty()
        progress_placeholder = st.empty()
        
    # Metrics placeholders
    with col_metrics:
        st.markdown("### 📊 Real-Time Metrics")
        m_person = st.empty()
        m_vehicle = st.empty()
        m_fps = st.empty()
        m_unique = st.empty()
        st.markdown("---")
        breakdown_table = st.empty()
        
    # If not running, display current status/summary
    if not st.session_state.running:
        if st.session_state.summary:
            summary = st.session_state.summary
            with col_main:
                st.success("🎉 Processing completed successfully!")
                st.markdown("### 📈 Session Statistics Summary")
                
                sum_col1, sum_col2, sum_col3 = st.columns(3)
                sum_col1.metric("Frames Processed", summary["total_frames"])
                sum_col2.metric("Total Unique Tracks", summary["lifetime_objects"])
                sum_col3.metric("Average FPS", f"{summary['avg_fps']:.1f}")
                
                if "output_path" in summary and summary["output_path"]:
                    st.info(f"💾 Saved video output file to: `{summary['output_path']}`")
        else:
            with col_main:
                st.info("ℹ️ Select a source, adjust thresholds, and press **START** to begin processing.")
                # Add default interface image/graphic
                st.image("https://images.unsplash.com/photo-1507146426996-ef05306b995a?q=80&w=640&auto=format&fit=crop", 
                         caption="DeepMind Object Detection & Tracking System", use_column_width=True)
        return

    # ------------------ PROCESSING LOOP ------------------
    # Setup video reader
    cap = None
    tfile_path = None
    
    if source == "Webcam":
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            # Request high resolution from webcam hardware to maximize quality
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            st.error("❌ Webcam could not be accessed. Please ensure a camera is connected and permissions are granted.")
            st.session_state.running = False
            st.rerun()
    else:  # Uploaded file
        if uploaded_file is None:
            st.error("⚠️ Please upload a video file in the sidebar before clicking START.")
            st.session_state.running = False
            st.rerun()
            
        # Write file contents to a temp file for OpenCV reading
        import tempfile
        suffix = os.path.splitext(uploaded_file.name)[1]
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tfile.write(uploaded_file.read())
        tfile_path = tfile.name
        tfile.close()
        
        cap = cv2.VideoCapture(tfile_path)
        if not cap.isOpened():
            st.error("❌ Could not open the uploaded video file.")
            st.session_state.running = False
            st.rerun()
            
    # Setup video writer if requested
    out = None
    output_filepath = None
    if save_output:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"output_{timestamp}.mp4"
        output_filepath = os.path.join("outputs", output_filename)
        
        # Determine FPS
        fps_val = cap.get(cv2.CAP_PROP_FPS)
        if fps_val <= 0 or np.isnan(fps_val):
            fps_val = 20.0  # Fallback
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filepath, fourcc, fps_val, (640, 480))

    # Initialize model and SORT tracker
    with st.spinner(f"Initializing {model_name} model..."):
        detector = YOLODetector(model_name=model_name)
        tracker = Sort()
        
    # Metrics histories
    prev_metrics = {
        "person": 0,
        "vehicle": 0,
        "fps": 0.0,
        "unique": 0
    }
    
    # Running FPS calculations over a sliding window
    frame_times: List[float] = []
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if source == "Upload Video File" else 0
    
    try:
        while cap.isOpened() and st.session_state.running:
            start_time = time.time()
            ret, frame = cap.read()
            
            if not ret:
                break  # Video sequence ended
                
            frame_idx += 1
            
            # 2. Resize to 640x480 for processing (preserving aspect ratio via letterbox)
            frame = resize_letterbox(frame, 640, 480)
            
            # 3. Run detector
            detections = detector.detect(frame, conf_threshold=conf_thresh)
            
            # 4. Format detections for SORT: [[x1, y1, x2, y2, score, class_id], ...]
            sort_input = []
            for bbox, score, class_id in detections:
                x1, y1, x2, y2 = bbox
                sort_input.append([x1, y1, x2, y2, score, class_id])
                
            if len(sort_input) > 0:
                sort_input_arr = np.array(sort_input)
            else:
                sort_input_arr = np.empty((0, 6))
                
            # 5. Run tracker
            tracks = tracker.update(sort_input_arr)
            
            # 6. Draw boxes + labels + IDs on frame
            annotated_frame = detector.draw(
                frame=frame,
                tracks=tracks if len(tracks) > 0 else sort_input_arr,
                show_boxes=show_boxes,
                show_ids=show_ids and len(tracks) > 0
            )
            
            # 7. Calculate FPS (sliding window of 30 frames)
            process_time = time.time() - start_time
            frame_times.append(process_time)
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_fps = 1.0 / np.mean(frame_times)
            
            # 8. Update all Streamlit metrics
            person_count = 0
            vehicle_count = 0
            class_counts = {name: 0 for name in CLASS_NAMES.values()}
            
            # Count active classes using the track output if tracks are present
            active_elements = tracks if len(tracks) > 0 else sort_input_arr
            for elem in active_elements:
                class_id = int(elem[5])
                class_name = CLASS_NAMES.get(class_id, "Unknown")
                
                if class_name in class_counts:
                    class_counts[class_name] += 1
                    
                if class_id == 0:
                    person_count += 1
                elif class_id in [2, 3, 5, 7]:
                    vehicle_count += 1
                    
            unique_tracks = tracker.get_lifetime_count()
            
            # Deltas calculation (change from previous frame)
            delta_person = person_count - prev_metrics["person"]
            delta_vehicle = vehicle_count - prev_metrics["vehicle"]
            delta_fps = avg_fps - prev_metrics["fps"]
            delta_unique = unique_tracks - prev_metrics["unique"]
            
            # Update metrics placeholders with standard st.metric
            m_person.metric(label="👤 Total Persons", value=person_count, delta=delta_person)
            m_vehicle.metric(label="🚗 Total Vehicles", value=vehicle_count, delta=delta_vehicle)
            m_fps.metric(label="📊 FPS (Rolling 30f)", value=f"{avg_fps:.1f}", delta=f"{delta_fps:.1f}" if prev_metrics["fps"] > 0 else None)
            m_unique.metric(label="🎯 Lifetime Unique IDs", value=unique_tracks, delta=delta_unique)
            
            # Store values for next delta calculation
            prev_metrics["person"] = person_count
            prev_metrics["vehicle"] = vehicle_count
            prev_metrics["fps"] = avg_fps
            prev_metrics["unique"] = unique_tracks
            
            # Render breakdown table
            breakdown_df = pd.DataFrame(
                list(class_counts.items()),
                columns=["Object Class", "Active Count"]
            )
            breakdown_table.dataframe(breakdown_df, hide_index=True, use_container_width=True)
            
            # 9. Convert BGR -> RGB and show in Streamlit
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
            
            # 10. Write output frame if enabled
            if save_output and out is not None:
                out.write(annotated_frame)
                
            # Progress bar update for file uploads
            if source == "Upload Video File" and total_frames > 0:
                prog_pct = min(1.0, float(frame_idx) / total_frames)
                progress_placeholder.progress(prog_pct, text=f"Processing video frame {frame_idx}/{total_frames} ({prog_pct*100:.1f}%)")
                
    except Exception as e:
        st.error(f"⚠️ Error inside the processing loop: {str(e)}")
        
    finally:
        # Release CV resources
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
            
        # Clean up temporary file
        if tfile_path and os.path.exists(tfile_path):
            try:
                os.remove(tfile_path)
            except OSError:
                pass
                
        # Save session stats and switch off running state
        st.session_state.running = False
        avg_fps_final = 1.0 / np.mean(frame_times) if len(frame_times) > 0 else 0.0
        
        st.session_state.summary = {
            "total_frames": frame_idx,
            "lifetime_objects": tracker.get_lifetime_count() if 'tracker' in locals() else 0,
            "avg_fps": avg_fps_final,
            "output_path": output_filepath if (save_output and output_filepath) else None
        }
        
        st.rerun()

if __name__ == "__main__":
    main()
