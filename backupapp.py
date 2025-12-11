import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
from pathlib import Path
import tempfile

# Set page configuration
st.set_page_config(
    page_title="Deteksi Sawit - YOLO",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton > button {
        width: 100%;
        padding: 10px;
        font-size: 16px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🌴 Aplikasi Deteksi Sawit")
st.markdown("Menggunakan YOLOv12s untuk deteksi kelapa sawit secara real-time")

# Sidebar
st.sidebar.header("⚙️ Konfigurasi")

# Model selection and loading
@st.cache_resource
def load_model(model_path):
    """Load YOLO model"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model_path = "best_yolo_model.pt"
if not os.path.exists(model_path):
    st.error(f"❌ Model file '{model_path}' tidak ditemukan!")
    st.info(f"Pastikan file '{model_path}' ada di direktori yang sama dengan script ini.")
    st.stop()

model = load_model(model_path)
if model is None:
    st.stop()

# Sidebar configuration
confidence = st.sidebar.slider("🎯 Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("📐 IOU Threshold", 0.0, 1.0, 0.45, 0.05)

# Tab selection
tab1, tab2, tab3, tab4 = st.tabs(["📷 Deteksi Gambar", "🎥 Deteksi Video", "📊 Info Model", "📈 Panduan"])

# ============================================
# TAB 1: Image Detection
# ============================================
with tab1:
    st.header("Deteksi pada Gambar")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Gambar")
        uploaded_image = st.file_uploader(
            "Pilih gambar untuk dianalisis",
            type=["jpg", "jpeg", "png", "bmp"],
            key="image_upload"
        )
    
    with col2:
        st.subheader("Opsi Deteksi")
        show_boxes = st.checkbox("Tampilkan Bounding Boxes", value=True)
        show_labels = st.checkbox("Tampilkan Label & Confidence", value=True)
        show_stats = st.checkbox("Tampilkan Statistik", value=True)
    
    if uploaded_image is not None:
        # Display uploaded image
        image = Image.open(uploaded_image)
        image_array = np.array(image)
        
        # Run inference
        with st.spinner("🔍 Melakukan deteksi..."):
            results = model(image_array, conf=confidence, iou=iou_threshold)
        
        # Get results
        result = results[0]
        
        # Prepare visualization
        if show_boxes:
            annotated_image = result.plot()
            annotated_image_pil = Image.fromarray(annotated_image)
        else:
            annotated_image_pil = image
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Hasil Deteksi")
            st.image(annotated_image_pil, use_column_width=True)
        
        with col2:
            st.subheader("📊 Detail Deteksi")
            
            # Get detection info
            detections = result.boxes
            num_detections = len(detections)
            
            st.metric("Total Deteksi", num_detections)
            
            if show_stats and num_detections > 0:
                st.markdown("---")
                st.write("**Detail Objek yang Terdeteksi:**")
                
                # Display detection details
                detection_data = []
                class_counts = {}
                
                for box in detections:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls_id]
                    
                    # Count classes
                    if class_name not in class_counts:
                        class_counts[class_name] = 0
                    class_counts[class_name] += 1
                    
                    detection_data.append({
                        "Kelas": class_name,
                        "Confidence": f"{conf:.2%}",
                        "Koordinat": f"({box.xyxy[0][0]:.0f}, {box.xyxy[0][1]:.0f})"
                    })
                
                # Display as table
                import pandas as pd
                df = pd.DataFrame(detection_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                st.write("**Ringkasan Kelas:**")
                for class_name, count in class_counts.items():
                    st.write(f"- **{class_name}**: {count} objek")
            
            elif not show_stats and num_detections > 0:
                st.write(f"✅ Terdeteksi {num_detections} objek")
                unique_classes = set([model.names[int(box.cls[0])] for box in detections])
                for cls in unique_classes:
                    count = sum(1 for box in detections if model.names[int(box.cls[0])] == cls)
                    st.write(f"- {cls}: {count}")
            else:
                st.info("ℹ️ Tidak ada objek yang terdeteksi")

# ============================================
# TAB 2: Video Detection
# ============================================
with tab2:
    st.header("Deteksi pada Video")
    
    st.info("⚠️ Proses video bisa memakan waktu tergantung durasi video")
    
    uploaded_video = st.file_uploader(
        "Pilih file video untuk dianalisis",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )
    
    if uploaded_video is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚙️ Pengaturan Video")
            max_frames = st.slider("Frame maksimal untuk diproses", 1, 500, 100)
            process_video = st.button("🎬 Proses Video", use_container_width=True)
        
        with col2:
            st.subheader("📊 Info Video")
            
        if process_video:
            # Save uploaded video to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_video.read())
                tmp_path = tmp_file.name
            
            try:
                # Open video
                cap = cv2.VideoCapture(tmp_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                st.write(f"- **FPS**: {fps}")
                st.write(f"- **Total Frame**: {total_frames}")
                st.write(f"- **Resolusi**: {width}x{height}")
                
                # Process video
                output_path = "output_video.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                frame_count = 0
                detection_summary = {}
                
                while cap.isOpened() and frame_count < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Run inference
                    results = model(frame, conf=confidence, iou=iou_threshold)
                    result = results[0]
                    
                    # Annotate frame
                    annotated_frame = result.plot()
                    out.write(annotated_frame)
                    
                    # Count detections
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        class_name = model.names[cls_id]
                        if class_name not in detection_summary:
                            detection_summary[class_name] = 0
                        detection_summary[class_name] += 1
                    
                    frame_count += 1
                    progress = frame_count / min(max_frames, total_frames)
                    progress_bar.progress(progress)
                    status_text.text(f"Memproses frame {frame_count}/{min(max_frames, total_frames)}")
                
                cap.release()
                out.release()
                
                st.success("✅ Video berhasil diproses!")
                
                # Display output video
                st.subheader("🎬 Hasil Video")
                st.video(output_path)
                
                # Display summary
                st.subheader("📊 Ringkasan Deteksi")
                total_detections = sum(detection_summary.values())
                st.metric("Total Deteksi", total_detections)
                
                if detection_summary:
                    import pandas as pd
                    summary_df = pd.DataFrame(list(detection_summary.items()), columns=["Kelas", "Jumlah"])
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                # Download button
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Video Hasil",
                        data=f.read(),
                        file_name="deteksi_sawit.mp4",
                        mime="video/mp4"
                    )
                
            except Exception as e:
                st.error(f"Error memproses video: {e}")
            finally:
                # Cleanup
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

# ============================================
# TAB 3: Model Info
# ============================================
with tab3:
    st.header("📋 Informasi Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Spesifikasi Model")
        st.write(f"**Nama Model**: YOLOv12s")
        st.write(f"**Task**: Object Detection")
        st.write(f"**Framework**: PyTorch")
        
        # Get class information
        st.subheader("🏷️ Kelas yang Dapat Dideteksi")
        if hasattr(model, 'names'):
            for idx, class_name in enumerate(model.names.items()):
                st.write(f"{idx + 1}. {class_name[1]}")
    
    with col2:
        st.subheader("📁 File Model")
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
            st.write(f"**Path**: {model_path}")
            st.write(f"**Ukuran**: {file_size:.2f} MB")
            st.write(f"**Status**: ✅ Loaded")
        
        st.subheader("⚙️ Konfigurasi Default")
        st.write(f"**Confidence Threshold**: {confidence}")
        st.write(f"**IOU Threshold**: {iou_threshold}")

# ============================================
# TAB 4: Panduan Penggunaan
# ============================================
with tab4:
    st.header("📖 Panduan Penggunaan")
    
    st.subheader("🚀 Memulai")
    st.markdown("""
    1. **Deteksi Gambar**: Upload gambar untuk mendeteksi objek sawit
       - Pilih gambar dalam format JPG, JPEG, PNG, atau BMP
       - Sesuaikan Confidence Threshold di sidebar untuk mengubah sensitifitas deteksi
       - Hasil akan ditampilkan dengan bounding boxes dan confidence score
    
    2. **Deteksi Video**: Upload video untuk analisis frame-by-frame
       - Pilih file video dalam format MP4, AVI, MOV, atau MKV
       - Tentukan jumlah frame maksimal untuk diproses
       - Proses akan berjalan dan menghasilkan video dengan deteksi
    
    3. **Konfigurasi**:
       - **Confidence Threshold**: Nilai minimum kepercayaan diri model (0-1)
         - Nilai lebih tinggi = deteksi lebih ketat, false positives lebih sedikit
         - Nilai lebih rendah = deteksi lebih sensitif, mungkin ada false positives
       - **IOU Threshold**: Threshold untuk Non-Maximum Suppression (0-1)
    """)
    
    st.subheader("💡 Tips & Trik")
    st.markdown("""
    - **Kualitas Gambar**: Gunakan gambar dengan kualitas tinggi untuk hasil terbaik
    - **Pencahayaan**: Pastikan pencahayaan yang baik pada objek yang ingin dideteksi
    - **Sudut Pandang**: Berbagai sudut pandang akan meningkatkan deteksi
    - **Confidence Threshold**: 
      - 0.25 (Default): Keseimbangan antara recall dan precision
      - 0.5: Lebih konservatif, mengurangi false positives
      - <0.25: Lebih agresif, mendeteksi lebih banyak objek
    """)
    
    st.subheader("📊 Interpretasi Hasil")
    st.markdown("""
    - **Bounding Box**: Kotak yang mengelilingi objek yang terdeteksi
    - **Confidence Score**: Persentase kepercayaan model terhadap deteksi (0-100%)
    - **Class Label**: Jenis objek yang terdeteksi (e.g., 'Kelapa Sawit')
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
    <p>🌴 Aplikasi Deteksi Sawit menggunakan YOLOv12s</p>
    <p style='font-size: 12px; color: gray;'>© 2025 - Final Project PDST</p>
    </div>
    """,
    unsafe_allow_html=True
)
