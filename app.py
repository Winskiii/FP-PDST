import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# --- Streamlit page setup ---
st.set_page_config(
    page_title="Deteksi Sawit",
    page_icon="🌴",
    layout="wide"
)

st.title("🌴 Deteksi Sawit – YOLOv12s (Image Only)")
st.markdown("Upload gambar, dan model akan mendeteksi objek sawit.")

# --- Load model ---
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal load model: {e}")
        return None

model_path = "best_yolo_model.pt"

if not os.path.exists(model_path):
    st.error(f"❌ File model '{model_path}' tidak ditemukan!")
    st.stop()

model = load_model(model_path)
if model is None:
    st.stop()

# --- Sidebar ---
st.sidebar.header("⚙️ Pengaturan Deteksi")

confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.05)
show_boxes = st.sidebar.checkbox("Tampilkan Bounding Box", True)
show_labels = st.sidebar.checkbox("Tampilkan Label", True)
show_stats = st.sidebar.checkbox("Tampilkan Statistik", True)

# --- Upload Gambar ---
uploaded_image = st.file_uploader(
    "📤 Upload Gambar (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image_array = np.array(image)

    st.subheader("📸 Gambar Asli")
    st.image(image, use_column_width=True)

    # --- Inference ---
    with st.spinner("🔍 Mendeteksi objek..."):
        results = model(image_array, conf=confidence, iou=iou_threshold)

    result = results[0]

    # --- Annotated output ---
    if show_boxes:
        annotated_image = result.plot()
        annotated_image_pil = Image.fromarray(annotated_image)
    else:
        annotated_image_pil = image

    st.subheader("✅ Hasil Deteksi")
    st.image(annotated_image_pil, use_column_width=True)

    # --- Detection Statistics ---
    detections = result.boxes
    num_det = len(detections)

    st.metric("Total Deteksi", num_det)

    if show_stats and num_det > 0:
        import pandas as pd

        det_rows = []
        class_counts = {}

        for box in detections:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cname = model.names[cls_id]

            # count per class
            class_counts[cname] = class_counts.get(cname, 0) + 1

            det_rows.append({
                "Kelas": cname,
                "Confidence": f"{conf:.2%}",
                "X1": f"{box.xyxy[0][0]:.0f}",
                "Y1": f"{box.xyxy[0][1]:.0f}",
                "X2": f"{box.xyxy[0][2]:.0f}",
                "Y2": f"{box.xyxy[0][3]:.0f}"
            })

        df = pd.DataFrame(det_rows)
        st.subheader("📊 Detail Deteksi")
        st.dataframe(df, hide_index=True, use_container_width=True)

        st.subheader("📌 Ringkasan Kelas")
        for cls, count in class_counts.items():
            st.write(f"- **{cls}** : {count} objek")

    elif num_det == 0:
        st.info("Tidak ada objek terdeteksi.")
