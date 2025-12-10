# 🌴 Aplikasi Deteksi Sawit dengan YOLO

Aplikasi web berbasis **Streamlit** untuk deteksi kelapa sawit menggunakan model **YOLOv12s** yang telah dilatih.

## 📋 Persyaratan

- Python 3.8 atau lebih tinggi
- pip (Python package manager)
- File model: `best_yolo_model.pt`

## 🚀 Instalasi

### 1. Clone atau Download Project

```bash
cd "d:\Final Project PDST"
```

### 2. Buat Virtual Environment (Opsional tapi disarankan)

```bash
# Untuk Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Untuk Command Prompt
python -m venv venv
venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: Instalasi torch dan ultralytics mungkin memakan waktu beberapa menit

### 4. Pastikan File Model Ada

Pastikan file `best_yolo_model.pt` berada di direktori yang sama dengan `app.py`:

```
d:\Final Project PDST\
├── app.py
├── best_yolo_model.pt
├── requirements.txt
└── README.md
```

## 🏃 Menjalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser Anda di `http://localhost:8501`

## 📖 Fitur Aplikasi

### 1. 📷 Deteksi Gambar
- Upload gambar (JPG, JPEG, PNG, BMP)
- Deteksi objek sawit secara real-time
- Tampilkan bounding boxes dan confidence scores
- Download hasil deteksi

### 2. 🎥 Deteksi Video
- Upload video (MP4, AVI, MOV, MKV)
- Proses frame-by-frame dengan deteksi
- Buat video output dengan annotasi
- Download video hasil

### 3. 📊 Info Model
- Informasi detail tentang model
- Daftar kelas yang dapat dideteksi
- Spesifikasi file model

### 4. 📖 Panduan Penggunaan
- Tutorial lengkap penggunaan aplikasi
- Tips dan trik untuk hasil terbaik
- Penjelasan parameter deteksi

## ⚙️ Konfigurasi

Gunakan sidebar untuk menyesuaikan parameter:

- **Confidence Threshold** (0.0 - 1.0): Sensitivitas deteksi
  - Nilai lebih tinggi = deteksi lebih ketat
  - Nilai lebih rendah = deteksi lebih sensitif
  
- **IOU Threshold** (0.0 - 1.0): Threshold untuk NMS (Non-Maximum Suppression)

## 💡 Tips Penggunaan

1. **Kualitas Gambar Terbaik**:
   - Gunakan gambar resolusi tinggi
   - Pencahayaan yang baik
   - Sudut pandang yang jelas

2. **Confidence Threshold**:
   - 0.25 (default): Keseimbangan baik antara recall dan precision
   - 0.5: Lebih konservatif, sedikit false positives
   - <0.25: Lebih agresif, deteksi lebih banyak

3. **Proses Video**:
   - Video lebih pendek = proses lebih cepat
   - Tentukan max frames untuk kontrol waktu proses
   - Hasil video akan di-download secara otomatis

## 📦 File Structure

```
d:\Final Project PDST\
├── app.py                          # Main Streamlit application
├── best_yolo_model.pt             # Trained YOLO model
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── Copy_of_data_sains_sawit_dataset_asli_update1 (1).ipynb  # Training notebook
```

## 🔧 Troubleshooting

### Error: "Model file not found"
- Pastikan `best_yolo_model.pt` ada di direktori yang sama dengan `app.py`

### Error: "Module not found"
- Jalankan: `pip install -r requirements.txt`
- Pastikan virtual environment sudah aktif

### Aplikasi Lambat saat Memproses Video
- Kurangi nilai "Frame maksimal untuk diproses"
- Gunakan video dengan resolusi lebih rendah

### GPU Not Found
- Aplikasi akan otomatis menggunakan CPU jika GPU tidak tersedia
- Untuk performa lebih baik, gunakan GPU (CUDA)

## 📝 Model Details

- **Arsitektur**: YOLOv12s
- **Framework**: PyTorch + Ultralytics
- **Task**: Object Detection
- **Input Size**: 640x640 pixels (configurable)
- **Epoch**: 50
- **Batch Size**: 16-32

## 📊 Metrik Model

Model telah dilatih dan dievaluasi dengan metrik:
- **mAP50**: Mean Average Precision at IoU=0.5
- **Precision**: Akurasi deteksi positif
- **Recall**: Jumlah objek yang berhasil dideteksi
- **F1-Score**: Harmonic mean dari precision dan recall

## 🤝 Support & Kontribusi

Untuk pertanyaan atau issue, silakan buka issue di repository.

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik/Final Project PDST.

---

**Dibuat dengan ❤️ menggunakan Streamlit + YOLOv12s**
