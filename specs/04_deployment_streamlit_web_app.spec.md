# SPEC-04: Streamlit Web Application Deployment for Aspect-Based Sentiment Analysis

- **Spec ID:** SPEC-04
- **Title:** Phase 4 — Interactive Web Application Deployment for ABSA Analytics of Indonesian EV Comments
- **Status:** Proposed Specification / Rencana Awal
- **Author:** Antigravity AI & Wicaksono Hanif
- **Target Framework:** Streamlit (`streamlit`)
- **Inference Models:** XLM-RoBERTa-large Multi-Head Classifier (550M) & IndoRoBERTa-base Multi-Head Classifier (110M)
- **Execution Environment:** Local Execution / Cloud Hosting (Streamlit Community Cloud / HuggingFace Spaces)

---

## 1. Executive Summary & Application Goal

Spesifikasi ini mengatur perancangan dan implementasi aplikasi web interaktif berbasis **Streamlit** untuk **Aspect-Based Sentiment Analysis (ABSA)** pada komentar YouTube tentang mobil listrik (EV) China di Indonesia.

Aplikasi ini dirancang untuk memproses berkas CSV hasil ekstraksi komentar YouTube, menjalankan pipeline *preprocessing* otomatis, melakukan inferensi model PyTorch Fine-Tuned secara *real-time* atau *batch*, serta menampilkan dasbor analitik visual komprehensif yang menyoroti:
1. **Distribusi Sentimen per Aspek** (`infra`, `ekonomi`, `kualitas`, `purnajual`) melalui grafik interaktif.
2. **Top Liked Comments Gallery** (Komentar dengan jumlah *like* tertinggi berdasarkan aspek dan sentimen).
3. **Pemberian Label Otomatis & Unduh CSV Final.**

---

## 2. Definisi Workflow & Pipeline Pemprosesan

```
[ CSV Input Ekstraksi YouTube ]
               │
               ▼
[ 1. Validasi Berkas & Deteksi Kolom ] (text_original, like_count, video_id, dll)
               │
               ▼
[ 2. Preprocessing Teks Otomatis ] (Lowercasing, Slang Normalization, Emoji Cleaning)
               │
               ▼
[ 3. Inferensi PyTorch ABSA Model ] (XLM-RoBERTa 550M / IndoRoBERTa 110M + Dynamic Thresholding)
               │
               ▼
[ 4. Dasbor Analitik & Galeri Top Likes ] (Grafik Interaktif, Tab per Aspek, Export CSV)
```

---

## 3. Modul & Fitur Utama Aplikasi Web

### 3.1 Sidebar Konfigurasi & Unggah Berkas
- **Unggah Berkas CSV:** Menerima input CSV komentar ekstraksi YouTube.
- **Validasi Kolom Opsi:** Mendukung kolom wajib `text_original` (atau `comment_text`) dan kolom pendukung `like_count`, `video_id`, `published_at`.
- **Pemilihan Model Inferensi:**
  - `XLM-RoBERTa-large (550M)` — Best Accuracy & Macro F1 (Rekomendasi Utama)
  - `IndoRoBERTa-base (110M)` — Lightweight & Fast Inference
- **Pengaturan Dynamic Thresholding ($\theta_{\text{none}}$):** Slider interaktif untuk menyesuaikan ambang batas kelas `None` per aspek (Default: `infra: 0.40`, `ekonomi: 0.50`, `kualitas: 0.50`, `purnajual: 0.35`).

---

### 3.2 Dasbor Utama & Visualisasi Grafis (4 Tab Utama)

#### 📊 Tab 1: Executive Dashboard & Overview
- **Ringkasan Kartu KPI:**
  - Total Komentar Di-proses
  - Cakupan Komentar Ber-aspek (%)
  - Aspek Paling Dominan Didiskusikan
  - Sentimen Dominan Keseluruhan (Positif / Netral / Negatif)
- **Grafik Distribusi Sentimen per Aspek:**
  - *Grouped Bar Chart* interaktif yang menampilkan jumlah komentar Positif, Netral, dan Negatif di 4 aspek.
  - *Donut Chart* proporsi pembahasan aspek (`kualitas` vs `ekonomi` vs `purnajual` vs `infra`).

#### 🎯 Tab 2: Aspect Deep Dive (Analisis Mendalam 4 Aspek)
Tersedia 4 sub-tab internal (`🔋 Infra`, `💰 Ekonomi`, `🚗 Kualitas`, `🛠️ Purnajual`):
- Persentase pecahan sentimen spesifik aspek tersebut.
- Break-down distribusi kata kunci/topik utama yang paling sering muncul (WordCloud / Frequency Chart).

#### ⭐ Tab 3: Top Liked Comments Gallery (Komentar Terpopuler)
- Menampilkan komentar-komentar dengan `like_count` tertinggi yang dikelompokkan berdasarkan Aspek dan Sentimen.
- **Fitur Penyaringan:** Filter berdasarkan Aspek (`Kualitas`, `Ekonomi`, dll) dan Sentimen (`Positif`, `Negatif`).
- **Kartu Komentar (*Comment Card*):**
  - Teks Komentar Lengkap
  - Badge Jumlah Like (misal: `👍 142 Likes`)
  - Label Sentimen & Aspek Terprediksi
  - Waktu Publikasi & Link Video YouTube (jika tersedia `video_id`).

#### 📥 Tab 4: Raw Labeled Dataset & Download CSV
- Tabel dataframe interaktif dengan fitur pencarian teks (*search bar*) dan filter kolom.
- **Tombol Unduh (*Download Button*):** Mengunduh berkas CSV final yang sudah dilengkapi dengan kolom label inferensi (`infra_sentiment`, `ekonomi_sentiment`, `kualitas_sentiment`, `purnajual_sentiment`).

---

## 4. Struktur Berkas & Arsitektur Kodebase

```
usb_2026/
├── specs/
│   └── 04_deployment_streamlit_web_app.spec.md  # Spesifikasi Rencana Deployment (File Ini)
├── src/
│   └── deployment/
│       ├── __init__.py
│       ├── model_loader.py       # Caching PyTorch Model & Inference Pipeline
│       └── data_processor.py     # Preprocessing & Aspect Filtering Helper
├── models/
│   ├── indoroberta_absa/         # Checkpoint IndoRoBERTa Fine-Tuned
│   └── xlmroberta_absa/          # Checkpoint XLM-RoBERTa Fine-Tuned
└── app.py                        # Berkas Utama Antarmuka Streamlit (Entry Point)
```

---

## 5. Rencana Langkah Implementasi (Next Actions)

1. **Membuat Modul Helper Deployment:**
   - `src/deployment/model_loader.py` (Memuat bobot PyTorch model fine-tuned dengan `@st.cache_resource`).
   - `src/deployment/data_processor.py` (Integrasi `TextPreprocessor` dengan fungsi analisis top-liked comments).
2. **Membuat Script Antarmuka Utama (`app.py`):**
   - Menyusun tata letak Streamlit dengan sidebar, KPI metrics, grafik Plotly/Altair, dan galeri komentar top-likes.
3. **Pengujian Inferensi End-to-End:**
   - Menguji aplikasi mengunggah CSV ekstraksi mentah dan memverifikasi kecepatan serta keakuratan inferensi.
