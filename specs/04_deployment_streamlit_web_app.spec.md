# SPEC-04: Streamlit Web Application Deployment for Aspect-Based Sentiment Analysis

- **Spec ID:** SPEC-04
- **Title:** Phase 4 — Dual-Mode Interactive Web Application Deployment (SaaS Executive Dashboard & XLM-RoBERTa Inference Engine)
- **Status:** Approved Specification / Rencana Implementasi Terbarui
- **Author:** Antigravity AI & Wicaksono Hanif
- **Target Framework:** Streamlit (`streamlit`)
- **Primary Inference Model:** XLM-RoBERTa Multi-Head Classifier (`models/xlmroberta_local_absa/` or `models/xlmroberta_absa/`)
- **Execution Environment:** Local Execution / Cloud Hosting (Streamlit Community Cloud / HuggingFace Spaces)

---

## 1. Executive Summary & Architecture Overview

Spesifikasi ini mengatur perancangan dan implementasi aplikasi web interaktif berbasis **Streamlit** untuk **Aspect-Based Sentiment Analysis (ABSA)** pada komentar YouTube tentang mobil listrik (EV) China di Indonesia.

Aplikasi dirancang dengan tata letak modular modern berpenampilan **SaaS Executive Dashboard** yang dilengkapi **Custom Header Banner** serta **2 Pilihan Mode Utama**:

1. **Header Banner Kustom (Top Banner):** Menampilkan banner grafis aplikasi di bagian paling atas halaman utama (mendukung file `.jpg` kustom seperti `assets/banner.jpg`).
2. **Mode 1 — Analisis Data Berlabel (SaaS Executive Dashboard):** Menerima input berkas master berlabel (`master_labeled_comments.csv` atau `valid_labeled_comments.csv`), lalu menyajikan dasbor eksekutif tingkat tinggi (metrik SaaS KPI, jumlah video YouTube yang dianalisis, rasio aspek, sentimen per aspek, dan galeri komentar terpopuler).
3. **Mode 2 — Pelabelan Data Otomatis (XLM-RoBERTa Inference Engine):** Menerima input berkas komentar mentah hasil tarikan YouTube API (`yt_comments_raw.csv`), menjalankan pipeline pembersihan teks & inferensi model **XLM-RoBERTa Fine-Tuned**, lalu menyediakan tombol unduh CSV berlabel yang siap dimasukkan ke Mode 1.

---

## 2. Definisi Workflow & Antarmuka Utama

```
[ Top Banner Image (assets/banner.jpg) ]
                │
                ▼
[ Radio Navigation / Pilihan Mode Utama ]
                ├──► 📊 Pilihan 1: Analisis Data Berlabel (Executive Dashboard)
                └──► 🤖 Pilihan 2: Lakukan Pelabelan Data (XLM-RoBERTa Inference)
```

---

## 3. Rincian Fitur Komponen & Antarmuka

### 3.1 Custom Top Header Banner
- Terletak di posisi paling atas halaman web Streamlit.
- Menampilkan gambar banner utama aplikasi (default: `assets/banner.jpg` / fallback ke header title bergaya SaaS jika file gambar belum tersedia).
- Pengguna dapat dengan mudah mengganti file gambar banner `.jpg` milik sendiri di folder `assets/`.

---

### 3.2 Mode 1: Analisis Data Berlabel (SaaS Executive Dashboard)

#### A. Input & Sumber Data
- **Pengunggah Berkas (File Uploader):** Pengguna mengunggah CSV master berlabel (`master_labeled_comments.csv` atau CSV hasil unduhan Mode 2).
- **Default Fallback Dataset:** Jika pengguna belum mengunggah file, aplikasi secara otomatis memuat dataset lokal `data/interim/master_labeled_comments.csv` (atau `valid_labeled_comments.csv`).

#### B. Metrik Ringkasan Eksekutif (SaaS Executive KPI Cards)
Kartu metrik KPI interaktif di bagian atas dasbor:
1. **Total Komentar (Total Comments):** Jumlah total baris komentar yang dianalisis.
2. **Total Video YouTube (Videos Analyzed):** Jumlah video unik (`video_id`) asal komentar.
3. **Komentar Ber-Aspek (% Coverage):** Persentase komentar yang membahas setidaknya 1 aspek EV.
4. **Dominasi Aspek Top:** Aspek yang paling banyak diperbincangkan konsumen.
5. **Sentimen Dominan Keseluruhan:** Proporsi Positif / Netral / Negatif dari seluruh aspek gabungan.

#### C. Tab Analitik Dashboard
- **Tab 1 — Executive Summary & Overview:** Donut Chart & Grouped Bar Chart untuk perbandingan sentimen per 4 aspek (`Infrastruktur`, `Ekonomi`, `Kualitas`, `Purnajual`).
- **Tab 2 — Aspect Deep Dive:** Analisis mendalam per aspek beserta breakdown sentimen dan kata kunci dominan (WordCloud).
- **Tab 3 — Top Liked Comments Gallery:** Galeri kartu komentar terpopuler (berdasarkan `like_count`) dilengkapi filter Aspek dan Sentimen.
- **Tab 4 — Data Viewer:** Tabel data mentah dengan fitur pencarian dan penyaringan interaktif.

---

### 3.3 Mode 2: Pelabelan Data Otomatis (XLM-RoBERTa Inference Engine)

#### A. Input Data Mentah
- **Pengunggah Berkas (File Uploader):** Menerima file CSV hasil ekstraksi mentah dari YouTube API (memuat kolom `text_original`, `like_count`, `video_id`, dll).

#### B. Pipeline Inferensi Model XLM-RoBERTa
1. **Pembersihan Teks Otomatis (*Text Cleaner*):** Normalisasi kata gaul/slang, lowercasing, pembersihan URL, mention, dan huruf berulang.
2. **Inferensi PyTorch XLM-RoBERTa:**
   - Memuat checkpoint model `models/xlmroberta_local_absa/` (atau `models/xlmroberta_absa/`).
   - Menjalankan inferensi multi-head untuk 4 aspek dengan ambang batas keputusan (*Dynamic Thresholding*):
     - `infra`: 0.40, `ekonomi`: 0.50, `kualitas`: 0.50, `purnajual`: 0.35.
3. **Bilah Kemajuan Inferensi (*Progress Bar & Spinner*):** Menampilkan estimasi waktu dan progres pelabelan baris demi baris/batch.

#### C. Unduh Berkas CSV Berlabel (*Download Labeled CSV*)
- Menampilkan cuplikan tabel hasil pelabelan (`infra_sentiment`, `ekonomi_sentiment`, `kualitas_sentiment`, `purnajual_sentiment`).
- **Tombol Unduh (*Download Button*):** Memungkinkan pengguna mengunduh file CSV hasil pelabelan otomatis. File hasil unduhan ini didesain agar kompatibel 100% untuk dimasukkan langsung ke Mode 1 (Executive Dashboard).

---

## 4. Struktur Berkas & Arsitektur Kodebase

```
usb_2026/
├── assets/
│   └── banner.jpg                    # Custom Header Banner (bisa diganti user)
├── specs/
│   └── 04_deployment_streamlit_web_app.spec.md  # Spesifikasi Rencana Deployment
├── src/
│   └── deployment/
│       ├── __init__.py
│       ├── model_loader.py           # PyTorch Model Caching (@st.cache_resource)
│       └── data_processor.py         # Preprocessing, Filtering & KPI Calculator
├── models/
│   └── xlmroberta_local_absa/        # Checkpoint Model XLM-RoBERTa Fine-Tuned
└── app.py                            # Entry Point Antarmuka Utama Streamlit
```

---

## 5. Rencana Langkah Implementasi

1. **Persiapan Berkas Aset & Helper Deployment:**
   - Menyiapkan folder `assets/` untuk gambar banner `.jpg`.
   - Membuat `src/deployment/model_loader.py` dan `src/deployment/data_processor.py`.
2. **Pembuatan Antarmuka Utama (`app.py`):**
   - Menulis komponen Streamlit dengan tampilan banner kustom, radio button pilihan mode (Mode 1 vs Mode 2), KPI cards, dan grafik interaktif (Plotly/Seaborn).
3. **Pengujian Inferensi & Integrasi End-to-End:**
   - Menguji alur Mode 2 (upload CSV mentah -> inferensi XLM-RoBERTa -> download CSV) lalu mengunggah CSV hasil unduhan ke Mode 1 untuk memverifikasi kecocokan dasbor eksekutif.
