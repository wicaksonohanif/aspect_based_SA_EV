# 📌 PROJECT STATEMENT & CONTEXT HANDOFF: USB 2026

### **Judul Proyek**
**Aspect-Based Sentiment Analysis (ABSA) Multi-Task Mobil Listrik (EV) China di Indonesia Menggunakan Arsitektur Discriminative Encoder Berbasis 5-Fold Stratified Cross-Validation**

- **Tim Peneliti / Author:** Wicaksono Hanif Supriyanto & Firman Pambudiansyah
- **Lokasi Workspace:** `E:\Documents\aspect_based_SA_EV`
- **Domain:** Natural Language Processing (NLP) / Deep Learning / Aspect-Based Sentiment Analysis.
- **Sumber Data:** Komentar YouTube warganet Indonesia mengenai mobil listrik pabrikan China (BYD, Wuling, Chery, Neta, Seres, dll).
- **Target Kompetisi:** Lomba Penambangan Data USB 2026 & Publikasi Jurnal Penelitian (Target: 5 Oktober 2026).

---

## 1. 🎯 Ringkasan Tujuan & Taksonomi Proyek

Membangun pipeline end-to-end klasifikasi sentimen multi-task 4 aspek otomotif EV (4 pasang target klasifikasi per komentar):

1. **4 Aspek Strategis (`infra`, `ekonomi`, `kualitas`, `purnajual`):**
   - `infra`: SPKLU, charging station, jarak tempuh, durasi pengisian daya.
   - `ekonomi`: Harga kendaraan, pajak, subsidi, depresiasi / nilai jual kembali (*resale value*).
   - `kualitas`: Quality control/build quality, suspensi, baterai LFP/Blade, fitur ADAS, interior/eksterior.
   - `purnajual`: Dealer resmi, layanan servis, ketersediaan & indent suku cadang, garansi.
2. **4 Polaritas Kelas per Aspek:**
   - `0`: `None` (Aspek tidak dibahas / Imbalance mayoritas >80%)
   - `1`: `positif` (Sentimen Positif)
   - `2`: `netral` (Sentimen Netral)
   - `3`: `negatif` (Sentimen Negatif)

---

## 2. 📂 Struktur Repositori & Modul Kode

```text
aspect_based_SA_EV/
├── brief/
│   ├── AGENTS.md                  # Directive & SOP agen AI
│   ├── PRD.md                     # Product Requirements Document v2.0 (Approved)
│   └── project_state.md           # Rangkuman status proyek & context handoff (Terkini)
├── specs/
│   ├── 01_data_extraction.spec.md # Kontrak YouTube API v3 & PII SHA-256
│   ├── 02_preprocessing_labeling.spec.md # Kontrak normalisasi & Weak Supervision
│   ├── 03_model_training.spec.md  # Kontrak 5-Fold Stratified CV Benchmark (v6)
│   └── 04_deployment_streamlit_web_app.spec.md # Spec deployment dashboard web
├── data/
│   ├── raw/                       # Komentar mentah (yt_comments_all_combined.csv, read-only)
│   ├── processed/                 # 100% Pure Human Gold Standard (954 data audit)
│   │   ├── train.csv              # Data latih baseline split awal
│   │   └── val.csv                # Data validasi baseline split awal
│   └── slang_dict.json            # Kamus normalisasi istilah gaul/otomotif
├── src/
│   ├── extraction/
│   │   └── extractor.py           # YouTubeCommentExtractor (Regex URL Parser, SHA-256)
│   ├── preprocessing/
│   │   └── text_cleaner.py        # Normalisasi slang otomotif, noise cleaner
│   ├── labeling/
│   │   ├── gemini_labeler.py      # Structured JSON Output via Gemini API
│   │   ├── weak_supervision.py    # Hybrid cascade (Rule-based + Gemini)
│   │   ├── eval_kappa.py          # Evaluator statistik Cohen's Kappa (κ)
│   │   └── dataset_splitter.py    # Stratified dataset splitter (Zero leakage)
│   └── models/
│       ├── indoroberta_classifier.py # IndoRoBERTa-base (110M) Multi-Head GELU MLP
│       ├── xlmroberta_classifier.py  # XLM-RoBERTa-large (550M) Multi-Head GELU MLP
│       ├── sahabatai_classifier.py   # Baseline model wrapper
│       └── metrics_evaluator.py      # Multi-aspect metrics calculator & confusion matrix
├── notebooks/
│   ├── 01_eda_raw_comments.ipynb
│   ├── 02_eda_cleaned_comments.ipynb
│   ├── 03_eda_labeled_dataset.ipynb
│   └── 04_kaggle_training_indoroberta_vs_xlmroberta.ipynb # Notebook eksekusi Kaggle GPU (5-Fold CV)
├── outputs/
│   └── iter-01/                   # Hasil benchmark evaluasi split awal (JSON, CSV, CM)
└── journal/                       # Berkas naskah publikasi (USB_2026.csv, audit remaining, proposal)
```

---

## 3. 🚀 Status Proyek Terkini & Guardrails Operasional

### A. Capaian Fase Selesai
1. **Fase 1: Ekstraksi Data Legal & Kepatuhan Etika**
   - Akuisisi >3.000 komentar YouTube via API v3 resmi.
   - Strict PII Anonymization via SHA-256 (`user_id_hash`).
2. **Fase 2: Preprocessing, Weak Supervision & Dataset Gold Standard**
   - Normalisasi slang otomotif lokal (`data/slang_dict.json`).
   - Hybrid Weak Supervision (Rule-based local matcher + Gemini 1.5 Flash structured JSON).
   - Validasi manusia menghasilkan **954 data Pure Human Gold Standard** ter-audit.
   - Audit sisa data menerapkan protokol **Blind Review** (Schroeder et al., 2025) guna meniadakan bias sugesti LLM.
3. **Fase 3: Transisi Benchmark 5-Fold Stratified Cross-Validation (SPEC-03 v6)**
   - Standar pengujian beralih dari *single split* (70/30) ke **5-Fold Stratified Cross-Validation** pada total 954 sampel:
     - Proporsi per fold: ~763 data latih (80%) dan ~191 data validasi (20%).
     - Komposit stratifikasi 4 aspek simultan tanpa kebocoran data (*zero data leakage*, seed=42).
   - Pelaporan metrik wajib menggunakan format ilmiah **Mean ± Standard Deviation**.

### B. Guardrails Teknis & Komputasi ($0 Infrastructure)
1. **Kaggle Disk Space Safeguard (`/kaggle/working < 20 GB`):**
   - Menerapkan **Single Best Model Checkpoint Safeguard**: hanya menyimpan 1 file bobot model terbaik (~2.2 GB) dari lipatan dengan skor evaluasi tertinggi, bukan seluruh 5 fold (menghemat >8.8 GB diska).
2. **Streamlit Deployment Memory Guardrail (RAM Limit 1 GB):**
   - Antarmuka web dasbor publik **hanya memuat IndoRoBERTa-base (110M / ~440 MB)** untuk menjamin stabilitas tanpa risiko *Out of Memory (OOM)* crash.
   - Model **XLM-RoBERTa-large (550M / ~2.1 GB)** diposisikan khusus untuk komparasi benchmark pada publikasi riset jurnal ilmiah.

---

## 4. 📊 Baseline Hasil Eksperimen Awal (Iterasi 01 - 287 Data Validasi)

*Catatan: Ini adalah acuan benchmark awal sebelum eksekusi 5-Fold Stratified CV lengkap.*

| Metrik Evaluasi | IndoRoBERTa-base (110M) | XLM-RoBERTa-large (550M) | Selisih Performa |
|---|---|---|---|
| **Overall Mean Accuracy** | 80.75% | **85.45%** | +4.70% (XLM Unggul) |
| **Mean Macro F1 (All-Class)** | 58.93% | **62.92%** | +3.99% (XLM Unggul) |
| **Mean Macro F1 (Active Sentiment)** | 48.57% | **52.81%** | +4.24% (XLM Unggul) |
| **Exact Match Ratio (Subset Acc.)** | 44.95% | **55.40%** | +10.45% (XLM Unggul) |

### Temuan Analitis per Aspek (Macro F1 Active Sentiment):
- **Infra:** IndoRoBERTa 41.01% vs **XLM-RoBERTa 49.58%**
- **Ekonomi:** IndoRoBERTa 47.28% vs **XLM-RoBERTa 61.38%**
- **Kualitas:** IndoRoBERTa 57.97% vs **XLM-RoBERTa 65.36%**
- **Purnajual:** **IndoRoBERTa 48.00%** vs XLM-RoBERTa 34.94% (XLM-RoBERTa mengalami degradasi pada kelas minoritas purnajual).

---

## 5. 🎯 Roadmap & Tindakan Selanjutnya (Sprint-Ready)

1. **Eksekusi 5-Fold Stratified CV di Kaggle GPU:**
   - Jalankan notebook `notebooks/04_kaggle_training_indoroberta_vs_xlmroberta.ipynb` menggunakan GPU NVIDIA T4.
   - Pastikan log real-time menampilkan Train Loss, Val F1-Active, Val F1-All, dan Val Accuracy per epoch.
   - Verifikasi bahwa hanya 1 model terbaik yang disimpan ke diska working.
2. **Penyusunan Hasil & Pelaporan:**
   - Ekstrak metrik rata-rata (Mean ± Std Dev) ke folder `outputs/iter-02/` (atau folder hasil 5-fold).
3. **Pembangunan Streamlit Web Dashboard (`specs/04_deployment_streamlit_web_app.spec.md`):**
   - Bangun antarmuka Streamlit berbasis model IndoRoBERTa-base dengan batas RAM < 1 GB.
4. **Publikasi Model & Jurnal:**
   - Unggah bobot IndoRoBERTa-base ke Hugging Face Hub.
   - Finalisasi draf paper penelitian di `journal/proposal_usb_2026.md` dan salindia presentasi.
