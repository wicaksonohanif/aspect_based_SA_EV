# 📌 PROJECT STATEMENT & CONTEXT HANDOFF: USB 2026

### **Judul Proyek**
**Analisis Sentimen Berbasis Aspek terhadap Opini Publik Mobil Listrik Tiongkok di Indonesia Menggunakan IndoRoBERTa dan XLM-RoBERTa**

- **Tim Peneliti / Author:** Wicaksono Hanif Supriyanto & Firman Pambudiansyah
- **Lokasi Workspace:** `C:\Users\Wicaksono Hanif\Desktop\Koding\deep_learning\usb_2026`
- **Domain:** Natural Language Processing (NLP) / Deep Learning / Aspect-Based Sentiment Analysis (ABSA).
- **Sumber Data:** Komentar YouTube warganet Indonesia mengenai mobil listrik pabrikan Tiongkok (BYD, Wuling, Chery, Neta, Seres, MG, dll).
- **Target Publikasi:** Lomba Penambangan Data USB 2026 & Jurnal Penelitian Ilmiah.

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
usb_2026/
├── brief/
│   ├── AGENTS.md                  # Directive & SOP agen AI
│   ├── PRD.md                     # Product Requirements Document v2.0 (Approved)
│   └── project_state.md           # Rangkuman status proyek & context handoff (Terkini)
├── specs/
│   ├── 01_data_extraction.spec.md # Kontrak YouTube API v3 & PII SHA-256
│   ├── 02_preprocessing_labeling.spec.md # Kontrak normalisasi & Weak Supervision
│   ├── 03_model_training.spec.md  # Kontrak 5-Fold Stratified CV Benchmark (v6)
│   ├── 04_deployment_streamlit_web_app.spec.md # Spec dashboard web Streamlit
│   └── 05_deployment_cloud_huggingface.spec.md # Spec cloud deployment & HF Hub
├── data/
│   ├── raw/                       # Komentar mentah (yt_comments_all_combined.csv, read-only)
│   ├── interim/                   # Data valid_labeled_comments.csv (1377 baris)
│   ├── dummy/                     # Data sampel inferensi mentah
│   └── processed/                 # 100% Pure Human Gold Standard (954 audit)
├── src/
│   ├── extraction/                # Extractor YouTube Data API v3 resmi
│   ├── preprocessing/             # Text cleaner & normalisasi slang otomotif
│   ├── labeling/                  # Hybrid Weak Supervision (Rule-based + Gemini 1.5)
│   └── deployment/                # Model loader (HF integration) & data processor
├── notebooks/
│   ├── 01_eda_raw_comments.ipynb
│   ├── 02_eda_cleaned_comments.ipynb
│   ├── 03_eda_labeled_dataset.ipynb
│   ├── 04_kaggle_training_indoroberta_vs_xlmroberta.ipynb # 5-Fold CV Kaggle GPU
│   └── 05_eval.ipynb              # Evaluasi komparatif & barchart per-aspek
├── outputs/
│   ├── iter-03/                   # Metrik 5-Fold CV IndoRoBERTa-base
│   └── iter-04/                   # Metrik & visualisasi XLM-RoBERTa-large
├── app.py                         # Main Entrypoint Web Dashboard SentyBoard v1.0.0
├── upload_model_to_hf.py          # Skrip pengunggah bobot model ke Hugging Face
└── requirements.txt               # Dependensi proyek
```

---

## 3. 🚀 Status Proyek Terkini & Capaian Fase

### A. Capaian Fase Selesai (Completed Milestones)
1. **Fase 1: Ekstraksi Data Legal & Kepatuhan Etika (SPEC-01)**
   - Akuisisi >3.000 komentar YouTube via API v3 resmi dengan *Strict PII Anonymization* SHA-256 (`user_id_hash`).
2. **Fase 2: Preprocessing, Weak Supervision & Dataset Gold Standard (SPEC-02)**
   - Normalisasi slang otomotif lokal (`data/slang_dict.json`).
   - Hybrid Weak Supervision (Rule-based + Gemini 1.5 Flash structured JSON).
   - Validasi manusia menghasilkan **954 data Pure Human Gold Standard** ter-audit.
3. **Fase 3: Training 5-Fold Stratified Cross-Validation (SPEC-03)**
   - Training 20 epoch per fold menggunakan Multi-Head Focal Loss ($\gamma=1.5$) & Dynamic Inverse Class Weights.
   - Evaluasi 5-Fold Stratified CV menghasilkan model pemenang **XLM-RoBERTa-large (550M)** dengan skor **Macro F1 Active 58.26% ± 3.38%** (unggul +10.05% dibanding IndoRoBERTa 48.20%).
4. **Fase 4: Dashboard Interaktif Streamlit SentyBoard v1.0.0 (SPEC-04)**
   - Dibangun di `app.py` dengan tema default **Dark Mode**, tipografi **Inter**, dan sidebar **Blue Gradient**.
   - **Mode 1 (Analytics):** Dashboard SaaS KPI 2-kolom grid layout, Donut Chart, Grouped Bar Chart, Heatmap Matrix Ko-okurensi, Boxplot Distribusi Kata, dan Galeri Komentar Terpopuler.
   - **Mode 2 (Labeling):** Inferensi otomatis komentar mentah dengan opsi unduh CSV/XLSX berlabel.
   - **Sample Data One-Click Button:** Tombol sekali klik untuk memuat dataset sampel bawaan `valid_labeled_comments.csv`.
5. **Fase 5: Benchmark Notebook Evaluasi (`notebooks/05_eval.ipynb`)**
   - Notebook evaluasi komparatif lengkap yang sudah dieksekusi (*pre-rendered cell outputs*), menyajikan barchart Macro F1 Active dan Macro F1 All per-aspek.
6. **Fase 6: Cloud Deployment & Hugging Face Hub Integration (SPEC-05)**
   - Unggah bobot model XLM-RoBERTa-large (`pytorch_model.bin` ~2.15 GB) ke Hugging Face Hub: [`wicaksonohanif/xlm-roberta-ev-absa`](https://huggingface.co/wicaksonohanif/xlm-roberta-ev-absa).
   - Implementasi **Serverless Cloud API & Zero-RAM Protection** pada `src/deployment/model_loader.py` untuk mencegah *Out of Memory (OOM)* pada Streamlit Community Cloud (batas RAM 1.0 GB).

---

## 4. 📊 Ringkasan Hasil Evaluasi Final (5-Fold Stratified CV Benchmark)

| Metrik Evaluasi | IndoRoBERTa-base (110M) | XLM-RoBERTa-large (550M) | Selisih Performa |
|---|:---:|:---:|:---:|
| **Mean Macro F1 (Active Sentiments)** | 0.4820 ± 0.0245 | **0.5826 ± 0.0338** | **+0.1005** (+20.8% Relatif) |
| **Mean Macro F1 (All Classes)** | 0.5988 ± 0.0211 | **0.6666 ± 0.0264** | **+0.0678** (+11.3% Relatif) |
| **Mean Accuracy** | 0.7937 ± 0.0100 | **0.8382 ± 0.0080** | **+0.0445** (+5.6% Relatif) |

### Breakdown Macro F1 All Classes per Aspek:
- 🏗️ **Infrastruktur EV:** IndoRoBERTa 0.6659 vs **XLM-RoBERTa 0.6470**
- 💰 **Ekonomi & Harga:** IndoRoBERTa 0.6127 vs **XLM-RoBERTa 0.7011** (+8.84%)
- 🚘 **Kualitas Produk:** IndoRoBERTa 0.5975 vs **XLM-RoBERTa 0.6775** (+8.00%)
- 🛠️ **Purna Jual:** IndoRoBERTa 0.5191 vs **XLM-RoBERTa 0.6407** (+12.16%)

---

## 5. 🌐 Artefak Terpublikasi & Tautan Penting

- **Hugging Face Model Hub:** [`wicaksonohanif/xlm-roberta-ev-absa`](https://huggingface.co/wicaksonohanif/xlm-roberta-ev-absa)
- **GitHub Repository:** [`wicaksonohanif/aspect_based_SA_EV`](https://github.com/wicaksonohanif/aspect_based_SA_EV)
- **Notebook Evaluasi:** [`notebooks/05_eval.ipynb`](file:///C:/Users/Wicaksono%20Hanif/Desktop/Koding/deep_learning/usb_2026/notebooks/05_eval.ipynb)
- **Spesifikasi Deployment Cloud:** [`specs/05_deployment_cloud_huggingface.spec.md`](file:///C:/Users/Wicaksono%20Hanif/Desktop/Koding/deep_learning/usb_2026/specs/05_deployment_cloud_huggingface.spec.md)
