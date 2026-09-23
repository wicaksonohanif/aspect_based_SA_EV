# 📌 PROJECT STATEMENT & CONTEXT HANDOFF: USB 2026

### **Judul Proyek**
**Aspect-Based Sentiment Analysis (ABSA) Mobil Listrik (EV) China di Indonesia Menggunakan Arsitektur Discriminative Encoder (IndoRoBERTa vs. XLM-RoBERTa)**

- **Lokasi Workspace:** `C:\Users\Wicaksono Hanif\Desktop\Koding\deep_learning\usb_2026`
- **Domain:** Natural Language Processing (NLP) / Deep Learning / Aspect-Based Sentiment Analysis.
- **Sumber Data:** Komentar YouTube warganet Indonesia mengenai mobil listrik pabrikan China (BYD, Wuling, Chery, Neta, Seres, dll).

---

## 1. 🎯 Ringkasan Tujuan & Taksonomi Proyek

Proyek ini bertujuan membangun pipeline end-to-end klasifikasi sentimen berbasis 4 aspek otomotif EV secara multi-task (4 pasang target klasifikasi per komentar):

1. **4 Aspek Strategis (`infra`, `ekonomi`, `kualitas`, `purnajual`):**
   - `infra`: SPKLU, charging station, jarak tempuh, durasi pengisian daya.
   - `ekonomi`: Harga kendaraan, pajak, subsidi, depresiasi / nilai jual kembali (*resale value*).
   - `kualitas`: Quality control/build quality, suspensi, baterai LFP/Blade, fitur ADAS, interior/eksterior.
   - `purnajual`: Dealer resmi, layanan servis, ketersediaan & indent suku cadang, garansi.
2. **4 Polaritas Kelas per Aspek:**
   - `0`: `None` (Aspek tidak dibahas)
   - `1`: `positif` (Sentimen Positif)
   - `2`: `netral` (Sentimen Netral)
   - `3`: `negatif` (Sentimen Negatif)

---

## 2. 📂 Struktur Repositori & Modul Kode

```
usb_2026/
├── data/
│   ├── raw/                       # Komentar mentah YouTube (yt_comments_all_combined.csv)
│   ├── processed/                 # Dataset 100% Pure Human Gold Standard
│   │   ├── train.csv              # 667 data latih (70%)
│   │   └── val.csv                # 287 data validasi/uji (30%)
├── specs/                         # Spesifikasi Teknis Formal
│   ├── 01_data_extraction.spec.md # Spec ekstraksi YouTube API v3
│   ├── 02_preprocessing_labeling.spec.md # Spec cleaning & weak supervision
│   └── 03_model_training.spec.md  # Spec arsitektur & benchmark model
├── src/
│   ├── extraction/
│   │   └── extractor.py           # Class YouTubeCommentExtractor (Regex URL Parser, PII Hashing SHA-256)
│   ├── preprocessing/
│   │   └── text_cleaner.py        # Normalisasi slang otomotif, repeat reduction, noise clean
│   ├── labeling/
│   │   ├── gemini_labeler.py      # Structured JSON Output via Google Gemini API
│   │   ├── weak_supervision.py    # Hybrid cascade engine (Rule-based + Gemini API)
│   │   ├── eval_kappa.py          # Evaluator statistik Cohen's Kappa (κ)
│   │   └── dataset_splitter.py    # Stratified dataset splitter (Zero leakage)
│   └── models/
│       ├── indoroberta_classifier.py # IndoRoBERTa-base (110M) Multi-Head GELU MLP
│       ├── xlmroberta_classifier.py  # XLM-RoBERTa-large (550M) Multi-Head GELU MLP
│       ├── sahabatai_classifier.py   # Baseline model / comparative wrapper
│       └── metrics_evaluator.py      # Multi-aspect metrics calculator & confusion matrix
├── notebooks/
│   ├── 01_eda_raw_comments.ipynb
│   ├── 02_eda_cleaned_comments.ipynb
│   ├── 03_eda_labeled_dataset.ipynb
│   └── 04_kaggle_training_indoroberta_vs_xlmroberta.ipynb # Notebook eksekusi Kaggle GPU
├── outputs/
│   └── iter-01/                   # Hasil evaluasi benchmark iterasi 01 (JSON, CSV, CM)
└── journal/                       # Berkas penyusunan paper/jurnal (USB_2026.csv, audit xlsx)
```

---

## 3. 🚀 Pekerjaan yang Telah Selesai (Fase 1 hingga Fase 3)

### **Fase 1: Data Extraction & PII Guardrails**
- Membangun modul `src/extraction/extractor.py` berbasis YouTube Data API v3 resmi.
- Fitur auto URL parser (mendukung format standard, short `youtu.be`, shorts, embed, hingga raw video ID).
- Penerapan **Strict PII Masking** (`user_id_hash` via SHA-256) untuk perlindungan privasi warganet.
- Berhasil mengumpulkan >3.000 komentar mentah dari video ulasan EV China.

### **Fase 2: Preprocessing, Weak Supervision & Dataset Gold Standard**
- Membangun `src/preprocessing/text_cleaner.py` beserta kamus slang otomotif (`data/slang_dict.json`) untuk menormalisasi istilah informal (seperti *SPKLU, mobkas, ngecas, batre, resale*).
- Membangun Hybrid Cascade Engine (`src/labeling/weak_supervision.py`):
  - *Step 1:* Rule-based local matcher (0 API calls, instan).
  - *Step 2:* Google Gemini API (`gemini-1.5-flash`) dengan Structured JSON Output untuk teks ambigu/sarkasme.
- Mengirim sampel audit terstratifikasi (954 komentar) ke manusia, menghasilkan **100% Pure Human Gold Standard Dataset**.
- Menjalankan pembagian dataset tanpa kebocoran data (*zero leakage*): **Train set (667 baris / 70%)** dan **Val set (30% / 287 baris)**.

### **Fase 3: Model Benchmark (IndoRoBERTa-base vs XLM-RoBERTa-large)**
- **Spesifikasi Model & Head:**
  - **IndoRoBERTa-base** (`indolem/indobert-base-uncased`, 110M params): Monolingual Encoder.
  - **XLM-RoBERTa-large** (`xlm-roberta-large`, 550M params): Multilingual Encoder.
  - **Head:** Concatenation `[CLS] + Mean Pooling` $\rightarrow$ 4 x GELU MLP Multi-Task Heads (`Linear -> GELU -> Dropout -> Linear`).
- **Strategi Imbalance & Loss:**
  - Multi-Aspect Focal Loss ($\gamma = 1.5$) + Smoothed Class Weights ($\sqrt{w}$).
  - Aspect-Specific Decision Thresholding ($\theta_{\text{purnajual}}=0.35$, $\theta_{\text{infra}}=0.40$, $\theta_{\text{ekonomi}}=0.50$, $\theta_{\text{kualitas}}=0.50$).
  - Targeted Class Alpha Scaling ($\alpha_{\text{purnajual, positif}} = 2.5\times$, $\alpha_{\text{infra, positif}} = 1.8\times$).
- **Lingkungan Eksekusi:** Kaggle Notebooks (NVIDIA T4 GPU 16GB VRAM, Full Fine-Tuning 20 Epochs, tanpa kuantisasi 4-bit).

---

## 4. 📊 Hasil Benchmark Terkini (Iterasi 01 - Validation Set: 287 Data Uji)

Hasil evaluasi pada `outputs/iter-01/model_comparison_metrics.json`:

| Metrik Evaluasi | IndoRoBERTa-base (110M) | XLM-RoBERTa-large (550M) | Selisih / Keunggulan |
|---|---|---|---|
| **Overall Mean Accuracy** | 80.75% | **85.45%** | **+4.70%** (XLM-RoBERTa unggul) |
| **Mean Macro F1 (All-Class)** | 58.93% | **62.92%** | **+3.99%** (XLM-RoBERTa unggul) |
| **Mean Macro F1 (Active Sentiment)** | 48.57% | **52.81%** | **+4.24%** (XLM-RoBERTa unggul) |
| **Exact Match Ratio (Subset Acc.)** | 44.95% | **55.40%** | **+10.45%** (XLM-RoBERTa unggul) |

### Detail Per-Aspek (Macro F1 Active Sentiment / Accuracy):
- **Infra:** IndoRoBERTa (41.01% F1 / 89.90% Acc) vs **XLM-RoBERTa (49.58% F1 / 91.64% Acc)**
- **Ekonomi:** IndoRoBERTa (47.28% F1 / 74.56% Acc) vs **XLM-RoBERTa (61.38% F1 / 81.53% Acc)**
- **Kualitas:** IndoRoBERTa (57.97% F1 / 70.73% Acc) vs **XLM-RoBERTa (65.36% F1 / 78.40% Acc)**
- **Purnajual:** **IndoRoBERTa (48.00% F1 / 87.80% Acc)** vs XLM-RoBERTa (34.94% F1 / 90.24% Acc)

---

## 5. 🎯 Rencana Langkah Selanjutnya (Roadmap Agent Baru)

Saat memulai pada chat baru, instruksikan agen AI baru untuk melanjutkan tugas berikut:

1. **Iterasi 02 Optimization Strategy (Augmentasi Data Teks Minoritas Kontekstual):**
   - Menerapkan augmentasi teks kontekstual (Synonym replacement + Context-preserving paraphrasing) khusus pada **Train Set (`train.csv`)** untuk kelas minoritas ekstrem (`purnajual positif` dan `infra positif`) sesuai spesifikasi `specs/03_model_training.spec.md` Bagian 3.4.
   - Menjaga Validation Set (`val.csv`) **100% murni data asli manusia** untuk pengujian fair.
2. **Re-run & Evaluation di Kaggle:**
   - Menjalankan kembali eksperimen Iterasi 02 pada Kaggle Notebook (`notebooks/04_kaggle_training_indoroberta_vs_xlmroberta.ipynb`).
   - Menganalisis apakah augmentasi meningkatkan Active Macro F1 pada aspek `purnajual` dan `infra`.
3. **Penyusunan Paper/Draf Jurnal:**
   - Menyusun analisis hasil eksperimen dan confusion matrix ke dalam draf publikasi penelitian pada folder `journal/`.
