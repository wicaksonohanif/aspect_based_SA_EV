# AGENTS.md — Research & Development Directive

## 1. Project Overview & Scope

- **Project Name:** Chinese EV Perception Data Mining & Aspect-Based Sentiment Analysis (ABSA)
- **Domain:** Natural Language Processing (Multi-Aspect / Multi-Task Sentiment Analysis)
- **Primary Objective:** Ekstraksi sentimen dan 4 aspek persepsi publik (`infra`, `ekonomi`, `kualitas`, `purnajual`) terhadap mobil listrik pabrikan China dari komentar YouTube di Indonesia.
- **Core Models:** Benchmark arsitektur *Discriminative Encoder* (IndoRoBERTa-base 110M vs XLM-RoBERTa-large 550M) dengan 4 GELU MLP Multi-Task Classifier Heads.

---

## 2. Core Operational Philosophy: Spec-Driven Development (SDD)

Sebagai agen AI, Anda **WAJIB** tunduk pada prinsip Spec-Driven Development:

1. **Spec First, Code Later:** Jangan membuat skrip implementasi, modul, atau *pipeline* sebelum dokumen spesifikasi teknis (`./specs/<feature>.spec.md`) diverifikasi dan disetujui.
2. **Traceability:** Setiap berkas eksperimen dan kode harus merujuk ke file pada berkas spesifikasi terkait (`specs/01_data_extraction.spec.md`, `specs/02_preprocessing_labeling.spec.md`, `specs/03_model_training.spec.md`).
3. **No Hallucinated Packages:** Hanya gunakan pustaka yang tercantum di `requirements.txt` / lingkungan Python proyek. Jika membutuhkan dependensi baru, usulkan pembaruan spesifikasi terlebih dahulu.

---

## 3. Directory Structure & Key Artifacts

```text
usb_2026/
├── brief/
│   ├── AGENTS.md              # Directive & kontrak agentic AI ini
│   ├── PRD.md                 # Product Requirements Document
│   └── project_state.md       # Rangkuman status proyek & context handoff terkini
├── specs/
│   ├── 01_data_extraction.spec.md       # Kontrak YouTube Data API v3 & PII SHA-256
│   ├── 02_preprocessing_labeling.spec.md # Kontrak normalisasi & Weak Supervision (Gemini API)
│   └── 03_model_training.spec.md        # Kontrak IndoRoBERTa vs XLM-RoBERTa benchmark
├── data/
│   ├── raw/                   # Immutable raw comments (yt_comments_all_combined.csv)
│   └── processed/             # 100% Pure Human Gold Standard (train.csv: 667, val.csv: 287)
├── src/
│   ├── extraction/
│   │   └── extractor.py       # YouTubeCommentExtractor (Regex URL Parser, SHA-256 PII Hashing)
│   ├── preprocessing/
│   │   └── text_cleaner.py    # Slang normalizer (data/slang_dict.json), noise cleaner
│   ├── labeling/
│   │   ├── gemini_labeler.py  # Structured JSON Output via Gemini API
│   │   ├── weak_supervision.py# Hybrid cascade (Rule-Based + Gemini API)
│   │   ├── eval_kappa.py      # Cohen's Kappa (κ) evaluator
│   │   └── dataset_splitter.py# Stratified dataset splitter (Zero leakage)
│   └── models/
│       ├── indoroberta_classifier.py # IndoRoBERTa-base (110M) Multi-Head GELU MLP
│       ├── xlmroberta_classifier.py  # XLM-RoBERTa-large (550M) Multi-Head GELU MLP
│       ├── sahabatai_classifier.py   # Baseline model / comparative wrapper
│       └── metrics_evaluator.py      # Multi-aspect metrics calculator & confusion matrix
├── notebooks/
│   ├── 01_eda_raw_comments.ipynb
│   ├── 02_eda_cleaned_comments.ipynb
│   ├── 03_eda_labeled_dataset.ipynb
│   └── 04_kaggle_training_indoroberta_vs_xlmroberta.ipynb # Notebook pelatihan GPU Kaggle T4
├── outputs/
│   └── iter-01/               # Hasil evaluasi benchmark iterasi 01 (JSON, CSV, CM)
└── journal/                   # Berkas penyusunan paper/jurnal (USB_2026.csv, audit xlsx)
```

---

## 4. Engineering & Scientific Guardrails

### A. Data Governance & Ethics
- **Strict PII Masking:** Hapus atau samarkan atribut identitas pribadi (`authorDisplayName`, `authorChannelId`, foto profil). Wajib gunakan hash SHA-256 (`user_id_hash`).
- **Official API Extraction Only:** Dilarang mengimplementasikan *unauthorized web scraping* pada antarmuka YouTube. Seluruh akuisisi data wajib menggunakan `google-api-python-client` (`YouTube Data API v3`).
- **Data Immutability & Zero Data Leakage:** Jangan pernah menimpa file di `data/raw/`. Validation Set (`data/processed/val.csv`, 287 baris) **TIDAK BOLEH** disentuh oleh augmentasi atau prapemrosesan terikat.

### B. Machine Learning & Modeling Rules
- **Multi-Task GELU MLP Classifier Heads:** Gunakan arsitektur penggabungan `[CLS]` token representation + *Mean Pooling* (1536 dim untuk IndoRoBERTa / 2048 dim untuk XLM-RoBERTa) yang diteruskan ke 4 MLP heads terpisah (`Linear -> GELU -> Dropout -> Linear`).
- **Class Imbalance Loss & Thresholding:**
  - Multi-Aspect Focal Loss ($\gamma = 1.5$) dengan Smoothed Class Weights ($\sqrt{w}$).
  - Aspect-Specific Decision Thresholding ($\theta_{\text{purnajual}}=0.35, \theta_{\text{infra}}=0.40, \theta_{\text{ekonomi}}=0.50, \theta_{\text{kualitas}}=0.50$).
  - Targeted Class Alpha Scaling ($\alpha_{\text{purnajual, positif}}=2.5\times, \alpha_{\text{infra, positif}}=1.8\times$).
- **Contextual Data Augmentation Rule:** Augmentasi data (substitusi sinonim & parafrase kontekstual) **HANYA BISA DITERAPKAN** pada Train Set (`train.csv`).
- **Evaluation Standard:** Evaluasi model tidak hanya mengukur *Accuracy*, melainkan wajib mengukur **Active-Sentiment Macro F1**, All-Class Macro F1, Weighted F1, dan **Exact Match Ratio (Subset Accuracy)**.
- **Reproducibility:** Selalu kunci *seed* acak (`random_seed = 42`) pada PyTorch, Transformers, NumPy, dan Sklearn.

### C. Coding Standards
- Bahasa utama: Python $\ge 3.10$.
- Tuliskan *docstring* ringkas yang memuat: tujuan fungsi, parameter input, dan return type.
- Tangani error secara eksplisit (*graceful exception handling*).

---

## 5. Agent Interaction Workflow

Saat menerima instruksi pengembangan modul atau eksperimen baru:

1. **Analyze:** Baca spesifikasi terkait di direktori `./specs/` dan `brief/project_state.md`.
2. **Clarify:** Jika terdapat instruksi yang ambigu, tanyakan sebelum menulis kode.
3. **Plan:** Tampilkan ringkasan rencana perubahan berkas atau pembuatan kode.
4. **Execute:** Hasilkan kode yang modular, bebas bug, dan langsung dapat dieksekusi.
