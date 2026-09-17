# AGENTS.md — Research Directive

## 1. Project Overview & Scope

- **Project Name:** Chinese EV Perception Data Mining
- **Domain:** Natural Language Processing (Aspect-Based Sentiment Analysis / ABSA)
- **Primary Objective:** Ekstraksi sentimen dan aspek persepsi publik terhadap mobil listrik (fokus EV pabrikan China) dari komentar YouTube menggunakan komparasi model (IndoRoBERTa vs Sahabat-AI).

---

## 2. Core Operational Philosophy: Spec-Driven Development (SDD)

Sebagai agen AI, Anda **WAJIB** tunduk pada prinsip Spec-Driven Development:

1. **Spec First, Code Later:** Jangan membuat skrip implementasi, modul, atau _pipeline_ sebelum dokumen spesifikasi teknis (`./specs/<feature>.spec.md`) dibuat, diverifikasi, dan disetujui.
2. **Traceability:** Setiap berkas eksperimen harus merujuk ke file pada berkas spesifikasi terkait.
3. **No Hallucinated Packages:** Hanya gunakan pustaka yang tercantum di `requirements.txt`. Jika butuh dependensi baru, usulkan pembaruan spesifikasi dependensi terlebih dahulu.

---

## 3. Directory Structure

```text
├── README.md
│   └── Ringkasan proyek yang tampil pada github
|
├── brief/
│   └── AGENTS.md
│   │   └── Kontrak agentic AI
│   └── PRD.md
│   │   └── Dokumen kebutuhan proyek
│   └── project_state.md
│   │   └── Dokumentasi pekerjaan terakhir
|
├── specs/
│   ├── 01_data_extraction.spec.md
│   │   └── Spesifikasi kontrak YouTube Data API v3
│   ├── 02_preprocessing_labeling.spec.md
│   │   └── Kontrak normalisasi teks & weak supervision
│   ├── 03_model_comparison.spec.md
│   │   └── Kontrak pelatihan IndoRoBERTa & inferensi Sahabat-AI
│   └── 04_pivot_analysis.spec.md
│       └── Kontrak agregasi pivot aspek × sentimen
│
├── docs/
│   ├── LICENSE.md
│   │   └── Sitasi lisensi model
│   └── LITERATURE_REVIEW.md
│       └── Sitasi penelitian terkait
│
├── data/
│   ├── README.md
│   │   └── Penjelasan struktur direktori data & tata kelola
│   ├── raw/
│   │   └── Data mentah YouTube API
│   ├── interim/
│   │   └── Data setelah normalisasi teks
│   └── processed/
│       └── Dataset berlabel final (Train, Val, Test sets)
│
├── src/
│   ├── README.md
│   │   └── Deskripsi modul sumber kode Python
│   ├── extraction/
│   │   └── Integrasi YouTube Data API v3 resmi
│   ├── preprocessing/
│   │   └── Cleaning & tokenization
│   ├── labeling/
│   │   └── Weak supervision (LLM prompt + Cohen's Kappa)
│   ├── models/
│   │   └── Model IndoRoBERTa & Sahabat-AI
│
├── notebooks/
│   └── README.md
│       └── Petunjuk eksperimen Jupyter
│
├── outputs/
│   ├── EXPERIMENT_LOGS.md
│   │   └── Rekapitulasi hasil metrik evaluasi model
│   └── reports/
│       └── Tabel pivot dan artefak visual
│
├── requirements.txt
```

## 4. Engineering & Scientific Guardrails

### A. Data Governance & Ethics

- **Strict PII Masking:** Hapus atau samarkan atribut identitas pribadi (`authorDisplayName`, `authorChannelId`, foto profil). Gunakan penomoran acak anonim (`user_id_hash`).
- **Official API Extraction Only:** Dilarang mengimplementasikan _unauthorized web scraping_ (BeautifulSoup/Selenium) pada antarmuka YouTube. Seluruh akuisisi data wajib menggunakan modul resmi `google-api-python-client` (`YouTube Data API v3`).
- **Data Immutability:** Jangan pernah menimpa file di dalam direktori `data/raw/`.

### B. Machine Learning & Modeling Rules

- **Anti Data Leakage:** Prapemrosesan teks (seperti fit tokenizer/vectorizer) dan sampling anotasi manusia harus strictly terisolasi dari _Test Set_.
- **Evaluation Standard:** Karena distribusi sentimen media sosial bersifat _imbalanced_, jangan jadikan _Accuracy_ sebagai tolok ukur utama. Wajib utamakan **Macro-averaged F1-Score**, Precision, dan Recall.
- **Reproducibility:** Selalu inisialisasi _seed_ acak (`random_seed = 42`) pada modul PyTorch, Transformers, NumPy, dan Sklearn.

### C. Coding Standards

- Bahasa utama: Python $\ge$ 3.10.
- Tuliskan _docstring_ ringkas yang memuat: tujuan fungsi, parameter input, dan return type.
- Tangani error secara eksplisit.

---

## 5. Agent Interaction Workflow

Saat menerima instruksi pengembangan modul baru:

1. **Analyze:** Baca spesifikasi terkait di direktori `./specs/`.
2. **Clarify:** Jika terdapat instruksi yang ambigu, tanyakan sebelum menulis kode.
3. **Plan:** Tampilkan ringkasan rencana perubahan berkas atau pembuatan kode.
4. **Execute:** Hasilkan kode yang modular, bebas bug, dan langsung dapat dieksekusi.
