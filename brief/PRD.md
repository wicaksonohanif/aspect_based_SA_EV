# PRD.md - Product Requirements Document

## 1. Executive Summary & Context
* **Project Name:** Aspect-Based Sentiment Analysis (ABSA) Mobil Listrik (EV) China di Indonesia
* **Target Competition / Goal:** Lomba Penambangan Data USB 2026 & Publikasi Jurnal Penelitian
* **Theme:** *Ekosistem Digital Cerdas untuk Masa Depan Indonesia yang Inklusif dan Berkelanjutan*
* **Target Deadline:** 5 Oktober 2026
* **Problem Statement:** Penetrasi mobil listrik (EV) pabrikan China (seperti Wuling, BYD, Chery, Neta, Seres) membuka akses mobilitas hijau yang inklusif di Indonesia. Namun, adopsi massal terhambat oleh keraguan publik terkait durabilitas baterai, depresiasi nilai jual kembali, kesiapan infrastruktur SPKLU, dan keandalan purnajual. Riset terdahulu berbasis transformer bahasa Indonesia terhenti pada performa Macro F1-Score yang rendah akibat variasi bahasa informal warganet dan ketimpangan kelas (*class imbalance*) yang ekstrem.
* **Proposed Solution:** 
  1. Membangun *pipeline* Aspect-Based Sentiment Analysis (ABSA) multi-task 4 aspek (`infra`, `ekonomi`, `kualitas`, `purnajual`) berbasis *Hybrid Weak Supervision* (Rule-Based + Google Gemini API) yang divalidasi hingga *100% Pure Human Gold Standard Dataset*.
  2. Mengomparasi secara ketat performa arsitektur **Discriminative Encoder**: **IndoRoBERTa-base (110M Monolingual)** vs **XLM-RoBERTa-large (550M Multilingual)** menggunakan Multi-Task GELU MLP Classifier Heads, Multi-Aspect Focal Loss ($\gamma = 1.5$), dan *Aspect-Specific Decision Thresholding*.
  3. Mengarahkan iterasi berikutnya pada *Contextual Data Augmentation* untuk memperkuat generalisasi model pada kelas minoritas ekstrem (`purnajual positif` & `infra positif`).
  4. Mempublikasikan bobot model ke **Hugging Face Hub** serta menyediakan prototipe web berbasis **Streamlit** (Streamlit Community Cloud) demi transparansi dan reproduktibilitas.

---

## 2. Goals & Success Metrics

### A. Research & Competition Goals
* **G-1:** Menghasilkan korpus data opini EV teranotasi 4 aspek & sentimen yang valid 100% audit manusia (*Pure Human Gold Standard*) dengan zero data leakage.
* **G-2:** Menguji secara empiris keunggulan arsitektur *Monolingual Encoder* (IndoRoBERTa-base) vs *Multilingual Large Encoder* (XLM-RoBERTa-large) pada data teks informal YouTube Indonesia.
* **G-3:** Mengatasi ketimpangan kelas mayoritas `None` (>80%) dan minoritas aktif melalui kombinasi Focal Loss, Smoothed Class Weights, Targeted Alpha Scaling, dan Aspect-Specific Decision Thresholding.
* **G-4:** Menyediakan *Interactive Policy Dashboard* berbasis web ($0 operational cost).

### B. Measurable Metrics (KPIs) & Hasil Iterasi 01

| Dimensi | Metrik Sukses | Target Nilai | Capaian Terkini (Iterasi 01 - XLM-RoBERTa) | Status |
|---|---|---|---|---|
| **Data Quality** | Inter-Annotator Agreement ($\kappa$) | $\kappa \ge 0,61$ (*Substantial Agreement*) | Validasi Manusia 100% Audit (954 sample) | 🟢 Achieved |
| **Overall Accuracy** | Mean Accuracy 4 Aspek | $> 80,0\%$ | **85,45%** (XLM-RoBERTa Large) | 🟢 Achieved |
| **Macro F1 (All-Class)** | Mean Macro F1 4 Kelas | $> 60,0\%$ | **62,92%** (XLM-RoBERTa Large) | 🟢 Achieved |
| **Macro F1 (Active)** | Mean Macro F1 Sentimen Aktif | $> 50,0\%$ | **52,81%** (XLM-RoBERTa Large) | 🟢 Achieved |
| **Subset Accuracy** | Exact Match Ratio 4 Aspek | $> 50,0\%$ | **55,40%** (XLM-RoBERTa Large) | 🟢 Achieved |
| **Deployment Cost** | Total Cloud & Hosting Expenditure | **Rp0 ($0)** — 100% Free Tier | Kaggle T4 GPU + Streamlit Cloud ($0) | 🟢 Achieved |
| **System Integrity** | PII Masking Compliance | 100% anonim (SHA-256) | Terimplementasi di `src/extraction/` | 🟢 Achieved |

---

## 3. System Architecture & Tech Stack

```
[YouTube Data API v3] ──> data/raw/ (yt_comments_all_combined.csv)
│
▼
[Preprocessing & Normalization] ──> src/preprocessing/text_cleaner.py
│
▼
[Weak Supervision (Rule-Based + Gemini API)] ──> src/labeling/weak_supervision.py
│
▼
[Human Audit & Split (70/30 Zero Leakage)] ──> data/processed/ (train.csv: 667, val.csv: 287)
│
┌──────────────────────────────────────┴──────────────────────────────────────┐
▼                                                                             ▼
[Model 1: IndoRoBERTa-base (110M)]                            [Model 2: XLM-RoBERTa-large (550M)]
Concatenated [CLS] + Mean Pooling                              Concatenated [CLS] + Mean Pooling
4 x GELU MLP Heads                                             4 x GELU MLP Heads
Multi-Aspect Focal Loss (γ = 1.5)                              Multi-Aspect Focal Loss (γ = 1.5)
│                                                                             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
                     [Kaggle T4 GPU Execution & Metrics Evaluator]
                     outputs/iter-01/ (JSON, CSV, Confusion Matrices)
                                       │
                                       ▼
                     [Contextual Data Augmentation - Iterasi 02]
                     (Targeted Synonym & Paraphrase on Train Set)
                                       │
                                       ▼
                     [Hugging Face Hub & Streamlit Dashboard]
```

### Technology Stack Specifications
* **Core Language:** Python $\ge 3.10$
* **Data Extraction:** `google-api-python-client` (YouTube Data API v3)
* **Weak Supervision:** `google-generativeai` (Google Gemini API `gemini-1.5-flash` with Structured JSON Output)
* **NLP & Deep Learning:** PyTorch, Hugging Face `transformers`, `scikit-learn`
* **Execution Environment:** Kaggle Notebooks (NVIDIA T4 GPU 16GB VRAM, Full Fine-Tuning)
* **Visualization & Frontend:** Streamlit, Plotly Express, Pandas

---

## 4. Functional Requirements

### FR-1: Data Extraction Module (`specs/01_data_extraction.spec.md`)
* **FR-1.1:** Mengekstrak komentar *top-level* dari daftar tautan/ID video YouTube ulasan EV.
* **FR-1.2:** Menjaga kuota API harian di bawah 10.000 unit/hari.
* **FR-1.3:** Melakukan *PII Masking* seketika saat data disimpan (SHA-256 hashing `user_id_hash`).

### FR-2: Preprocessing & Weak Supervision Module (`specs/02_preprocessing_labeling.spec.md`)
* **FR-2.1:** Pembersihan teks (normalisasi slang otomotif `data/slang_dict.json`, reduksi karakter ganda, pembersihan mention/URL).
* **FR-2.2:** Pelabelan otomatis *Hybrid Cascade* (Rule-based lokal untuk teks eksplisit + Gemini API Structured JSON untuk teks ambigu/sarkastik).
* **FR-2.3:** Validasi manual audit manusia hingga dataset 100% *Pure Human Gold Standard* terbebas dari kesalahan mesin.
* **FR-2.4:** Pembagian terstratifikasi tanpa kebocoran data (*zero data leakage*): Train (70% / 667 baris) dan Val (30% / 287 baris).

### FR-3: Model Experimentation & Benchmark (`specs/03_model_training.spec.md`)
* **FR-3.1:** *Full fine-tuning* model **IndoRoBERTa-base (110M)** dan **XLM-RoBERTa-large (550M)** di GPU Kaggle T4.
* **FR-3.2:** Pengaplikasian Multi-Aspect Focal Loss ($\gamma = 1.5$), Smoothed Class Weights ($\sqrt{w}$), dan *Aspect-Specific Thresholding* ($\theta_{\text{purnajual}}=0.35, \theta_{\text{infra}}=0.40, \theta_{\text{ekonomi}}=0.50, \theta_{\text{kualitas}}=0.50$).
* **FR-3.3:** Pelaksanaan Iterasi 02: *Contextual Data Augmentation* (substitusi sinonim & parafrase kontekstual) khusus pada kelas minoritas ekstrem (`purnajual positif` & `infra positif`) di Train Set.
* **FR-3.4:** Evaluasi komparatif pada Val Set (287 baris) dengan metrik: Accuracy, Macro/Weighted F1, Active Macro F1, Exact Match Ratio, dan Matriks Kebingungan 4x4.

### FR-4: Streamlit Web Dashboard & Policy Insights
* **FR-4.1:** Visualisasi interaktif Plotly (matriks aspek $\times$ sentimen, perbandingan performa IndoRoBERTa vs XLM-RoBERTa).
* **FR-4.2:** *Live Inference Playground* untuk pengujian teks input pengguna secara real-time.

---

## 5. Non-Functional & Operational Guardrails
* **Zero Infrastructure Cost:** Seluruh eksperimen dan deployment beroperasi pada tier gratis (Kaggle GPU T4, GitHub, Streamlit Cloud).
* **Anti Data Leakage:** Validation Set (287 baris) dikunci ketat dan tidak boleh disentuh oleh augmentasi atau prapemrosesan terikat.
* **Traceability & Spec Compliance:** Setiap eksperimen dan log metrik wajib dicatat di `outputs/iter-XX/` dan dirangkum di `brief/project_state.md`.
