# PRD.md - Product Requirements Document

## 1. Executive Summary & Context
* **Project Name:** Chinese EV Perception Data Mining & Sentiment Dashboard
* **Target Competition:** Lomba Penambangan Data USB 2026
* **Theme:** *Ekosistem Digital Cerdas untuk Masa Depan Indonesia yang Inklusif dan Berkelanjutan*
* **Target Deadline:** 5 Oktober 2026
* **Problem Statement:** Penetrasi mobil listrik (EV) pabrikan China (seperti Wuling, BYD, Chery) membuka akses mobilitas hijau yang inklusif di Indonesia. Namun, adopsi massal terhambat oleh keraguan publik terkait durabilitas baterai, depresiasi nilai jual kembali, dan kesiapan infrastruktur SPKLU. Riset terdahulu berbasis transformer bahasa Indonesia terhenti pada performa Macro F1-Score yang rendah (~0,3319) akibat variasi bahasa informal warganet.
* **Proposed Solution:** 
  1. Membangun *pipeline* Aspect-Based Sentiment Analysis (ABSA) berbasis *Weak Supervision* terkalibrasi statistik.
  2. Mengomparasi performa model *baseline* **IndoRoBERTa** (*Encoder-only*) secara langsung melawan Foundation LLM lokal **Sahabat-AI** (*Decoder-only*).
  3. Mempublikasikan bobot model ke **Hugging Face Hub** demi transparansi dan reproduktibilitas riset.
  4. Mengembangkan prototipe web publik berbasis **Streamlit** (dihosting secara gratis di Streamlit Community Cloud) yang memvisualisasikan matriks persepsi konsumen dan simulasi inferensi.

---

## 2. Goals & Success Metrics

### A. Research & Competition Goals
* **G-1:** Menghasilkan korpus data teks opini EV teranotasi aspek-sentimen yang valid secara etika dan statistik.
* **G-2:** Membandingkan efektivitas arsitektur *Encoder-only* (IndoRoBERTa) vs *Decoder-only LLM* (Sahabat-AI) pada domain teks informal YouTube.
* **G-3:** Menyediakan *Interactive Policy Dashboard* berbasis web yang mudah diakses publik tanpa biaya operasional ($0).

### B. Measurable Metrics (KPIs)
| Dimensi | Metrik Sukses | Target Nilai |
|---|---|---|
| **Data Quality** | Inter-Annotator Agreement (Cohen's Kappa $\kappa$) | $\kappa \ge 0,61$ (*Substantial Agreement*) |
| **Model Performance** | Primary: Macro-averaged F1-Score | $> 0,65$ (Aspek & Polaritas) |
| **Deployment Cost** | Total Cloud & Hosting Expenditure | **Rp0 ($0)** — 100% Free Tier |
| **Web Accessibility** | Deployment Platform | Streamlit Community Cloud (Public HTTPS) |
| **Reproducibility** | Model Hub Artifacts | Model & Tokenizer terunggah di Hugging Face |
| **System Integrity** | PII Masking Compliance | 100% data akun YouTube teranonimkan |

---

## 3. System Architecture & Tech Stack


```

[YouTube Data API v3] ──> data/raw/
│
▼
[Preprocessing & Masking] ──> data/interim/
│
▼
[Weak Supervision Labeling] ──> data/processed/ (Validated κ)
│
┌──────────────────┴──────────────────┐
▼                                     ▼
[Baseline: IndoRoBERTa]                 [Comparator: Sahabat-AI]
(Encoder-only)                         (Decoder-only LLM)
│                                     │
└──────────────────┬──────────────────┘
▼
[Evaluation & Selection]
│
▼
[Hugging Face Hub] (Artifacts Hosting)
│
▼
[Pre-computed / API Bridge]
│
▼
[Streamlit Community Cloud]
(Interactive Dashboard Prototipe)

```

### Technology Stack Specifications
* **Core Language:** Python $\ge 3.10$
* **Data Extraction:** `google-api-python-client` (YouTube Data API v3)
* **NLP & Modeling:** Hugging Face `transformers`, `torch`, `peft` (jika menggunakan LoRA), `scikit-learn`
* **Model Artifacts:** Hugging Face Hub (Model Repository)
* **Frontend Dashboard:** `Streamlit`, `Plotly Express`, `Pandas`
* **Cloud Hosting:** Streamlit Community Cloud (Koneksi CI/CD langsung dari repositori GitHub)

---

## 4. Functional Requirements

### FR-1: Data Extraction Module (`specs/01_data_extraction.spec.md`)
* **FR-1.1:** Mengekstrak komentar publik dari daftar tautan video ulasan dan penggunaan mobil listrik China di Indonesia.
* **FR-1.2:** Menjaga kuota API harian di bawah 10.000 unit/hari.
* **FR-1.3:** Melakukan *PII Masking* seketika saat data disimpan: menghapus nama pengguna, ID kanal, dan URL profil warganet, lalu menyematkan `user_id_hash`.

### FR-2: Preprocessing & Weak Supervision Module (`specs/02_preprocessing_labeling.spec.md`)
* **FR-2.1:** Pembersihan teks (pembersihan karakter non-ASCII, tautan URL, normalisasi singkatan/slang warganet Indonesia).
* **FR-2.2:** Pelabelan otomatis awal (*heuristic pre-labeling*) berbasis *prompting* LLM untuk 4 aspek strategis:
  1. *Infrastruktur & Jangkauan*
  2. *Ekonomi & Finansial*
  3. *Kualitas & Durabilitas*
  4. *Purna Jual & Ekosistem*
* **FR-2.3:** Validasi manual bertingkat (*stratified human audit*) sebesar 20–30% dari total data.
* **FR-2.4:** Kalkulasi koefisien **Cohen's Kappa ($\kappa$)** antara label mesin dan verifikasi manusia.

### FR-3: Model Experimentation & Hugging Face Sync (`specs/03_model_comparison.spec.md`)
* **FR-3.1:** *Fine-tuning* model *baseline* **IndoRoBERTa** menggunakan data latih terstratifikasi.
* **FR-3.2:** Evaluasi model **Sahabat-AI** menggunakan *Few-Shot In-Context Prompting* atau *LoRA Fine-Tuning*.
* **FR-3.3:** Evaluasi komparatif pada *Test Set* yang sama dengan metrik: Macro Precision, Recall, Macro F1-Score, dan *Latency*.
* **FR-3.4:** Mengunggah artefak model *fine-tuned* dan *tokenizer* ke Hugging Face Hub.

### FR-4: Pivot Aggregation & Policy Insight (`specs/04_pivot_analysis.spec.md`)
* **FR-4.1:** Melakukan inferensi pada seluruh korpus menggunakan model terbaik.
* **FR-4.2:** Mengagregasi data ke dalam matriks pivot dua dimensi: **Aspek** $\times$ **Polaritas Sentimen**.
* **FR-4.3:** Mengidentifikasi *Key Drivers* pada sentimen negatif di tiap aspek sebagai dasar rekomendasi kebijakan transportasi berkelanjutan.

### FR-5: Streamlit Web Dashboard (`specs/05_web_dashboard.spec.md`)
* **FR-5.1:** Halaman *Overview & Data Ingestion*: Menampilkan ringkasan statistik dataset korpus komentar.
* **FR-5.2:** Halaman *Comparative Evaluation*: Menampilkan tabel perbandingan metrik IndoRoBERTa vs Sahabat-AI.
* **FR-5.3:** Halaman *Perception Insights*: Visualisasi interaktif Plotly (diagram batang bertumpuk aspek $\times$ sentimen, filter berdasarkan brand/aspek).
* **FR-5.4:** Halaman *Live Inference Playground*: Kotak input teks tunggal untuk menguji klasifikasi aspek dan sentimen secara langsung menggunakan model IndoRoBERTa (via inferensi ringan lokal atau HF Inference API).

---

## 5. Non-Functional & Operational Guardrails
* **Zero Infrastructure Cost:** Seluruh arsitektur wajib beroperasi di atas tier gratis (GitHub, Hugging Face Hub, Streamlit Cloud).
* **Crash Prevention (Anti-OOM):** Dashboard Streamlit memuat data hasil inferensi terkomputasi (*pre-computed data*) untuk agregasi visual utama, guna mencegah lonjakan memori (RAM $> 1$ GB) pada tier hosting gratis.
* **Traceability:** Setiap eksperimen, log metrik, dan keputusan arsitektur dicatat ke dalam berkas markdown pendukung (`outputs/EXPERIMENT_LOGS.md` dan `PROJECT_STATE.md`).
