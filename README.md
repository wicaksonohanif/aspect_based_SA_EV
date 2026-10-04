# 🚗 SentyBoard: Chinese EV Perception Data Mining & Aspect-Based Sentiment Analysis (USB 2026)

[![Streamlit App](https://img.shields.io/badge/Streamlit-SentyBoard%20v1.0.0-1e3c72?style=flat&logo=streamlit)](https://share.streamlit.io/)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-XLM--RoBERTa--EV--ABSA-ffD21E?style=flat&logo=huggingface)](https://huggingface.co/wicaksonohanif/xlm-roberta-ev-absa)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python)](https://www.python.org/)

SentyBoard adalah platform riset data mining dan sistem analisis sentimen berbasis aspek (**Aspect-Based Sentiment Analysis / ABSA**) cerdas untuk memetakan persepsi publik Indonesia terhadap ekosistem mobil listrik (EV) pabrikan Tiongkok (seperti Wuling, BYD, Chery, Seres, Neta, MG) yang bersumber dari media sosial YouTube.

---

## 📌 Ringkasan Proyek

- **Fokus Domain:** NLP, Text Mining, ABSA Multi-Head Classification, Fine-Tuning Transformer.
- **Sumber Data:** Komentar publik YouTube (menggunakan YouTube Data API v3 resmi dengan *Strict PII Masking*).
- **Model Terbaik:** **XLM-RoBERTa-large (550M)** dengan arsitektur Multi-Head GELU MLP & Focal Loss ($\gamma=1.5$).
- **Hasil Benchmark 5-Fold CV:** XLM-RoBERTa-large mengungguli IndoRoBERTa-base dengan skor **Macro F1 Active 58.26% ± 3.38%** (+10.05% peningkatan absolut).
- **4 Aspek Utama yang Dianalisis:**
  1. 🔌 **Infrastruktur (`infra`)**: SPKLU, charging station, jarak tempuh, durasi pengisian daya.
  2. 💰 **Ekonomi (`ekonomi`)**: Harga kendaraan, pajak, insentif/subsidi, depresiasi nilai jual kembali.
  3. 🚘 **Kualitas (`kualitas`)**: Quality control, build quality, suspensi, baterai LFP/Blade, ADAS.
  4. 🛠️ **Purna Jual (`purnajual`)**: Dealer resmi, layanan servis, ketersediaan suku cadang, garansi.
- **Target Output:** Model terpublikasi di Hugging Face Hub (`wicaksonohanif/xlm-roberta-ev-absa`) & Interactive Web Dashboard berbasis **Streamlit** (SentyBoard v1.0.0).

---

## 💻 Fitur Utama Dashboard SentyBoard (v1.0.0)

Aplikasi **SentyBoard** (`app.py`) memiliki tampilan **Dark Mode** elegan berbasis tipografi **Inter** dan sidebar **Blue Gradient**, yang terbagi dalam 2 Mode Utama:

1. **📊 Mode Analytics (Executive Dashboard):**
   - **SaaS Executive KPI Cards:** Total Komentar, Video YouTube, % Coverage Aspek, Aspek Dominan, dan Sentimen Dominan.
   - **2-Column Grid Visualization:** Grouped Bar Chart Distribusi Sentimen, Overall Sentiment Pie Chart, Aspect Donut Chart, Aspect Co-occurrence Heatmap Matrix, dan Word Count Distribution Boxplot per Sentimen.
   - **Aspect Deep-Dive & WordCloud:** Visualisasi kata kunci per-sentimen (Positif, Netral, Negatif) untuk setiap aspek.
   - **Top Liked Comments Gallery:** Galeri kartu komentar terpopuler dengan filter aspek & sentimen interaktif.
   - **Sample Data One-Click Loading:** Tombol sekali klik untuk memuat dataset sampel `valid_labeled_comments.csv`.

2. **🤖 Mode Labeling (Inference Engine):**
   - **Automated Multi-Head Classification:** Inferensi otomatis untuk komentar mentah hasil tarikan YouTube API.
   - **Serverless Cloud API & Zero-RAM Protection:** Kompatibel dengan Streamlit Cloud (bebas crash Out of Memory).
   - **Standardized Export:** Pengunduhan hasil pelabelan otomatis dalam format CSV/XLSX standar 4 aspek.

---

## 📊 Hasil Benchmark Evaluasi (5-Fold Stratified Cross-Validation)

| Model | Mean Macro F1 (Active Sentiments) | Mean Macro F1 (All Classes) | Mean Accuracy |
| :--- | :---: | :---: | :---: |
| **IndoRoBERTa-base (110M)** | `0.4820 ± 0.0245` | `0.5988 ± 0.0211` | `0.7937 ± 0.0100` |
| **XLM-RoBERTa-large (550M)** | **`0.5826 ± 0.0338`** | **`0.6666 ± 0.0264`** | **`0.8382 ± 0.0080`** |

*Detail lengkap grafik per-aspek & kurva pembelajaran tersedia di notebook [`notebooks/05_eval.ipynb`](notebooks/05_eval.ipynb).*

---

## 📂 Struktur Repositori

```text
usb_2026/
├── brief/               # Dokumen PRD, Agent Directives, & State Proyek
├── specs/               # Spesifikasi Teknis Modul (01..05 Spec-Driven Development)
├── data/
│   ├── raw/             # Data mentah ekstraksi YouTube (CSV/JSONL) - Immutable
│   ├── interim/         # Data hasil preprocessing & valid_labeled_comments.csv
│   ├── dummy/           # Dataset dummy untuk tes inferensi
│   └── processed/       # Dataset terlabel Gold Standard (954 audit)
├── src/
│   ├── extraction/      # Modul integrasi YouTube Data API v3 resmi
│   ├── preprocessing/   # Text cleaner & normalisasi slang bahasa Indonesia
│   ├── labeling/        # Weak supervision (Rule-based + Gemini 1.5 Flash)
│   └── deployment/      # Engine pemuat model (model_loader.py) & data processor
├── notebooks/           # Notebook EDA (01-03), Kaggle Training (04), & Evaluasi (05_eval)
├── models/              # Checkpoint lokal model XLM-RoBERTa
├── outputs/             # Metrik evaluasi 5-Fold CV (iter-03, iter-04)
├── assets/              # Aset gambar banner & visual UI
├── app.py               # Main Entrypoint Dashboard Streamlit SentyBoard
├── upload_model_to_hf.py# Skrip otomatis pengunggah bobot model ke Hugging Face
├── requirements.txt     # Dependensi proyek
└── README.md
```

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Persiapan Environment
```bash
# Clone repositori
git clone https://github.com/wicaksonohanif/aspect_based_SA_EV.git
cd aspect_based_SA_EV

# Install dependensi
pip install -r requirements.txt
```

### 2. Jalankan Dashboard Streamlit
```bash
streamlit run app.py
```
Buka peramban di `http://localhost:8501` dan klik tombol **"📊 Gunakan Data Sampel"** di halaman utama!

---

## 🌐 Publikasi Artifact

- **Hugging Face Model:** [`wicaksonohanif/xlm-roberta-ev-absa`](https://huggingface.co/wicaksonohanif/xlm-roberta-ev-absa) (Weights: `pytorch_model.bin` ~2.15 GB)
- **Deployment Spec:** [`specs/05_deployment_cloud_huggingface.spec.md`](specs/05_deployment_cloud_huggingface.spec.md)
