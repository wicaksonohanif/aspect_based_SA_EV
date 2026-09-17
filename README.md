# 🚗 Chinese EV Perception Data Mining & Aspect-Based Sentiment Analysis (USB 2026)

Proyek riset data mining dan analisis sentimen berbasis aspek (**Aspect-Based Sentiment Analysis / ABSA**) untuk memetakan persepsi publik Indonesia terhadap ekosistem mobil listrik (EV) pabrikan China (seperti Wuling, BYD, Chery, Seres, dan Neta) dari media sosial YouTube.

---

## 📌 Ringkasan Proyek

- **Fokus Domain:** NLP (Text Mining, ABSA, Fine-Tuning LLM & Transformer).
- **Sumber Data:** Komentar publik YouTube (menggunakan YouTube Data API v3 resmi dengan *Strict PII Masking*).
- **Komparasi Model:** Membandingkan performa model **IndoRoBERTa** (*Encoder-only*) vs **Sahabat-AI** (*Decoder-only LLM*) pada teks informal warganet Indonesia.
- **4 Aspek Utama yang Dianalisis:**
  1. 🔌 **Infrastruktur & Jangkauan** (SPKLU, baterai, daya jelajah)
  2. 💰 **Ekonomi & Finansial** (Harga jual, depresiasi, pajak)
  3. 🛠️ **Kualitas & Durabilitas** (Material, perakitan, performa)
  4. 🏬 **Purna Jual & Ekosistem** (Dealer, garansi, suku cadang)
- **Target Output:** Artefak model terpublikasi di Hugging Face Hub & Dashboard Interaktif berbasis **Streamlit**.

---

## 📂 Struktur Repositori

```text
├── brief/               # Dokumen PRD, Agent Directives, & State Proyek
├── specs/               # Spesifikasi Teknis Modul (Spec-Driven Development)
├── data/
│   ├── raw/             # Data mentah ekstraksi YouTube (CSV/JSONL) - Immutable
│   ├── interim/         # Data hasil preprocessing & pembersihan teks
│   └── processed/       # Dataset terlabel (Train, Val, Test)
├── src/
│   ├── extraction/      # Modul integrasi YouTube Data API v3 resmi
│   ├── preprocessing/   # Cleaning & normalisasi slang bahasa Indonesia
│   ├── labeling/        # Weak supervision & kalkulasi Cohen's Kappa
│   └── models/          # Script eksperimen model IndoRoBERTa & Sahabat-AI
├── requirements.txt     # Dependensi proyek
└── README.md
```

---

## 🚀 Panduan Memulai (Quick Start untuk Kontributor)

### 1. Kloning & Persiapan Environment
```bash
# Aktifkan virtual environment Anda (Python >= 3.10)
# Instal dependensi proyek
pip install -r requirements.txt
```

### 2. Konfigurasi API Key
Salin file `.env.example` menjadi `.env` lalu masukkan API Key YouTube Data API v3 Anda:
```env
YOUTUBE_API_KEY=API_KEY_YOUTUBE_ANDA_DI_SINI
```

### 3. Ekstraksi Data Komentar YouTube
1. Masukkan tautan/ID video YouTube target pada file `data/video_list.txt` (satu link per baris).
2. Jalankan modul ekstraksi:
   ```bash
   python -m src.extraction.extractor
   ```
3. Hasil ekstraksi akan tersimpan otomatis di `data/raw/yt_comments_<timestamp>.csv`.

---

## ⚙️ Prinsip Pengkodean (Spec-Driven Development)

1. **Spec First, Code Later:** Setiap pengembangan modul baru **wajib** diawali dengan dokumen spesifikasi di direktori `./specs/`.
2. **Strict PII Protection:** Dilarang menyimpan nama/ID akun YouTube asli warganet. Semua identitas disamarkan menggunakan hash SHA-256 (`user_id_hash`).
3. **Official API Only:** Ekstraksi data wajib menggunakan modul resmi (`google-api-python-client`), dilarang menggunakan *web scraping* tak resmi.
