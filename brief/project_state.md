# project_state.md — Project Snapshot & Activity Log

- **Tanggal Sesi:** 19 September 2026
- **Sesi:** Sesi 3 — Pengumpulan Korpus Data 3.000+ Komentar, Pembuatan & Verifikasi Modul Tahap 2 (SPEC-02), Pembersihan Data Gabungan, & Notebook EDA 02
- **Status Proyek:** Phase 1 Selesai 100% (3.016 Komentar Mentah Unik) & Phase 2 Implementation Complete (Ready for Auto-Labeling)

---

## 📌 Ringkasan Aktivitas yang Telah Diselesaikan Kemarin (19 September 2026)

### 1. Ekstraksi Skala Besar & Penggabungan Korpus Data Mentah (Phase 1 Final)
- Menjalankan modul ekstraksi `src/extraction/extractor.py` untuk 17 video YouTube ulasan & komparasi EV China (JAECOO J5, BYD Atto 1, Geely EX-2, dll.).
- **Penggabungan Data (*Concat & Deduplication*):**
  - Menggabungkan 3 berkas raw CSV (`yt_comments_20260918_113832.csv`, `yt_comments_20260919_072315.csv`, dan `yt_comments_20260919_072855.csv`).
  - Menghapus duplikasi berdasarkan `comment_id`.
  - **Hasil Akhir Korpus Mentah:** Berhasil mengumpulkan **3.016 komentar mentah unik** yang disimpan di `data/raw/yt_comments_all_combined.csv` (Resmi melampaui target ideal 3.000 data riset!).

### 2. Pembangunan & Pemasangan Seluruh Modul Tahap 2 (SPEC-02)
- Memperbarui `requirements.txt` dan menginstal pustaka `google-generativeai` & `scikit-learn` pada virtual environment.
- Mengonfigurasi `GEMINI_API_KEY` pada berkas `.env` dan `.env.example`.
- Membangun modul-modul berikut sesuai spesifikasi `specs/02_preprocessing_labeling.spec.md`:
  - `data/slang_dict.json`: Kamus normalisasi kata gaul, singkatan, & istilah otomotif Indonesia (`spklu`, `mobkas`, `ngecas`, `boncos`, `worth it`, dll.).
  - `src/preprocessing/text_cleaner.py`: Modul `TextCleaner` untuk lowercasing, pembersihan URL/mention, reduksi huruf berulang, dan substitusi slang.
  - `src/labeling/gemini_labeler.py`: Modul `GeminiAspectLabeler` yang terhubung dengan **Google Gemini API** (`gemini-1.5-flash` / `gemini-1.5-pro`) menggunakan fitur *Structured JSON Output Mode*.
  - `src/labeling/weak_supervision.py`: Modul `WeakSupervisionEngine` yang mengimplementasikan *Hybrid Cascade Flow* (Rule-Based lokal instan $\rightarrow$ Google Gemini API Cloud untuk teks ambigu/sarkastik).
  - `src/labeling/eval_kappa.py`: Modul `CohenKappaEvaluator` untuk membuat sampel *Human Audit* 20% (`data/interim/human_audit_sample.csv`) dan menghitung statistik koefisien **Cohen's Kappa ($\kappa$)** antara Label Mesin vs 1 Auditor Manusia.
  - `src/labeling/dataset_splitter.py`: Modul `DatasetSplitter` untuk membagi dataset final 70/15/15 ke `train.csv`, `val.csv`, dan `test.csv` di `data/processed/` dengan seed acak 42.

### 3. Pembersihan Data Gabungan (*Text Cleaning Execution*)
- Menjalankan `TextCleaner` pada seluruh 3.016 komentar mentah gabungan di `data/raw/yt_comments_all_combined.csv`.
- **Hasil Pembersihan:** Berhasil menghasilkan dataset bersih di `data/interim/cleaned_comments_all.csv` (3.016 baris terbersihkan) dalam waktu < 1 detik.

### 4. Penyusunan Notebook Analisis EDA 02
- Membuat dan memvalidasi Jupyter Notebook `notebooks/02_eda_cleaned_comments.ipynb` untuk menginspeksi perbandingan teks asli vs bersih, distribusi panjang kata/karakter, dan frekuensi kata paling dominan pada data interim gabungan.

---

## 📂 Status Artefak Proyek Currently Active

| Artefak / File | Status | Keterangan |
|---|---|---|
| `brief/PRD.md` | Active | Reference PRD |
| `brief/AGENTS.md` | Active | Directives & Guardrails |
| `brief/project_state.md` | Updated | Snapshot Sesi 3 (19 Sep 2026) |
| `specs/01_data_extraction.spec.md` | Approved | Data Extraction Spec |
| `specs/02_preprocessing_labeling.spec.md` | Approved | Preprocessing & Gemini API Hybrid Spec |
| `data/raw/yt_comments_all_combined.csv` | Completed | 3.016 Unique Raw Comments |
| `data/interim/cleaned_comments_all.csv` | Completed | 3.016 Cleaned Comments |
| `data/slang_dict.json` | Completed | Automotive Slang Dictionary |
| `src/preprocessing/text_cleaner.py` | Verified | Text Cleaning Module |
| `src/labeling/gemini_labeler.py` | Verified | Google Gemini API Structured JSON Labeler |
| `src/labeling/weak_supervision.py` | Verified | Hybrid Cascade Weak Supervision Engine |
| `src/labeling/eval_kappa.py` | Verified | Cohen's Kappa & Audit Generator |
| `src/labeling/dataset_splitter.py` | Verified | 70/15/15 Dataset Splitter |
| `notebooks/01_eda_raw_comments.ipynb` | Verified | EDA Raw Comments Notebook |
| `notebooks/02_eda_cleaned_comments.ipynb` | Verified | EDA Cleaned Comments Notebook |
| `.env` & `.env.example` | Updated | Configured with YOUTUBE_API_KEY & GEMINI_API_KEY |
| `README.md` | Active | Contributor Overview |

---

## 🎯 Langkah Selanjutnya (Next Actions)

1. **Eksekusi Pelabelan Otomatis Hybrid (Gemini API + Rule-Based):**
   - Menjalankan `WeakSupervisionEngine` pada `data/interim/cleaned_comments_all.csv` untuk menghasilkan `data/interim/auto_labeled_comments_all.csv`.
2. **Generasi & Audit Sampel Manusia (Human Audit 20%):**
   - Menjalankan `CohenKappaEvaluator.generate_human_audit_sample()` untuk menghasilkan `data/interim/human_audit_sample.csv` ($\approx 600$ baris sampel).
   - Pengguna (1 Auditor Manusia) mengisikan kolom verifikasi `human_infra`, `human_ekonomi`, `human_kualitas`, `human_purnajual`.
3. **Kalkulasi Cohen's Kappa ($\kappa$) & Dataset Split (Phase 2 Finalization):**
   - Menjalankan `calculate_kappa_scores()` untuk memastikan target PRD $\kappa \ge 0,61$ tercapai.
   - Menjalankan `DatasetSplitter.split_dataset()` untuk menghasilkan `train.csv`, `val.csv`, dan `test.csv` di `data/processed/`.
