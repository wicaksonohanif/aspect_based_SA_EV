# project_state.md — Project Snapshot & Activity Log

- **Tanggal Sesi:** 22 September 2026
- **Sesi:** Sesi 5 — Pemfilteran 100% Murni Audit Manusia (Pure Human Gold Standard 954 Komentar), Re-execution EDA Notebook 03, & Pembagian Dataset Stratified 70/30 (Train 667, Val 287)
- **Status Proyek:** Phase 1 Selesai 100% (3.016 Komentar Mentah Unik) & Phase 2 Selesai 100% (Pure Gold Standard 954 Komentar Valid Ber-aspek di `data/processed/`)

---

## 📌 Ringkasan Aktivitas yang Telah Diselesaikan (22 September 2026)

### 1. Ekstraksi 100% Murni Audit Manusia (Pure Human Gold Standard)
- Mengeliminasi seluruh label mesin (*Rule-Based & Gemini API*) yang kolom human-nya tidak diisi oleh pengguna.
- **Hasil Pemfilteran Murni Manusia:**
  - **954 komentar valid ber-aspek** murni 100% hasil audit manusia (*Human-Audit-GoldStandard*).
  - Berkas master tersimpan di `data/interim/master_labeled_comments.csv` (3.016 baris) dan `data/interim/valid_labeled_comments.csv` (**954 komentar murni audit manusia**).

### 2. Pembagian Dataset Final 70/30 (Train vs Val Stratified Split)
- Menjalankan `DatasetSplitter.split_dataset()` pada `data/interim/valid_labeled_comments.csv` dengan rasio **70% Train** vs **30% Validation** (Stratified Multi-Aspect Split, `random_state=42`).
- **Hasil Akhir Pembagian Dataset (`data/processed/`):**
  - **Train Set (70%):** 667 baris -> `data/processed/train.csv`
  - **Val Set (30%):** 287 baris -> `data/processed/val.csv`
- **Distribusi Sentimen per Aspek (954 dataset Pure Human Gold Standard):**
  - **Infra:** 98 label (37 netral, 33 negatif, 28 positif)
  - **Ekonomi:** 433 label (181 positif, 137 netral, 59 negatif)
  - **Kualitas:** 527 label (218 positif, 208 negatif, 101 netral)
  - **Purnajual:** 141 label (77 negatif, 52 netral, 12 positif)

### 3. Pembaruan Notebook EDA 03 (`notebooks/03_eda_labeled_dataset.ipynb`)
- Re-eksekusi notebook EDA 03 dengan 6 grafik visualisasi lengkap yang diperbarui menggunakan 100% data Pure Human Gold Standard 954 komentar.

---

## 📂 Status Artefak Proyek Currently Active

| Artefak / File | Status | Keterangan |
|---|---|---|
| `brief/PRD.md` | Active | Reference PRD |
| `brief/AGENTS.md` | Active | Directives & Guardrails |
| `brief/project_state.md` | Updated | Snapshot Sesi 5 (22 Sep 2026 - Pure Gold Standard 70/30) |
| `specs/01_data_extraction.spec.md` | Approved | Data Extraction Spec |
| `specs/02_preprocessing_labeling.spec.md` | Approved | Preprocessing & Gemini API Hybrid Spec |
| `specs/03_model_training.spec.md` | Approved | Model Training Spec (IndoRoBERTa vs SahabatAI Discriminative Classifiers) |
| `data/raw/yt_comments_all_combined.csv` | Completed | 3.016 Unique Raw Comments |
| `data/interim/cleaned_comments_all.csv` | Completed | 3.016 Cleaned Comments |
| `data/interim/auto_labeled_comments_all.csv` | Completed | 3.016 Auto-Labeled Comments |
| `data/interim/human_audit_sample.xlsx` | Audited | 603 Sample Comments Audited by Human |
| `data/interim/human_audit_remaining.xlsx` | Audited | 2.413 Remaining Comments Audited by Human |
| `data/interim/master_labeled_comments.csv` | Completed | 3.016 Master Comments (Pure Human Gold Standard Applied) |
| `data/interim/valid_labeled_comments.csv` | Completed | 954 Pure Human Gold Standard Aspect Comments |
| `data/processed/train.csv` | Completed | 667 Train Comments (70%) |
| `data/processed/val.csv` | Completed | 287 Val Comments (30%) |
| `data/slang_dict.json` | Completed | Automotive Slang Dictionary |
| `src/preprocessing/text_cleaner.py` | Verified | Text Cleaning Module |
| `src/labeling/gemini_labeler.py` | Verified | Google Gemini API Structured JSON Labeler |
| `src/labeling/weak_supervision.py` | Verified | Hybrid Cascade Weak Supervision Engine |
| `src/labeling/eval_kappa.py` | Verified | Cohen's Kappa & Audit Generator |
| `src/labeling/dataset_splitter.py` | Verified | 70/30 Stratified Dataset Splitter |
| `notebooks/01_eda_raw_comments.ipynb` | Verified | EDA Raw Comments Notebook |
| `notebooks/02_eda_cleaned_comments.ipynb` | Verified | EDA Cleaned Comments Notebook |
| `notebooks/03_eda_labeled_dataset.ipynb` | Verified | EDA Pure Human Gold Standard Notebook (Executed) |
| `.env` & `.env.example` | Updated | Configured with YOUTUBE_API_KEY & GEMINI_API_KEY |
| `README.md` | Active | Contributor Overview |

---

## 🎯 Langkah Selanjutnya (Next Actions - Phase 3)

1. **Pembuatan Script PyTorch Multi-Head Classifier:**
   - Membangun `src/models/indoroberta_classifier.py` dan `src/models/sahabatai_classifier.py`.
2. **Pembuatan Kaggle Notebook 04:**
   - Menyusun `notebooks/04_kaggle_training_indoroberta_vs_sahabatai.ipynb` yang siap di-upload ke Kaggle Notebook dengan GPU T4.
3. **Pelatihan & Evaluasi Model 70/30:**
   - Melatih model pada `train.csv` (667 baris) dan menguji/memvalidasi pada `val.csv` (287 baris).


