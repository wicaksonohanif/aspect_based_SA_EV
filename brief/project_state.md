# project_state.md — Project Snapshot & Activity Log

- **Tanggal Sesi:** 22 September 2026
- **Sesi:** Sesi 5 — Integrasi Total Human Audit (Gold Standard 100%), Re-execution EDA Notebook 03, & Pembagian Dataset Final (Train 844, Val 181, Test 181)
- **Status Proyek:** Phase 1 Selesai 100% (3.016 Komentar Mentah Unik) & Phase 2 Selesai 100% (Gold Standard 1.206 Komentar Valid Ber-aspek di `data/processed/`)

---

## 📌 Ringkasan Aktivitas yang Telah Diselesaikan (22 September 2026)

### 1. Integrasi 100% Human Audit Gold Standard
- Pengguna telah menyelesaikan audit manual penuh pada berkas `data/interim/human_audit_sample.xlsx` (603 sampel) dan `data/interim/human_audit_remaining.xlsx` (2.413 sampel).
- **Hasil Penggabungan (Merge Gold Standard):**
  - **954 komentar ter-audit** (1.119 anotasi aspek manusia) berhasil digabungkan sebagai *Ground Truth Override* mutlak di atas label mesin.
  - Komentar tanpa isi sentimen pada kolom human diabaikan sesuai arahan.
  - Berkas master tersimpan di `data/interim/master_labeled_comments.csv` (3.016 baris) dan `data/interim/valid_labeled_comments.csv` (**1.206 komentar valid ber-aspek**).

### 2. Pembagian Dataset Final 70/15/15 (Phase 2 Finalization)
- Menjalankan `DatasetSplitter.split_dataset()` pada `data/interim/valid_labeled_comments.csv` dengan seed `random_state=42` tanpa kebocoran data.
- **Hasil Akhir Pembagian Dataset (`data/processed/`):**
  - **Train Set (70%):** 844 baris -> `data/processed/train.csv`
  - **Val Set (15%):** 181 baris -> `data/processed/val.csv`
  - **Test Set (15%):** 181 baris -> `data/processed/test.csv`
- **Distribusi Sentimen per Aspek (1.206 dataset valid Gold Standard):**
  - **Infra:** 123 label (58 netral, 33 negatif, 32 positif)
  - **Ekonomi:** 479 label (230 netral, 181 positif, 68 negatif)
  - **Kualitas:** 630 label (234 positif, 214 negatif, 182 netral)
  - **Purnajual:** 172 label (81 netral, 77 negatif, 14 positif)

### 3. Pembaruan Notebook EDA 03 (`notebooks/03_eda_labeled_dataset.ipynb`)
- Re-eksekusi notebook EDA 03 dengan 6 grafik visualisasi lengkap yang diperbarui menggunakan 100% data Gold Standard 1.206 komentar valid.

---

## 📂 Status Artefak Proyek Currently Active

| Artefak / File | Status | Keterangan |
|---|---|---|
| `brief/PRD.md` | Active | Reference PRD |
| `brief/AGENTS.md` | Active | Directives & Guardrails |
| `brief/project_state.md` | Updated | Snapshot Sesi 5 (22 Sep 2026 - Gold Standard Complete) |
| `specs/01_data_extraction.spec.md` | Approved | Data Extraction Spec |
| `specs/02_preprocessing_labeling.spec.md` | Approved | Preprocessing & Gemini API Hybrid Spec |
| `specs/03_model_training.spec.md` | Approved | Model Training Spec (IndoRoBERTa vs SahabatAI Discriminative Classifiers) |
| `data/raw/yt_comments_all_combined.csv` | Completed | 3.016 Unique Raw Comments |
| `data/interim/cleaned_comments_all.csv` | Completed | 3.016 Cleaned Comments |
| `data/interim/auto_labeled_comments_all.csv` | Completed | 3.016 Auto-Labeled Comments |
| `data/interim/human_audit_sample.xlsx` | Audited | 603 Sample Comments Audited by Human |
| `data/interim/human_audit_remaining.xlsx` | Audited | 2.413 Remaining Comments Audited by Human |
| `data/interim/master_labeled_comments.csv` | Completed | 3.016 Master Comments (100% Gold Standard Override Applied) |
| `data/interim/valid_labeled_comments.csv` | Completed | 1.206 Valid Aspect-Bearing Gold Standard Comments |
| `data/processed/train.csv` | Completed | 844 Train Comments (70%) |
| `data/processed/val.csv` | Completed | 181 Val Comments (15%) |
| `data/processed/test.csv` | Completed | 181 Test Comments (15%) |
| `data/slang_dict.json` | Completed | Automotive Slang Dictionary |
| `src/preprocessing/text_cleaner.py` | Verified | Text Cleaning Module |
| `src/labeling/gemini_labeler.py` | Verified | Google Gemini API Structured JSON Labeler |
| `src/labeling/weak_supervision.py` | Verified | Hybrid Cascade Weak Supervision Engine |
| `src/labeling/eval_kappa.py` | Verified | Cohen's Kappa & Audit Generator |
| `src/labeling/dataset_splitter.py` | Verified | 70/15/15 Dataset Splitter |
| `notebooks/01_eda_raw_comments.ipynb` | Verified | EDA Raw Comments Notebook |
| `notebooks/02_eda_cleaned_comments.ipynb` | Verified | EDA Cleaned Comments Notebook |
| `notebooks/03_eda_labeled_dataset.ipynb` | Verified | EDA Labeled Dataset Notebook (Executed with 6 Plots) |
| `.env` & `.env.example` | Updated | Configured with YOUTUBE_API_KEY & GEMINI_API_KEY |
| `README.md` | Active | Contributor Overview |

---

## 🎯 Langkah Selanjutnya (Next Actions - Phase 3)

1. **Penyusunan Spesifikasi Phase 3 (`specs/03_model_training.spec.md`):**
   - Merancang eksperimen arsitektur model baseline (TF-IDF + SVM/Logistic Regression) & Fine-Tuning Multi-Head IndoRoBERTa / IndoBERT.
2. **Implementasi Pipeline Multi-Head Training:**
   - Membangun script training dan evaluasi metrik (Accuracy, Precision, Recall, Macro/Micro F1-Score).
3. **Pelatihan & Evaluasi Baseline & Deep Learning Model:**
   - Melatih model pada `train.csv` (844 baris), memvalidasi pada `val.csv` (181 baris), dan menguji pada `test.csv` (181 baris).

