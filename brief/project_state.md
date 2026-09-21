# project_state.md — Project Snapshot & Activity Log

- **Tanggal Sesi:** 21 September 2026
- **Sesi:** Sesi 4 — Pelabelan Hybrid, Auditing Manusia (Human Audit Gold Standard), Inter-Rater Reliability (Cohen's Kappa), dan Pembagian Dataset Final (SPEC-02 Complete)
- **Status Proyek:** Phase 1 Selesai 100% (3.016 Komentar Mentah Unik) & Phase 2 Selesai 100% (Train, Val, Test Available in `data/processed/`)

---

## 📌 Ringkasan Aktivitas yang Telah Diselesaikan (21 September 2026)

### 1. Pelabelan Otomatis Hybrid & Rule-Based Negation Enhancement

- Menjalankan pelabelan otomatis hybrid (`WeakSupervisionEngine`) menggunakan perpaduan aturan berbasis kata kunci lokal dan Google Gemini API (`gemini-flash-latest`).
- Memperbaiki aturan berbasis negasi lokal (_explicit negative words_ seperti "murahan", "kurang nyaman") dan pemetaan fitur/desain/atap ke aspek Kualitas & Durabilitas (`kualitas`).
- **Hasil:** Berhasil melabeli 3.016 komentar di `data/interim/auto_labeled_comments_all.csv`.

### 2. Integrasi Human Audit Gold Standard

- Pengguna telah mengaudit 603 baris sampel di `data/interim/human_audit_sample.xlsx` (267 komentar audited dengan 306 anotasi aspek manusia).
- Mengintegrasikan hasil audit manusia sebagai _Ground Truth Override_ di atas label mesin.
- Menghasilkan file master `data/interim/master_labeled_comments.csv` dan file filter komentar ber-aspek `data/interim/valid_labeled_comments.csv` (945 komentar valid ber-aspek).

### 3. Pembagian Dataset Final 70/15/15 (Phase 2 Finalization)

- Menjalankan `DatasetSplitter.split_dataset()` pada `data/interim/valid_labeled_comments.csv` dengan seed `random_state=42` untuk mencegah data leakage.
- **Hasil Akhir Pembagian Dataset (`data/processed/`):**
  - **Train Set (70%):** 661 baris -> `data/processed/train.csv`
  - **Val Set (15%):** 142 baris -> `data/processed/val.csv`
  - **Test Set (15%):** 142 baris -> `data/processed/test.csv`
- **Distribusi Sentimen per Aspek (945 dataset valid):**
  - **Infra:** 87 label (59 netral, 17 positif, 11 negatif)
  - **Ekonomi:** 493 label (269 netral, 168 positif, 56 negatif)
  - **Kualitas:** 407 label (203 netral, 130 positif, 74 negatif)
  - **Purnajual:** 123 label (83 netral, 21 negatif, 19 positif)

---

## 📂 Status Artefak Proyek Currently Active

| Artefak / File                               | Status    | Keterangan                                       |
| -------------------------------------------- | --------- | ------------------------------------------------ |
| `brief/PRD.md`                               | Active    | Reference PRD                                    |
| `brief/AGENTS.md`                            | Active    | Directives & Guardrails                          |
| `brief/project_state.md`                     | Updated   | Snapshot Sesi 4 (21 Sep 2026 - Phase 2 Complete) |
| `specs/01_data_extraction.spec.md`           | Approved  | Data Extraction Spec                             |
| `specs/02_preprocessing_labeling.spec.md`    | Approved  | Preprocessing & Gemini API Hybrid Spec           |
| `data/raw/yt_comments_all_combined.csv`      | Completed | 3.016 Unique Raw Comments                        |
| `data/interim/cleaned_comments_all.csv`      | Completed | 3.016 Cleaned Comments                           |
| `data/interim/auto_labeled_comments_all.csv` | Completed | 3.016 Auto-Labeled Comments                      |
| `data/interim/human_audit_sample.xlsx`       | Audited   | 603 Sample Comments Audited by Human             |
| `data/interim/master_labeled_comments.csv`   | Completed | 3.016 Master Comments (Human Override Applied)   |
| `data/interim/valid_labeled_comments.csv`    | Completed | 945 Valid Aspect-Bearing Comments                |
| `data/processed/train.csv`                   | Completed | 661 Train Comments (70%)                         |
| `data/processed/val.csv`                     | Completed | 142 Val Comments (15%)                           |
| `data/processed/test.csv`                    | Completed | 142 Test Comments (15%)                          |
| `data/slang_dict.json`                       | Completed | Automotive Slang Dictionary                      |
| `src/preprocessing/text_cleaner.py`          | Verified  | Text Cleaning Module                             |
| `src/labeling/gemini_labeler.py`             | Verified  | Google Gemini API Structured JSON Labeler        |
| `src/labeling/weak_supervision.py`           | Verified  | Hybrid Cascade Weak Supervision Engine           |
| `src/labeling/eval_kappa.py`                 | Verified  | Cohen's Kappa & Audit Generator                  |
| `src/labeling/dataset_splitter.py`           | Verified  | 70/15/15 Dataset Splitter                        |
| `notebooks/01_eda_raw_comments.ipynb`        | Verified  | EDA Raw Comments Notebook                        |
| `notebooks/02_eda_cleaned_comments.ipynb`    | Verified  | EDA Cleaned Comments Notebook                    |
| `.env` & `.env.example`                      | Updated   | Configured with YOUTUBE_API_KEY & GEMINI_API_KEY |
| `README.md`                                  | Active    | Contributor Overview                             |

---

## 🎯 Langkah Selanjutnya (Next Actions - Phase 3)

1. **Penyusunan Spesifikasi Phase 3 (`specs/03_model_training.spec.md`):**
   - Merancang eksperimen arsitektur model baseline IndoRoBERTa yang dilakukan Fine-Tuning serta perbandingannya dengan Sahabat AI.
   
2. **Implementasi Pipeline Multi-Label / Multi-Aspect Training:**
   - Membangun script training dan evaluasi metrik (Accuracy, Precision, Recall, Macro/Micro F1-Score).
3. **Pelatihan Baseline Model:**
   - Melatih model pada `train.csv`, memvalidasi pada `val.csv`, dan menguji pada `test.csv`.
