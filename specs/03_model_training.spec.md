# SPEC-03: Model Training & Evaluation (IndoRoBERTa Classifier vs. SahabatAI Classifier)

- **Spec ID:** SPEC-03
- **Title:** Phase 3 — Multi-Aspect Sentiment Analysis Comparative Training & Evaluation (Discriminative Approach)
- **Status:** Approved / Finalized Specification
- **Author:** Antigravity AI & Wicaksono Hanif
- **Target Models:** IndoRoBERTa Classifier (Encoder 110M) vs. SahabatAI Classifier (Decoder LLM 8B)
- **Execution Environment:** Kaggle Notebooks (NVIDIA T4 GPU 16GB VRAM, $0 Cost)
- **Constraint:** Discriminative Classification Mode Only (No Naive Bayes / Random Forest / SVM, No Autoregressive Text Generation).

---

## 1. Executive Summary & Research Objective

Spesifikasi ini mengatur perancangan, pelatihan, dan evaluasi komparatif antara dua arsitektur kecerdasan buatan terdepan berbasis **Model Diskriminatif (*Discriminative Classification Models*)** untuk **Aspect-Based Sentiment Analysis (ABSA)** pada komentar YouTube mobil listrik (EV) China di Indonesia:

1. **Model 1 (Encoder Discriminative Classifier):** **IndoRoBERTa-base** (`indolem/indobert-base-uncased`, 110M Parameter) menggunakan arsitektur *Multi-Head Sequence Classification* (Full Fine-Tuning).
2. **Model 2 (Decoder LLM Discriminative Classifier):** **SahabatAI-Instruct-8B** (`SahabatAI/SahabatAI-Instruct-8B`, 8B Parameter) menggunakan arsitektur *Multi-Head Sequence Classification* berbasis **QLoRA / LoRA Partial Fine-Tuning**.

Pengujian dilakukan pada dataset **100% Gold Standard (Human Verified)** yang terbagi tanpa kebocoran data (*zero data leakage*):
- **Train Set (70%):** 844 komentar (`data/processed/train.csv`)
- **Validation Set (15%):** 181 komentar (`data/processed/val.csv`)
- **Test Set (15%):** 181 komentar (`data/processed/test.csv`)

---

## 2. Definisi Tugas & Target Output Klasifikasi

Setiap komentar teks diuji secara serentak terhadap **4 Aspek Otomotif EV**:
1. `infra` (Infrastruktur & Stasiun Pengisian/SPKLU)
2. `ekonomi` (Harga, Pajak, Subsidi, & Depresiasi)
3. `kualitas` (Kualitas Rakitan, Suspensi, Fitur ADAS, Interior/Eksterior)
4. `purnajual` (Dealer, Service, Suku Cadang, & Garansi)

### Klasifikasi per Aspek (4 Kelas Multi-Task):
- `0`: `None` (Aspek tidak didiskusikan dalam komentar)
- `1`: `positif` (Sentimen Positif)
- `2`: `netral` (Sentimen Netral)
- `3`: `negatif` (Sentimen Negatif)

---

## 3. Strategi Pelatihan & Jawaban Arsitektur (Dense vs. Partial Fine-Tune)

### ❓ Pertanyaan Arsitektur: Dense Layer Murni vs. Partial Fine-Tuning?
- **Dense Layer Only (Frozen Backbone):** Kurang disarankan karena representasi *backbone* tidak beradaptasi dengan kata gaul otomotif spesifik (`spklu`, `mobkas`, `boncos`, `adas`).
- **Rekomendasi Terpilih di SPEC-03:**
  - **IndoRoBERTa (110M):** **Full Fine-Tuning** (karena model 110M sangat ringan, hanya butuh VRAM ~2 GB dan selesai 5 epoch dalam < 2 menit di Kaggle T4 GPU).
  - **SahabatAI (8B):** **Partial Fine-Tuning via QLoRA** (membekukan backbone 8B dan melatih LoRA Adapter pada attention layers `q_proj`, `v_proj`, `k_proj`, `o_proj` + Classification Head). Ini melatih ~0.2% parameter, sangat efisien di Kaggle T4 GPU, dan memberikan akurasi jauh lebih tinggi daripada sekadar melatih dense layer murni.

---

## 4. Rincian Arsitektur Model

### 4.1 Model 1: IndoRoBERTa Multi-Head Classifier (`indolem/indobert-base-uncased`)

- **Model Backbone:** `indolem/indobert-base-uncased` (HuggingFace Transformers).
- **Arsitektur Classification Head:** Single shared transformer encoder $\rightarrow$ `[CLS]` pooler output $\rightarrow$ **4 Klasifikasi Head terpisah (Linear + Dropout 0.2)** untuk 4 aspek.
- **Fungsi Kerugian (Loss Function):** 
  $$\mathcal{L}_{total} = \mathcal{L}_{infra} + \mathcal{L}_{ekonomi} + \mathcal{L}_{kualitas} + \mathcal{L}_{purnajual}$$
  Menggunakan *Multi-Task Categorical Cross-Entropy Loss*.
- **Hyperparameter Training:**
  - **Environment:** Kaggle Notebook (NVIDIA T4 GPU)
  - **Epochs:** **5 Epochs**
  - **Batch Size:** 16
  - **Learning Rate:** $2 \times 10^{-5}$ dengan AdamW optimizer (`weight_decay = 0.01`)
  - **Scheduler:** Linear Warmup with Cosine Decay
  - **Random Seed:** 42

---

### 4.2 Model 2: SahabatAI-8B Multi-Head Classifier (`SahabatAI/SahabatAI-Instruct-8B`)

- **Model Backbone:** `SahabatAI/SahabatAI-Instruct-8B` (AutoModelForSequenceClassification / LlamaForSequenceClassification).
- **Metode Quantization & Fine-Tuning:** **4-bit NormalFloat (NF4) QLoRA Partial Fine-Tuning** via `bitsandbytes` & `peft`.
  - LoRA Rank ($r$): 16
  - LoRA Alpha ($\alpha$): 32
  - Target Modules: `['q_proj', 'k_proj', 'v_proj', 'o_proj']`
  - Classification Head: 4 Dense Heads terpisah pada token akhir `[EOS]`.
- **Hyperparameter Training:**
  - **Environment:** Kaggle Notebook (NVIDIA T4 GPU 16GB VRAM)
  - **Epochs:** **5 Epochs**
  - **Batch Size:** 8 (dengan Gradient Accumulation Steps = 2 $\rightarrow$ Effective Batch Size = 16)
  - **Learning Rate:** $2 \times 10^{-4}$ (standard LoRA learning rate)
  - **Random Seed:** 42

---

## 5. Metrik Evaluasi & Perbandingan

Kedua model dievaluasi secara adil pada **Test Set (181 data uji independen di `test.csv`)**:

1. **Metrik per Aspek (Infra, Ekonomi, Kualitas, Purnajual):**
   - **Accuracy**
   - **Precision (Macro & Weighted)**
   - **Recall (Macro & Weighted)**
   - **F1-Score (Macro & Weighted)**
   - **Matriks Kebingungan (Confusion Matrix 4x4 per Aspek)**

2. **Metrik Global Multi-Aspek:**
   - **Mean Macro F1-Score:** Rata-rata F1-Score Macro dari keempat aspek.
   - **Exact Match Ratio (Subset Accuracy):** Persentase komentar di mana keempat aspek tertebak 100% tepat.

---

## 6. Struktur Berkas & Skrip Kaggle Notebook

```
usb_2026/
├── specs/
│   └── 03_model_training.spec.md            # Dokumentasi Spesifikasi (File Ini)
├── src/
│   └── models/
│       ├── __init__.py
│       ├── indoroberta_classifier.py        # Module PyTorch IndoRoBERTa Multi-Head
│       ├── sahabatai_classifier.py          # Module PyTorch SahabatAI-8B QLoRA Multi-Head
│       └── metrics_evaluator.py             # Module Evaluasi Metrik & Confusion Matrix
├── notebooks/
│   └── 04_kaggle_training_indoroberta_vs_sahabatai.ipynb # Notebook Utama Siap Eksekusi di Kaggle
└── models/
    └── saved_checkpoints/                   # Output Bobot & Metrik Model
```

---

## 7. Langkah Implementasi (Next Actions)

1. **Membuat Modul PyTorch Local Classifier:** `src/models/indoroberta_classifier.py` dan `src/models/sahabatai_classifier.py`.
2. **Membuat Script Evaluasi Metrik:** `src/models/metrics_evaluator.py`.
3. **Penyusunan Notebook Kaggle:** `notebooks/04_kaggle_training_indoroberta_vs_sahabatai.ipynb` yang siap di-upload dan di-run di Kaggle Notebooks dengan GPU T4.
