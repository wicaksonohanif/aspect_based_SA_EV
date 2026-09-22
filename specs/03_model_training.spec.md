# SPEC-03: Model Training & Evaluation (IndoRoBERTa-base 110M vs. XLM-RoBERTa-large 550M)

- **Spec ID:** SPEC-03
- **Title:** Phase 3 — Multi-Aspect Sentiment Analysis Comparative Training & Evaluation (Discriminative Encoder Benchmark)
- **Status:** Approved / Finalized Specification (v3 - XLM-RoBERTa Large Benchmark)
- **Author:** Antigravity AI & Wicaksono Hanif
- **Target Models:** IndoRoBERTa Classifier (110M Monolingual) vs. XLM-RoBERTa-large Classifier (550M Multilingual)
- **Execution Environment:** Kaggle Notebooks (NVIDIA T4 GPU 16GB VRAM, $0 Cost)
- **Constraint:** Discriminative Encoder Classification Mode Only (Full Fine-Tuning 100%, No 4-bit Quantization Loss, No Generative Text Generation).

---

## 1. Executive Summary & Research Objective

Spesifikasi ini mengatur perancangan, pelatihan, dan evaluasi komparatif antara dua arsitektur kecerdasan buatan berbasis **Model Diskriminatif (*Discriminative Encoder Models*)** untuk **Aspect-Based Sentiment Analysis (ABSA)** pada komentar YouTube mobil listrik (EV) China di Indonesia:

1. **Model 1 (Monolingual Base Encoder):** **IndoRoBERTa-base** (`indolem/indobert-base-uncased`, 110M Parameter) menggunakan arsitektur *Concatenated CLS + Mean Pooling* dengan **2-Layer GELU MLP Multi-Head Classifier** (Full Fine-Tuning, 20 Epochs, LR $2 \times 10^{-5}$).
2. **Model 2 (Multilingual Large Encoder):** **XLM-RoBERTa-large** (`xlm-roberta-large`, 550M Parameter) menggunakan arsitektur *Concatenated CLS + Mean Pooling* dengan **2-Layer GELU MLP Multi-Head Classifier** (Full Fine-Tuning, 20 Epochs, LR $1.5 \times 10^{-5}$).

Pengujian dilakukan pada dataset **100% Pure Human Gold Standard (Murni Audit Manusia)** yang terbagi tanpa kebocoran data (*zero data leakage*):
- **Train Set (70%):** 667 komentar (`data/processed/train.csv`)
- **Validation Set (30%):** 287 komentar (`data/processed/val.csv`)

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

## 3. Strategi Pelatihan & Penanganan Class Imbalance

### 3.1 Penanganan Ketimpangan Kelas Mayoritas `None`
- **Focal Loss ($\gamma = 1.5$):** Menggunakan Multi-Aspect Focal Loss untuk menekan loss dari sampel mayoritas `None` dan menfokuskan pembaruan bobot gradien pada kelas sentimen aktif (`positif`, `netral`, `negatif`).
- **Smoothed Class Weights:** Menggunakan pemangkasan akar kuadrat dari *balanced weights* ($\sqrt{w}$) untuk menyeimbangkan penalti tanpa menyebabkan *gradient spike*.
- **Decision Thresholding ($\theta_{\text{none}} = 0.50$):** Menggunakan ambang batas probabilitas $P(\text{None}) > 0.50$ sebelum memutuskan kelas `None`. Jika probabilitas `None` $\le 0.50$, model dipaksa memilih kelas sentimen aktif tertinggi dari `[positif, netral, negatif]`.

---

## 4. Rincian Arsitektur Model

### 4.1 Model 1: IndoRoBERTa Multi-Head Classifier (`indolem/indobert-base-uncased`)

- **Model Backbone:** `indolem/indobert-base-uncased` (110M Monolingual Encoder).
- **Pooling Layer:** Concatenation dari `[CLS]` token representation + *Mean Pooling* (1536 hidden dimensions).
- **Arsitektur Classification Head:** 4 Multi-Task GELU MLP Heads terpisah (`Linear(1536 -> 256) -> GELU() -> Dropout(0.1) -> Linear(256 -> 4)`).
- **Fungsi Kerugian (Loss Function):** Multi-Aspect Focal Loss ($\gamma = 1.5$).
- **Hyperparameter Training:**
  - **Environment:** Kaggle Notebook (NVIDIA T4 GPU)
  - **Epochs:** **20 Epochs**
  - **Batch Size:** 16
  - **Learning Rate:** $2 \times 10^{-5}$ dengan AdamW optimizer (`weight_decay = 0.01`)
  - **Scheduler:** Linear Warmup (10%) dengan Cosine Decay
  - **Random Seed:** 42

---

### 4.2 Model 2: XLM-RoBERTa Large Multi-Head Classifier (`xlm-roberta-large`)

- **Model Backbone:** `xlm-roberta-large` (550M Multilingual Encoder, 24 Layers, 1024 Hidden Dimension).
- **Pooling Layer:** Concatenation dari `[CLS]` token representation + *Mean Pooling* (2048 hidden dimensions).
- **Arsitektur Classification Head:** 4 Multi-Task GELU MLP Heads terpisah (`Linear(2048 -> 256) -> GELU() -> Dropout(0.1) -> Linear(256 -> 4)`).
- **Fungsi Kerugian (Loss Function):** Multi-Aspect Focal Loss ($\gamma = 1.5$).
- **Hyperparameter Training:**
  - **Environment:** Kaggle Notebook (NVIDIA T4 GPU 16GB VRAM)
  - **Epochs:** **20 Epochs**
  - **Batch Size:** 8 (dengan Gradient Accumulation Steps = 2 $\rightarrow$ Effective Batch Size = 16)
  - **Learning Rate:** $1.5 \times 10^{-5}$ (Standard Large Model LR)
  - **Scheduler:** Linear Warmup (10%) dengan Cosine Decay
  - **Random Seed:** 42

---

## 5. Metrik Evaluasi & Perbandingan

Kedua model dievaluasi pada **Validation Set (287 data uji/validasi independen di `val.csv`)**:

1. **Metrik per Aspek (Infra, Ekonomi, Kualitas, Purnajual):**
   - **Accuracy**
   - **Macro F1-Score (All-Class & Active-Sentiment)**
   - **Weighted F1-Score**
   - **Matriks Kebingungan (Confusion Matrix 4x4 per Aspek)**

2. **Metrik Global Multi-Aspek:**
   - **Mean Macro F1 (Active Sentiment):** Rata-rata F1 Macro dari kelas sentimen aktif (`positif`, `netral`, `negatif`).
   - **Mean Macro F1 (All-Class):** Rata-rata F1 Macro dari keempat kelas (termasuk `None`).
   - **Exact Match Ratio (Subset Accuracy):** Persentase komentar di mana keempat aspek tertebak 100% tepat.

---

## 6. Proteksi Keamanan VRAM (Kaggle Free Tier GPU T4 16GB)

1. **Full Fine-Tuning Tanpa Quantization Loss:** XLM-RoBERTa Large (550M) hanya membutuhkan VRAM ~3.5 GB di GPU T4.
2. **Micro-Batching & Gradient Accumulation:** XLM-RoBERTa `batch_size = 8`, `grad_accum = 2`.
3. **Pembersihan Memori Seketika:** Explicit CUDA Garbage Collection `del model`, `gc.collect()`, `torch.cuda.empty_cache()`.
4. **Pembatasan Panjang Teks (*Max Sequence Length* = 128 Token).**

---

## 7. Berkas Deployment Output (Download Kaggle)

- `indoroberta_absa_model.zip` (~440 MB)
- `xlmroberta_absa_model.zip` (~2.1 GB)
- `model_comparison_metrics.csv` & `model_comparison_metrics.json`
- `confusion_matrices_indoroberta.png` & `confusion_matrices_xlmroberta.png`
