# SPEC-03: Model Training & Evaluation (IndoRoBERTa Classifier vs. SahabatAI Classifier)

- **Spec ID:** SPEC-03
- **Title:** Phase 3 — Multi-Aspect Sentiment Analysis Comparative Training & Evaluation (Discriminative Approach)
- **Status:** Approved / Finalized Specification (Optimized v2)
- **Author:** Antigravity AI & Wicaksono Hanif
- **Target Models:** IndoRoBERTa Classifier (Encoder 110M) vs. SahabatAI Classifier (Decoder LLM 8B)
- **Execution Environment:** Kaggle Notebooks (NVIDIA T4 GPU 16GB VRAM, $0 Cost)
- **Constraint:** Discriminative Classification Mode Only (No Naive Bayes / Random Forest / SVM, No Autoregressive Text Generation).

---

## 1. Executive Summary & Research Objective

Spesifikasi ini mengatur perancangan, pelatihan, dan evaluasi komparatif antara dua arsitektur kecerdasan buatan terdepan berbasis **Model Diskriminatif (*Discriminative Classification Models*)** untuk **Aspect-Based Sentiment Analysis (ABSA)** pada komentar YouTube mobil listrik (EV) China di Indonesia:

1. **Model 1 (Encoder Discriminative Classifier):** **IndoRoBERTa-base** (`indolem/indobert-base-uncased`, 110M Parameter) menggunakan arsitektur *Concatenated CLS + Mean Pooling* dengan **2-Layer GELU MLP Multi-Head Classifier** (Full Fine-Tuning, 20 Epochs).
2. **Model 2 (Decoder LLM Discriminative Classifier):** **SahabatAI-Instruct-8B** (`Sahabat-AI/SahabatAI-Instruct-8B`, 8B Parameter) menggunakan arsitektur *Multi-Head Sequence Classification* berbasis **4-bit NF4 QLoRA Partial Fine-Tuning** (5 Epochs).

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
- **Focal Loss ($\gamma = 2.0$):** Mengganti standard Cross-Entropy dengan Multi-Aspect Focal Loss untuk menekan loss dari sampel mayoritas `None` hingga 95% dan menfokuskan pembaruan bobot gradien pada kelas sentimen aktif (`positif`, `netral`, `negatif`).
- **Smoothed Class Weights:** Menggunakan pemangkasan akar kuadrat dari *balanced weights* ($\sqrt{w}$) untuk menyeimbangkan penalti tanpa menyebabkan *gradient spike*.
- **Dynamic Decision Thresholding ($\theta_{\text{none}} = 0.40$):** Menggunakan ambang batas probabilitas $P(\text{None}) > 0.40$ sebelum memutuskan kelas `None`. Jika probabilitas `None` $\le 0.40$, model dipaksa memilih kelas sentimen aktif tertinggi dari `[positif, netral, negatif]`.

---

## 4. Rincian Arsitektur Model

### 4.1 Model 1: IndoRoBERTa Multi-Head Classifier (`indolem/indobert-base-uncased`)

- **Model Backbone:** `indolem/indobert-base-uncased` (HuggingFace Transformers).
- **Pooling Layer:** Concatenation dari `[CLS]` token representation + *Mean Pooling* (1536 hidden dimensions).
- **Arsitektur Classification Head:** 4 Multi-Task GELU MLP Heads terpisah (`Linear(1536 -> 256) -> GELU() -> Dropout(0.2) -> Linear(256 -> 4)`).
- **Fungsi Kerugian (Loss Function):** Multi-Aspect Focal Loss ($\gamma = 2.0$).
- **Hyperparameter Training:**
  - **Environment:** Kaggle Notebook (NVIDIA T4 GPU)
  - **Epochs:** **20 Epochs**
  - **Batch Size:** 16
  - **Learning Rate:** $3 \times 10^{-5}$ dengan AdamW optimizer (`weight_decay = 0.01`)
  - **Scheduler:** Linear Warmup dengan Cosine Decay
  - **Random Seed:** 42

---

### 4.2 Model 2: SahabatAI-8B Multi-Head Classifier (`Sahabat-AI/SahabatAI-Instruct-8B`)

- **Model Backbone:** `Sahabat-AI/SahabatAI-Instruct-8B` (Fallback: `GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct`).
- **Metode Quantization & Fine-Tuning:** **4-bit NormalFloat (NF4) QLoRA Partial Fine-Tuning** via `bitsandbytes` & `peft`.
  - LoRA Rank ($r$): 16
  - LoRA Alpha ($\alpha$): 32
  - Target Modules: `['q_proj', 'k_proj', 'v_proj', 'o_proj']`
  - Classification Head: 4 Multi-Task GELU MLP Heads terpisah (`Linear(4096 -> 256) -> GELU() -> Dropout(0.1) -> Linear(256 -> 4)`).
- **Hyperparameter Training:**
  - **Environment:** Kaggle Notebook (NVIDIA T4 GPU 16GB VRAM)
  - **Epochs:** **5 Epochs**
  - **Batch Size:** 2 (dengan Gradient Accumulation Steps = 8 $\rightarrow$ Effective Batch Size = 16)
  - **Learning Rate:** $2 \times 10^{-4}$ (standard LoRA learning rate)
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

## 6. Proteksi Keamanan VRAM & OOM Prevention (Kaggle Free Tier GPU T4 16GB)

1. **4-Bit NF4 Quantization (`bitsandbytes`)**
2. **Gradient Checkpointing**
3. **Micro-Batching & Gradient Accumulation** (SahabatAI Effective Batch Size = 16)
4. **Mixed Precision Training (FP16/BF16)**
5. **Pembersihan Memori Seketika (Explicit CUDA Garbage Collection)**
6. **Pembatasan Panjang Teks (*Max Sequence Length* = 128 Token)**

---

## 7. Struktur Berkas

```
usb_2026/
├── specs/
│   └── 03_model_training.spec.md            # Dokumentasi Spesifikasi (File Ini)
├── src/
│   └── models/
│       ├── __init__.py
│       ├── indoroberta_classifier.py        # Module PyTorch IndoRoBERTa (Focal Loss & GELU MLP)
│       ├── sahabatai_classifier.py          # Module PyTorch SahabatAI-8B QLoRA (Focal Loss & GELU MLP)
│       └── metrics_evaluator.py             # Module Evaluasi Metrik & Dynamic Thresholding
├── notebooks/
│   └── 04_kaggle_training_indoroberta_vs_sahabatai.ipynb # Notebook Utama Siap Eksekusi di Kaggle
└── models/
    └── saved_checkpoints/                   # Output Bobot & Metrik Model
```
