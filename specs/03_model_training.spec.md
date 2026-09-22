# SPEC-03: Model Training & Evaluation (IndoRoBERTa-base 110M vs. XLM-RoBERTa-large 550M)

- **Spec ID:** SPEC-03
- **Title:** Phase 3 — Multi-Aspect Sentiment Analysis Comparative Training & Evaluation (Discriminative Encoder Benchmark & Optimization Strategy)
- **Status:** Approved / Finalized Specification (v4 - Customized Thresholds & Augmentation Strategy)
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
- **Focal Loss ($\gamma = 1.5$):** Menggunakan Multi-Aspect Focal Loss untuk menekan loss dari sampel mayoritas `None` hingga 95% dan memfokuskan pembaruan bobot gradien pada kelas sentimen aktif (`positif`, `netral`, `negatif`).
- **Smoothed Class Weights ($\sqrt{w}$):** Menggunakan akar kuadrat dari *balanced weights* untuk menyeimbangkan penalti tanpa menyebabkan *gradient spike*.

---

### 3.2 Decision Thresholding Terpisah per Aspek ($\theta_{\text{aspect}}$)
Berdasarkan hasil analisis confusion matrix iterasi 01 (di mana aspek `purnajual` dan `infra` memiliki tingkat kelangkaan tinggi >85% `None`), ambang batas keputusan $P(\text{None})$ diatur secara dinamis per aspek untuk meningkatkan sensitivitas deteksi kelas aktif:

- **Aspek `purnajual`:** $\theta_{\text{purnajual}} = 0.35$ (Meningkatkan sensitivitas deteksi sentimen purnajual yang langka)
- **Aspek `infra`:** $\theta_{\text{infra}} = 0.40$
- **Aspek `ekonomi`:** $\theta_{\text{ekonomi}} = 0.50$
- **Aspek `kualitas`:** $\theta_{\text{kualitas}} = 0.50$

---

### 3.3 Pembobotan Target Gradien Khusus (*Targeted Class Alpha*)
Untuk mengatasi masalah nol prediksi pada kelas `purnajual positif` (yang hanya memiliki 14 data latih), faktor pengali $\alpha$ pada Focal Loss disesuaikan secara khusus:
- $\alpha_{\text{purnajual, positif}} = 2.5 \times \alpha_{\text{default}}$
- $\alpha_{\text{infra, positif}} = 1.8 \times \alpha_{\text{default}}$

---

### 3.4 Strategi & Contoh Augmentasi Data Teks Kelas Minoritas

Untuk menambah variasi sintaksis dan memperkuat daya generatisasi model pada kelas ekstrem minoritas (`purnajual positif` dan `infra positif`) di data latih, diterapkan **Strategi Augmentasi Teks Kontekstual & Substitusi Sinonim (Synonym Replacement & Context-Preserving Paraphrasing)**.

> [!NOTE] 
> Augmentasi data **hanya diterapkan pada Train Set (667 data)**. Validation Set (287 data) **tetap 100% murni data manusia asli** tanpa augmentasi untuk menjamin validitas ilmiah evaluasi.

#### 📌 Contoh Augmentasi 1: Aspek `purnajual` (Sentimen: `positif`)
- **Teks Asli (Ground Truth):**
  > *"Pelayanan dealer BYD sangat memuaskan, klaim garansi baterai cepat dan tidak berbelit-belit."*
- **Variasi Augmentasi 1 (Substitusi Sinonim Otomotif):**
  > *"Servis dealer BYD sangat ramah, proses klaim garansi baterai cepat dan gampang banget."*
- **Variasi Augmentasi 2 (Parafrase Kontekstual):**
  > *"Layanan purnajual dealer BYD mantap sekali, klaim garansi baterainya cepat tanpa ribet."*

#### 📌 Contoh Augmentasi 2: Aspek `infra` (Sentimen: `positif`)
- **Teks Asli (Ground Truth):**
  > *"SPKLU di rest area tol Cipularang sudah banyak dan pengisian ultra fast charging cepat sekali."*
- **Variasi Augmentasi 1 (Substitusi Sinonim & Istilah Cas):**
  > *"Stasiun cas SPKLU rest area tol Cipularang makin melimpah dan ngecas daya fast charging kilat banget."*
- **Variasi Augmentasi 2 (Parafrase Kontekstual):**
  > *"Tempat ngecas SPKLU di jalan tol sudah tersebar banyak dan pengisian baterainya cepat sekali."*

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
