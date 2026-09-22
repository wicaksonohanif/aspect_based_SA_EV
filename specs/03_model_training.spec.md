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

Kedua model dievaluasi secara adil pada **Validation Set (287 data uji/validasi independen di `val.csv`)**:

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

## 6. Proteksi Keamanan VRAM & OOM Prevention (Kaggle Free Tier GPU T4 16GB)

Untuk menjamin eksekusi di **Kaggle Notebooks versi gratis** (Batas VRAM T4 GPU = 15,9 GB) berjalan 100% lancar tanpa *Out-Of-Memory (OOM)* error, diterapkan 6 lapis proteksi memori:

1. **4-Bit NF4 Quantization (`bitsandbytes`):**
   - Memuat bobot SahabatAI-8B dalam presisi 4-bit NormalFloat, memotong konsumsi VRAM awal dari **~16 GB menjadi hanya ~5.5 GB**.
2. **Gradient Checkpointing:**
   - Mengaktifkan `model.gradient_checkpointing_enable()` pada SahabatAI-8B untuk menghemat ~60% VRAM aktivasi saat *backward pass*.
3. **Micro-Batching & Gradient Accumulation:**
   - **SahabatAI-8B:** `per_device_train_batch_size = 2` (Micro-Batch) + `gradient_accumulation_steps = 8` $\rightarrow$ *Effective Batch Size = 16*. Ini menjaga puncak penggunaan VRAM < 8 GB.
   - **IndoRoBERTa:** `per_device_train_batch_size = 16`.
4. **Mixed Precision Training (FP16/BF16):**
   - Menggunakan `fp16=True` (atau `bf16=True`) pada PyTorch / HuggingFace Trainer untuk memotong penggunaan memori aktivasi hingga 50%.
5. **Pembersihan Memori Seketika (Explicit CUDA Garbage Collection):**
   - Di antara eksekusi Model 1 (IndoRoBERTa) dan Model 2 (SahabatAI-8B), dilakukan penghapusan objek secara tegas (`del model`, `del trainer`) diikuti oleh `gc.collect()` dan `torch.cuda.empty_cache()` untuk mengosongkan 100% VRAM.
6. **Pembatasan Panjang Teks (*Max Sequence Length* = 128 Token):**
   - Berdasarkan hasil EDA 03 (median komentar = 10 kata, 99th percentile < 50 kata), `max_length` dibatasi hingga **128 token**. Hal ini mencegah pengalokasian memori berlebih untuk 512 token.

---

## 7. Struktur Berkas & Skrip Kaggle Notebook

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

## 8. Panduan Berkas Input (Upload) & Output Deployment (Download) Kaggle

### 📤 8.1 Berkas Input yang Harus Diunggah ke Kaggle (*Input Dataset*)
Unggah 2 berkas CSV dari folder `data/processed/` lokal Anda ke fitur **Kaggle Datasets**:
1. `train.csv` (667 baris - Data Latih Pure Human Gold Standard)
2. `val.csv` (287 baris - Data Validasi/Uji Pure Human Gold Standard)

---

### 📥 8.2 Berkas Output yang Harus Didownload dari Kaggle untuk Deployment Aplikasi (*Output Artifacts*)

Setelah Notebook 04 selesai di-run di Kaggle (`Run All`), notebook akan otomatis mengompresi dan menyimpan berkas berikut pada direktori `/kaggle/working/` untuk Anda unduh:

#### 1. Folder Checkpoint IndoRoBERTa Classifier (`indoroberta_absa_model.zip` ~440 MB):
- `config.json` (Konfigurasi arsitektur multi-head & label mapping: `0: None, 1: positif, 2: netral, 3: negatif`)
- `model.safetensors` atau `pytorch_model.bin` (Bobot model fine-tuned IndoRoBERTa)
- `tokenizer_config.json`, `vocab.txt`, `special_tokens_map.json` (Tokenizer files)
*Fungsi:* Digunakan langsung untuk deployment aplikasi web (Streamlit / FastAPI / Flask) secara lokal maupun cloud.

#### 2. Folder Adapter LoRA SahabatAI-8B (`sahabatai_absa_lora.zip` ~40 MB):
- `adapter_config.json` (Konfigurasi LoRA adapter)
- `adapter_model.safetensors` (Bobot adapter LoRA ~40 MB)
*Fungsi:* Digunakan jika ingin me-load adapter SahabatAI-8B via HuggingFace PEFT / vLLM.

#### 3. Berkas Evaluasi & Laporan Skripsi:
- `model_comparison_metrics.csv` & `model_comparison_metrics.json` (Tabel lengkap metrik Accuracy, Precision, Recall, Macro F1 per aspek)
- `confusion_matrices_indoroberta.png` & `confusion_matrices_sahabatai.png` (Visualisasi matriks kebingungan 4x4 untuk Bab 4 Laporan)

---

## 9. Langkah Implementasi (Next Actions)

1. **Membuat Modul PyTorch Local Classifier:** `src/models/indoroberta_classifier.py` dan `src/models/sahabatai_classifier.py`.
2. **Membuat Script Evaluasi Metrik:** `src/models/metrics_evaluator.py`.
3. **Penyusunan Notebook Kaggle:** `notebooks/04_kaggle_training_indoroberta_vs_sahabatai.ipynb` yang siap di-upload dan di-run di Kaggle Notebooks dengan GPU T4.


