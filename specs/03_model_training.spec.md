# SPEC-03: Model Training & Evaluation (5-Fold Stratified Cross-Validation Benchmark)

- **Spec ID:** SPEC-03
- **Title:** Phase 3 — Multi-Aspect Sentiment Analysis 5-Fold Stratified Cross-Validation Comparative Benchmark & Optimization
- **Status:** Approved / Finalized Specification (v5 - 5-Fold CV & Expanded GELU MLP Head)
- **Author:** Antigravity AI & Wicaksono Hanif
- **Target Models:** IndoRoBERTa-base (110M Monolingual Baseline) vs. XLM-RoBERTa-large (550M Multilingual Target)
- **Execution Environment:** Kaggle Notebooks (NVIDIA T4 GPU 16GB VRAM, $0 Cost)
- **Constraint:** 100% Pure Human Gold Standard (954 Komentar), TANPA Augmentasi Data, Discriminative Encoder Mode Only.

---

## 1. Executive Summary & Research Objective

Spesifikasi ini mengatur perancangan, pelatihan, dan evaluasi ilmiah komparatif berbasis **5-Fold Stratified Cross-Validation (5-Fold CV)** antara dua arsitektur kecerdasan buatan berbasis **Model Diskriminatif (*Discriminative Encoder Models*)** untuk **Aspect-Based Sentiment Analysis (ABSA)** pada komentar YouTube mobil listrik (EV) China di Indonesia:

1. **Model 1 (Monolingual Base Baseline):** **IndoRoBERTa-base** (`indolem/indobert-base-uncased`, 110M Parameter) menggunakan arsitektur *Concatenated CLS + Mean Pooling* dengan **2-Layer GELU MLP Multi-Head Classifier** (Full Fine-Tuning 20 Epochs, Evaluasi 5-Fold CV).
2. **Model 2 (Multilingual Large Primary Target):** **XLM-RoBERTa-large** (`xlm-roberta-large`, 550M Parameter) menggunakan arsitektur *Concatenated CLS + Mean Pooling* dengan **Expanded 2-Layer GELU MLP Multi-Head Classifier + LayerNorm (`2048 -> 512 -> 4`)** (Full Fine-Tuning 20 Epochs, Evaluasi 5-Fold CV & 5-Fold Ensemble Voting).

Pengujian dilakukan pada dataset **100% Pure Human Gold Standard (954 Komentar Ter-audit)** tanpa augmentasi buatan, diproyeksikan ke 5 fold yang terbagi secara *stratified multi-aspect* tanpa kebocoran data (*zero data leakage*, `random_seed=42`).

---

## 2. Metodologi Evaluasi 5-Fold Stratified Cross-Validation

Untuk mengatasi keterbatasan jumlah sampel langka pada single split serta mengurangi volatilitas evaluasi, digunakan **5-Fold Stratified Cross-Validation**:

```
[ Total 954 Komentar Pure Human Gold Standard ]
                        │
                        ▼ (Multi-Aspect Composite Stratification, Seed=42)
  ├── Fold 1: Train (763) │ Val (191) ──► Model Checkpoint 1
  ├── Fold 2: Train (763) │ Val (191) ──► Model Checkpoint 2
  ├── Fold 3: Train (763) │ Val (191) ──► Model Checkpoint 3
  ├── Fold 4: Train (763) │ Val (191) ──► Model Checkpoint 4
  └── Fold 5: Train (763) │ Val (191) ──► Model Checkpoint 5
                        │
                        ▼
    [ Final Evaluation: Mean ± Std Dev across 5 Folds ]
    [ 5-Fold Ensemble Model for Streamlit Deployment ]
```

---

## 3. Strategi Pelatihan & Penanganan Class Imbalance

### 3.1 Multi-Aspect Focal Loss ($\gamma = 1.5$) & Smoothed Class Weights
- **Focal Loss ($\gamma = 1.5$):** Menekan loss dari sampel mayoritas `None` hingga 95% dan memfokuskan pembaharuan bobot gradien pada kelas sentimen aktif (`positif`, `netral`, `negatif`).
- **Smoothed Class Weights ($\sqrt{w}$):** Menggunakan pemangkasan akar kuadrat dari *balanced weights* untuk menyeimbangkan penalti tanpa menyebabkan *gradient spike*.

---

### 3.2 Decision Thresholding Terpisah per Aspek ($\theta_{\text{aspect}}$)
Menggunakan ambang batas keputusan $P(\text{None})$ dinamis per aspek untuk memaksimalkan sensitivitas deteksi kelas aktif:
- **Aspek `purnajual`:** $\theta_{\text{purnajual}} = 0.35$ (Meningkatkan sensitivitas deteksi sentimen purnajual yang langka)
- **Aspek `infra`:** $\theta_{\text{infra}} = 0.40$
- **Aspek `ekonomi`:** $\theta_{\text{ekonomi}} = 0.50$
- **Aspek `kualitas`:** $\theta_{\text{kualitas}} = 0.50$

---

## 4. Rincian Arsitektur Model

### 4.1 Model 1: IndoRoBERTa Multi-Head Classifier (`indolem/indobert-base-uncased`)
- **Model Backbone:** `indolem/indobert-base-uncased` (110M Monolingual Encoder).
- **Pooling Layer:** Concatenation `[CLS]` + *Mean Pooling* (1536 dimensions).
- **Classification Head:** 4 GELU MLP Heads (`Linear(1536 -> 256) -> GELU() -> Dropout(0.1) -> Linear(256 -> 4)`).
- **Loss Function:** Multi-Aspect Focal Loss ($\gamma = 1.5$).
- **Hyperparameter:** 20 Epochs, Batch Size 16, LR $2 \times 10^{-5}$, Cosine Warmup 10%.

---

### 4.2 Model 2: XLM-RoBERTa Large Expanded Multi-Head Classifier (`xlm-roberta-large`)
- **Model Backbone:** `xlm-roberta-large` (550M Multilingual Encoder, 24 Layers, 1024 Hidden Dim).
- **Pooling Layer:** Concatenation `[CLS]` + *Mean Pooling* (2048 dimensions).
- **Expanded Classification Head:** 4 GELU MLP Heads dengan Layer Normalization (`Linear(2048 -> 512) -> LayerNorm() -> GELU() -> Dropout(0.1) -> Linear(512 -> 4)`).
- **Loss Function:** Multi-Aspect Focal Loss ($\gamma = 1.5$).
- **Hyperparameter:** 20 Epochs, Micro Batch 8, Grad Accum 2 (Effective Batch 16), LR $1.5 \times 10^{-5}$, Cosine Warmup 10%.

---

## 5. Tampilan Log Pelatihan per Epoch

Pada setiap epoch pelatihan di Kaggle Notebook, sistem **wajib menampilkan log terperinci** yang mencakup:
1. `Epoch [X/20]`
2. `Train Loss`
3. `Val F1-Macro (Active Sentiments)`
4. `Val F1-Macro (All Classes)`
5. `Val Accuracy`

*Contoh Format Display Output:*
```text
Epoch  5/20 | Train Loss: 0.1245 | Val F1 (Active): 0.5621 | Val F1 (All): 0.6540 | Val Acc: 0.8620
Epoch 10/20 | Train Loss: 0.0832 | Val F1 (Active): 0.6145 | Val F1 (All): 0.7012 | Val Acc: 0.8845
```

---

## 6. Metrik Evaluasi Akhir & Ensemble Deployment

Hasil akhir 5-Fold Cross-Validation dilaporkan dalam bentuk **Mean $\pm$ Standard Deviation**:
- **Mean Macro F1 (Active Sentiments):** Rata-rata F1 Macro dari kelas `positif`, `netral`, `negatif`.
- **Mean Macro F1 (All Classes):** Rata-rata F1 Macro dari 4 kelas (termasuk `None`).
- **Mean Accuracy & Exact Match Ratio.**

### 📦 Checkpoint Deployment Streamlit App
Ke-5 checkpoint model dari 5 fold disiapkan untuk **5-Fold Ensemble Voting (Probability Averaging)** saat inferensi pada aplikasi web Streamlit (`app.py`).
