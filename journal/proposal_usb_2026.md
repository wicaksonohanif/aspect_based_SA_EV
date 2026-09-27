# PROPOSAL PENELITIAN & INOVASI DATA MINING (USB 2026)

## 1. Identitas Proposal
- **Judul Proyek:** Aspect-Based Sentiment Analysis (ABSA) Multi-Task Mobil Listrik China di Indonesia Menggunakan Arsitektur Discriminative Encoder Berbasis 5-Fold Stratified Cross-Validation
- **Tema Kompetisi:** Ekosistem Digital Cerdas untuk Masa Depan Indonesia yang Inklusif dan Berkelanjutan
- **Sub-Tema:** Rekayasa Data & Artificial Intelligence Sektor Otomotif Berkelanjutan
- **Target Deadline:** 5 Oktober 2026
- **Lokasi Berkas:** `journal/proposal_usb_2026.md`

---

## 2. Ringkasan Eksekutif
Penetrasi kendaraan listrik (EV) pabrikan China membuka opsi mobilitas hijau terjangkau di Indonesia. Adopsi massal terhambat keraguan publik seputar durabilitas baterai, depresiasi nilai jual kembali, ketersediaan SPKLU, dan keandalan layanan purnajual. Opini warganet di media sosial YouTube menyimpan data persepsi riil bernilai tinggi namun terhambat struktur bahasa informal dan ketimpangan kelas (*class imbalance*) ekstrem. Proyek ini menghadirkan sistem penambangan data end-to-end: ekstraksi beretika dengan penyamaran identitas (SHA-256), kurasi dataset *100% Pure Human Gold Standard* (954 data), pemodelan multi-task 4 aspek memakai *Discriminative Encoder* (IndoRoBERTa-base vs XLM-RoBERTa-large) dengan *Multi-Aspect Focal Loss* dan *Aspect-Specific Thresholding*, serta penyusunan dasbor interaktif berbasis Streamlit tanpa biaya operasional.

---

## 3. Latar Belakang & Rumusan Masalah

### 3.1 Latar Belakang
1. Transisi energi nasional mendorong penetrasi merek EV China (BYD, Wuling, Chery, Neta).
2. Diskusi publik digital di kanal YouTube menjadi cerminan adopsi konsumen di lapangan.
3. Riset sentimen konvensional gagal menangkap sentimen per komponen teknis spesifik (hanya positif/negatif global).

### 3.2 Rumusan Masalah
1. Bagaimana mengotomatisasi ekstraksi 4 aspek spesifik otomotif (`infra`, `ekonomi`, `kualitas`, `purnajual`) dan polaritas sentimennya dari teks komentar informal?
2. Bagaimana mengatasi ketimpangan distribusi kelas dominan `None` (>80%) terhadap sentimen minoritas aktif?
3. Sejauh mana keunggulan performa model *Multilingual Large Encoder* (XLM-RoBERTa-large 550M) dibandingkan *Monolingual Base Encoder* (IndoRoBERTa-base 110M) dalam skema validasi 5-Fold Stratified Cross-Validation?

---

## 4. Tujuan & Kebaruan Penelitian (*Novelty*)

### 4.1 Tujuan Penelitian
1. Membangun korpus dataset sentimen berbasis 4 aspek otomotif EV pertama beranotasi 100% audit manusia di Indonesia.
2. Membandingkan secara empiris kinerja arsitektur diskriminatif IndoRoBERTa dan XLM-RoBERTa dalam format multi-task GELU MLP heads.
3. Mengembangkan dasbor analitik publik interaktif untuk rekomendasi pembuat kebijakan dan pelaku industri.

### 4.2 Kebaruan Ilmiah (*Novelty*)
1. **Multi-Aspect Multi-Task Architecture:** 4 pasang klasifikasi independen per teks dengan penanganan ambang batas dinamis per aspek ($\theta_{\text{aspect}}$).
2. **Loss Khusus Imbalance:** Integrasi *Multi-Aspect Focal Loss* ($\gamma = 1.5$) dipadukan pembobotan kelas akar kuadrat (*Smoothed Class Weights*).
3. **Data Integrity & Zero Leakage:** Ekstraksi API resmi dengan kepatuhan PII (SHA-256 anonymization) dan 5-Fold Stratified Cross-Validation tanpa kebocoran data.

---

## 5. Metodologi Penelitian

```
[YouTube Data API v3] -> PII Anonymization (SHA-256)
        |
        v
[Preprocessing & Normalization] -> Slang Dict Otomotif
        |
        v
[Hybrid Weak Supervision] -> Rule-Based + Gemini API -> 100% Human Audit (954 Sampel)
        |
        v
[5-Fold Stratified CV Benchmark]
  ├── Model A: IndoRoBERTa-base (110M) + Concatenated CLS & Mean Pooling + 4 GELU MLP
  └── Model B: XLM-RoBERTa-large (550M) + Concatenated CLS & Mean Pooling + 4 GELU MLP
        |
        v
[Metrics Evaluation] -> Mean Macro F1, Active Sentiment F1, Exact Match Ratio
        |
        v
[Interactive Deployment] -> Streamlit Web Dashboard + Hugging Face Model Hub
```

### 5.1 Taksonomi 4 Aspek & Polaritas Sentimen
1. **Infrastruktur (`infra`):** Kesiapan SPKLU, pengecasan rumahan, durasi isi daya, jarak tempuh.
2. **Ekonomi (`ekonomi`):** Harga jual, subsidi pemerintah, pajak, depresiasi nilai jual bekas.
3. **Kualitas (`kualitas`):** Kualitas rakitan, suspensi, baterai LFP/Blade, fitur ADAS, interior.
4. **Purnajual (`purnajual`):** Layanan bengkel resmi, inden suku cadang, garansi baterai/mesin.
- **Polaritas:** `0` (None), `1` (Positif), `2` (Netral), `3` (Negatif).

---

## 6. Hasil Eksperimen Awal (Baseline Benchmark)

Pengujian pada data validasi terstandardisasi:

| Metrik Evaluasi | IndoRoBERTa-base (110M) | XLM-RoBERTa-large (550M) | Kenaikan Performa |
|---|---|---|---|
| **Overall Mean Accuracy** | 80.75% | **85.45%** | +4.70% |
| **Mean Macro F1 (All-Class)** | 58.93% | **62.92%** | +3.99% |
| **Mean Macro F1 (Active Sentiment)** | 48.57% | **52.81%** | +4.24% |
| **Exact Match Ratio (Subset Accuracy)** | 44.95% | **55.40%** | +10.45% |

Temuan: XLM-RoBERTa unggul signifikan pada aspek Kualitas (65.36% F1) dan Ekonomi (61.38% F1), sementara IndoRoBERTa tetap kompetitif pada aspek Purnajual berkat kosakata slang lokal.

---

## 7. Dampak Praktis & Rekomendasi Solusi
1. **Regulator (Kementerian ESDM / Kemenhub):** Pemetaan kebutuhan prioritas SPKLU di jalur non-tol berdasarkan sebaran sentimen negatif aspek infrastruktur.
2. **Pelaku Industri (APM EV):** Identifikasi titik lemah persepsi publik pada ketersediaan suku cadang dan depresiasi harga untuk mitigasi strategi garansi purnajual.
3. **Masyarakat:** Transparansi informasi kelebihan dan kelemahan adopsi EV dari ribuan pengalaman pemilik riil.

---

## 8. Rencana Implementasi & Anggaran

### 8.1 Alokasi Waktu Kerja
| Tahap | Aktivitas | Durasi |
|---|---|---|
| Tahap 1 | Finalisasi 5-Fold Cross-Validation & Error Analysis | 3 Hari |
| Tahap 2 | Pembuatan Dasbor Interaktif Streamlit (`app.py`) | 2 Hari |
| Tahap 3 | Unggah Model ke Hugging Face Hub | 1 Hari |
| Tahap 4 | Penyusunan Laporan Akhir & Salindia Presentasi Lomba | 3 Hari |

### 8.2 Rencana Anggaran Biaya (RAB)
- **Komputasi GPU:** Kaggle Notebooks (NVIDIA T4 16GB) = Rp0
- **Hosting Aplikasi:** Streamlit Community Cloud = Rp0
- **Penyimpanan Model:** Hugging Face Model Hub = Rp0
- **Pustaka Data:** YouTube Data API v3 (Free Tier Quota) = Rp0
- **Total Biaya Operasional:** **Rp0 (Efisiensi 100%)**
