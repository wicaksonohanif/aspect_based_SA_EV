# Technical Specification: Preprocessing & Weak Supervision Labeling (Google Gemini API & Hybrid Engine)

- **Spec ID:** SPEC-02
- **Feature Title:** Pembersihan Teks Informal & Pelabelan Weak Supervision (Google Gemini API Pro + Rule-Based Hybrid)
- **Target Components:** `src/preprocessing/` & `src/labeling/`
- **Related PRD Section:** FR-2 (Preprocessing & Weak Supervision Module)
- **Status:** Approved / In Implementation

---

## 1. Overview & Objectives

Dokumen spesifikasi ini mengatur modul pembersihan teks (*preprocessing*) dan pelabelan otomatis berbasis **Hybrid Weak Supervision** (Rule-Based Filter + **Google Gemini API**) untuk komentar media sosial YouTube mengenai kendaraan listrik (*Electric Vehicle* / EV) pabrikan China di Indonesia.

### Key Mandates & Scope Boundaries:
1. **Normalisasi Bahasa Informal:** Mengubah teks warganet yang sarat dengan *slang*, singkatan, *typo*, emoji, dan istilah otomotif informal menjadi bentuk teks terstandarisasi.
2. **Taksonomi Aspek & Sentimen Multidimensi:** Mengklasifikasikan opini ke dalam 4 aspek strategis (Infrastruktur, Ekonomi, Kualitas, Purna Jual) dan 3 polaritas sentimen (Positif, Netral, Negatif).
3. **Seamless Hybrid Cascade Labeling Engine:**
   - **Step 1 (Rule-Based Lokal):** Penyaringan instan berbasis kamus kata kunci & regex di komputer lokal (0 API calls, ultra-cepat untuk teks eksplisit).
   - **Step 2 (Google Gemini API Cloud):** Pengiriman teks ke Google Gemini API (`gemini-1.5-flash` / `gemini-1.5-pro`) menggunakan fitur *Structured JSON Output* untuk menangani kalimat ambigu, implisit, sarkasme, dan analisis konteks mendalam.
4. **Single-Evaluator Cohen's Kappa Verification ($\kappa$):** Penyiapan sampel 20–30% untuk diaudit oleh 1 orang anotator manusia (Auditor Standar Emas) guna menghitung statistik koefisien **Cohen's Kappa ($\kappa$)** antara Label Mesin ($M$) vs Label Manusia ($H$), dengan target nilai $\kappa \ge 0,61$.

---

## 2. Pipeline Pembersihan Teks (`src/preprocessing/`)

### 2.1 Alur Pembersihan Teks (`src/preprocessing/text_cleaner.py`)
Setiap string komentar mentah (`text_original`) diproses melalui tahapan bertahap (*sequential cleaning pipeline*):

```
text_original ──> [1. Lowercasing] ──> [2. URL & Mention Removal] ──> [3. Character Repeat Reduction]
              ──> [4. Slang Normalization] ──> [5. Noise Clean] ──> text_cleaned
```

### 2.2 Aturan Cleaning & Regex
| Tahap | Deskripsi | Aturan / Regex | Contoh Sebelum $\rightarrow$ Sesudah |
|---|---|---|---|
| **1. Case Folding** | Konversi ke huruf kecil | `.lower()` | `"Mobil BYD Bagus!"` $\rightarrow$ `"mobil byd bagus!"` |
| **2. Remove URL** | Hapus tautan web | `r"https?://\S+\|www\.\S+"` | `"cek https://ev.id keren"` $\rightarrow$ `"cek keren"` |
| **3. Remove Mention** | Hapus sebutan akun | `r"@\w+"` | `"setuju @otomotif_id"` $\rightarrow$ `"setuju"` |
| **4. Character Repeat** | Reduksi huruf berulang $>2$ | `r"(.)\1{2,}"` $\rightarrow$ `\1\1` | `"baguuuusss"` $\rightarrow$ `"baguus"` |
| **5. Slang Normalization** | Substitusi kamus slang | Lookup Dictionary | `"gak pake spklu mobkas"` $\rightarrow$ `"tidak pakai stasiun pengisian kendaraan listrik umum mobil bekas"` |
| **6. Whitespace Clean** | Hapus spasi ganda & newline | `r"\s+"` $\rightarrow$ `" "` | `"keren \n\n banget"` $\rightarrow$ `"keren banget"` |

### 2.3 Kamus Normalisasi Slang Otomotif Indonesia (`data/slang_dict.json`)
Kamus penyamaan istilah disiapkan dalam bentuk JSON:

```json
{
  "gak": "tidak",
  "ga": "tidak",
  "pake": "pakai",
  "bgt": "banget",
  "mobkas": "mobil bekas",
  "spklu": "stasiun pengisian kendaraan listrik umum",
  "ngecas": "mengisi daya",
  "cas": "isi daya",
  "batre": "baterai",
  "resale": "nilai jual kembali",
  "fast charging": "pengisian daya cepat"
}
```

---

## 3. Taksonomi Aspek & Sentimen

### 3.1 4 Aspek Strategis EV China
1. **`Infrastruktur & Jangkauan` (`infra`)**
   - *Kata Kunci / Konsep:* SPKLU, charging station, kabel cas, jarak tempuh (km), baterai habis di jalan, charging lama, PLN.
2. **`Ekonomi & Finansial` (`ekonomi`)**
   - *Kata Kunci / Konsep:* Harga baru, harga bekas, depresiasi, nilai jual kembali (resale value), pajak EV, efisiensi biaya listrik vs BBM, garansi baterai.
3. **`Kualitas & Durabilitas` (`kualitas`)**
   - *Kata Kunci / Konsep:* Durabilitas baterai (Blade Battery, LFP), suspensi, kualitas interior, build quality, fitur ADAS, rem, software error, akselerasi.
4. **`Purna Jual & Ekosistem` (`purnajual`)**
   - *Kata Kunci / Konsep:* Dealer resmi, ketersediaan sparepart, indent suku cadang, layanan sales, mekanik bengkel, komitmen brand China.

### 3.2 Polaritas Sentimen
- **`Positif` (+1):** Memuji, merekomendasikan, atau menyatakan kepuasan.
- **`Netral` (0):** Pertanyaan deskriptif, fakta teknis tanpa opini, atau perbandingan seimbang.
- **`Negatif` (-1):** Keluhan, keraguan, kritik keras, atau pengalaman buruk.
- **`None` (Null):** Aspek tersebut tidak dibahas dalam komentar.

---

## 4. Hybrid Cascade Labeling Engine & Google Gemini API (`src/labeling/`)

### 4.1 Arsitektur Cascade (Rule-Based Filter $\rightarrow$ Gemini API Cloud)

```
                            [ Teks Komentar Input ]
                                       │
                                       ▼
                       [ Tahap 1: Rule-Based (Lokal) ]
             Mencocokkan kamus kata kunci 4 aspek & pola sentimen jelas
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
      (Komentar Jelas & Eksplisit)               (Komentar Ambigu / Sarkasme)
   Langsung diberi label di lokal                 Di-pass ke Gemini API Pro/Flash
   (Kecepatan: Instant / 0 API call)             (Structured JSON Response Mode)
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       ▼
                     [ Data Terlabel Komplit & Valid ]
```

### 4.2 Integrasi Google Gemini API (`src/labeling/gemini_labeler.py`)
Kredensial API Key dibaca otomatis dari berkas `.env` (`GEMINI_API_KEY`).

```python
import os
import json
import google.generativeai as genai

class GeminiAspectLabeler:
    def __init__(self, api_key: str | None = None, model_name: str = "gemini-1.5-flash"):
        """Inisialisasi Google Gemini API Client dengan Structured JSON Output Config."""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY tidak ditemukan di .env atau parameter!")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"response_mime_type": "application/json"}
        )

    def label_comment(self, text: str) -> dict:
        """
        Mengirim teks ambigu/kompleks ke Gemini API untuk klasifikasi aspek & sentimen JSON.
        """
        prompt = f"""
        Kamu adalah pakar Aspect-Based Sentiment Analysis otomotif Indonesia.
        Analisis komentar berikut dan berikan nilai polaritas ('Positif', 'Netral', 'Negatif', atau null) untuk 4 aspek:
        1. infra (Infrastruktur & Jangkauan)
        2. ekonomi (Ekonomi & Finansial)
        3. kualitas (Kualitas & Durabilitas)
        4. purnajual (Purna Jual & Ekosistem)

        Komentar: "{text}"

        Output wajib berupa format JSON persis seperti berikut:
        {{
          "infra_sentiment": "Positif" | "Netral" | "Negatif" | null,
          "ekonomi_sentiment": "Positif" | "Netral" | "Negatif" | null,
          "kualitas_sentiment": "Positif" | "Netral" | "Negatif" | null,
          "purnajual_sentiment": "Positif" | "Netral" | "Negatif" | null
        }}
        """
        response = self.model.generate_content(prompt)
        return json.loads(response.text)
```

### 4.3 Output Format Dataset Terlabel
Hasil pelabelan otomatis disimpan ke `data/interim/auto_labeled_comments_<timestamp>.csv` dengan kolom:
- `comment_id`
- `text_cleaned`
- `label_infra` (`Positif` / `Netral` / `Negatif` / `None`)
- `label_ekonomi` (`Positif` / `Netral` / `Negatif` / `None`)
- `label_kualitas` (`Positif` / `Netral` / `Negatif` / `None`)
- `label_purnajual` (`Positif` / `Netral` / `Negatif` / `None`)
- `label_source` (`Rule-Based` / `Gemini-API`)

---

## 5. Single-Evaluator Cohen's Kappa Verification (`src/labeling/eval_kappa.py`)

### 5.1 Stratified Human Audit (20–30%)
- Skrip mengambil sampel acak terstratifikasi sebesar **20% hingga 30%** dari data terlabel mesin (`label_source`).
- Sampel diekspor ke file audit CSV: `data/interim/human_audit_sample.csv`.
- Anotator Manusia (1 orang Auditor) mengecek dan memasukkan label verifikasi pada kolom `human_label_<aspek>`.

### 5.2 Formulasi Cohen's Kappa ($\kappa$) Antara Mesin vs 1 Auditor Manusia
Perhitungan kesepakatan statistik antara **Anotator 1 (Label Mesin Hybrid)** vs **Anotator 2 (Auditor Manusia)**:

$$\kappa = \frac{P_o - P_e}{1 - P_e}$$

- $P_o$: Proporsi kesepakatan teramati (*observed agreement*).
- $P_e$: Proporsi kesepakatan secara kebetulan (*expected chance agreement*).

### 5.3 Ambang Batas Kelayakan KPI PRD
- **Target Minimal PRD:** $\kappa \ge 0,61$ (*Substantial Agreement* berdasarkan skala Landis & Koch).
- **Tindakan Korektif:** Jika $\kappa < 0,61$, lakukan penyesuaian pada kamus kata kunci atau prompt Gemini API sebelum melangkah ke tahap pelatihan model IndoRoBERTa vs Sahabat-AI (`specs/03_model_comparison.spec.md`).

---

## 6. Pembagian Dataset Final (`data/processed/`)

Setelah label tervalidasi ($\kappa \ge 0,61$), dataset dibagi menjadi 3 subset terpisah dengan penguncian *random seed* (`seed = 42`):

```
Total Dataset Terlabel (100%)
├── Train Set (70%)  ──> data/processed/train.csv
├── Val Set   (15%)  ──> data/processed/val.csv
└── Test Set  (15%)  ──> data/processed/test.csv  (Isolasi Ketat / Zero Leakage)
```

---

## 7. Acceptance Criteria & Verification Plan

| No | Kriteria Pengujian | Cara Verifikasi | Expectation / Target |
|---|---|---|---|
| **AC-1** | Normalisasi Slang & Noise | Jalankan `text_cleaner.py` | URL, mention, & kata gaul terkonversi menjadi teks baku. |
| **AC-2** | Hybrid Cascade Routing | Jalankan `weak_supervision.py` | Teks eksplisit dilabeli via Rule-Based (0 API call); teks ambigu dilabeli via Gemini API. |
| **AC-3** | Gemini Structured JSON Output | Audit hasil respon Gemini API | Output 100% berupa JSON valid sesuai skema tanpa markdown corrupt. |
| **AC-4** | Stratified Human Audit Generator | Jalankan `eval_kappa.py --sample-ratio 0.2` | Menghasilkan file audit sampel persis 20% dari total data untuk 1 auditor manusia. |
| **AC-5** | Kalkulasi Cohen's Kappa | Hitung kesepakatan Mesin vs Auditor Manusia | Nilai $\kappa$ terhitung dan tercatat ($\kappa \ge 0,61$). |
