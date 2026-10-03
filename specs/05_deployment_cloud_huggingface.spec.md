# SPEC-05: Cloud Deployment & Hugging Face Hub Integration

- **Spec ID:** SPEC-05
- **Title:** Phase 5 — Cloud Deployment Specification & Hugging Face Hub Model Hosting Strategy
- **Status:** Approved Specification / Panduan Deployment Terintegrasi
- **Author:** Antigravity AI & Wicaksono Hanif
- **Target Cloud Platform:** Streamlit Community Cloud (`share.streamlit.io`)
- **Model Hosting Hub:** Hugging Face Hub (`huggingface.co`)
- **Primary App Entrypoint:** `app.py`

---

## 1. Executive Summary & System Architecture

Spesifikasi ini mengatur tata cara dan alur kerja deployment aplikasi **SentyBoard v1.0.0** (Dashboard Aspect-Based Sentiment Analysis EV) ke infrastruktur cloud publik secara gratis, stabil, dan optimal.

Mengingat ukuran bobot model fine-tuned **XLM-RoBERTa-large (550M parameter)** mencapai **~2.2 GB** (melebihi batas maksimal 100 MB file GitHub), strategi deployment memisahkan penyimpanan kode sumber (*source code*) dan bobot model AI (*model weights*).

```
 +----------------------------------+          +------------------------------------+
 |         GitHub Repository        |          |         Hugging Face Hub           |
 | (wicaksonohanif/aspect_based_SA) |          |  (username/xlm-roberta-ev-absa)    |
 | - app.py                         |          | - pytorch_model.bin (~2.2 GB)      |
 | - src/ deployment & data logic   |          | - config.json / tokenizer files    |
 | - requirements.txt               |          +-----------------+------------------+
 +----------------+-----------------+                            |
                  |                                              |
                  +-----------------------+----------------------+
                                          |
                                          v
                         +--------------------------------+
                         |    Streamlit Community Cloud   |
                         |  - Running SentyBoard v1.0.0   |
                         |  - Auto-downloads model on 1st |
                         |    inference run               |
                         +--------------------------------+
```

---

## 2. Checklist Langkah-Langkah Deployment

- [x] **Langkah 1:** Upload Bobot Model (`~2.2 GB`) ke Hugging Face Hub.
- [x] **Langkah 2:** Perbarui `src/deployment/model_loader.py` untuk membaca model dari Hugging Face Hub.
- [x] **Langkah 3:** Siapkan Berkas Konfigurasi Deployment (`requirements.txt` & `.streamlit/config.toml`).
- [x] **Langkah 4:** Commit & Push Perubahan Kode ke Repositori GitHub.
- [x] **Langkah 5:** Hubungkan Repositori GitHub ke **Streamlit Community Cloud** & Deploy.
- [x] **Langkah 6:** Verifikasi & Pengujian Pasca-Deployment.

---

## 3. Rincian Pelaksanaan Deployment

### 3.1 Langkah 1: Upload Model Weights ke Hugging Face Hub

1. **Buat Akun & Access Token di Hugging Face:**
   - Daftar di [huggingface.co](https://huggingface.co/).
   - Masuk ke **Settings ➔ Access Tokens ➔ Create new token** (Pilih Role: `Write`).

2. **Install `huggingface_hub` di Lingkungan Lokal:**
   ```bash
   pip install huggingface_hub
   ```

3. **Jalankan Skrip Upload Model (`upload_model_to_hf.py`):**
   ```python
   from huggingface_hub import HfApi, create_repo
   import os

   HF_USERNAME = "wicaksonohanif"
   REPO_NAME = "xlm-roberta-ev-absa"
   REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"
   LOCAL_MODEL_DIR = "outputs/iter-04"

   print(f"🚀 Membuat repositori di Hugging Face: {REPO_ID}")
   create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

   api = HfApi()
   print(f"📦 Mengunggah bobot model dari '{LOCAL_MODEL_DIR}'...")
   api.upload_folder(
       folder_path=LOCAL_MODEL_DIR,
       repo_id=REPO_ID,
       repo_type="model"
   )
   print(f"✅ SUKSES: Model berhasil diunggah ke https://huggingface.co/{REPO_ID}")
   ```

---

### 3.2 Langkah 2: Perbarui Engine Pemuat Model (`src/deployment/model_loader.py`)

Perbarui fungsi inisialisasi pada `src/deployment/model_loader.py` agar mengunduh bobot secara otomatis dari Hugging Face Hub jika belum ada di lokal:

```python
import os
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModel

HF_REPO_ID = "wicaksonohanif/xlm-roberta-ev-absa"
LOCAL_CACHE_DIR = Path("models/xlmroberta_absa")

class XLMInferenceEngine:
    def __init__(self, repo_id=HF_REPO_ID, local_dir=LOCAL_CACHE_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.repo_id = repo_id
        self.local_dir = Path(local_dir)
        self.is_weights_loaded = False
        
        self.load_model()

    def load_model(self):
        try:
            if self.local_dir.exists() and (self.local_dir / "pytorch_model.bin").exists():
                model_path = str(self.local_dir)
                print(f"📂 Memuat model dari folder lokal: {model_path}")
            else:
                print(f"🌐 Memuat/Mengunduh model dari Hugging Face Hub: {self.repo_id}")
                model_path = self.repo_id

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = XLMRoBERTaMultiHeadClassifier.from_pretrained(model_path).to(self.device)
            self.model.eval()
            self.is_weights_loaded = True
            print("✅ Model XLM-RoBERTa berhasil dimuat ke memory!")
        except Exception as e:
            print(f"⚠️ Gagal memuat bobot model: {e}")
            self.is_weights_loaded = False
```

---

### 3.3 Langkah 3: Berkas Konfigurasi Deployment

1. **`requirements.txt`:**
   ```text
   streamlit>=1.30.0
   pandas>=2.0.0
   numpy>=1.24.0
   torch>=2.0.0
   transformers>=4.35.0
   huggingface-hub>=0.20.0
   plotly>=5.18.0
   matplotlib>=3.7.0
   seaborn>=0.12.0
   wordcloud>=1.9.0
   openpyxl>=3.1.0
   ```

2. **`.streamlit/config.toml`:**
   ```toml
   [theme]
   primaryColor = "#1e3c72"
   backgroundColor = "#FFFFFF"
   secondaryBackgroundColor = "#F8F9FA"
   textColor = "#2C3E50"
   font = "sans serif"

   [server]
   headless = true
   enableCORS = false
   maxUploadSize = 200
   ```

3. **`.gitignore`:**
   ```gitignore
   models/
   outputs/**/*.zip
   *.bin
   *.pth
   *.pt
   ```

---

### 3.4 Langkah 4 & 5: Push ke GitHub & Deployment Cloud

1. Commit & Push perubahan kode ke repositori GitHub:
   ```bash
   git add .
   git commit -m "feat(deploy): finalize SPEC-05 cloud deployment specification"
   git push origin main
   ```
2. Buka [share.streamlit.io](https://share.streamlit.io/), pilih repositori `wicaksonohanif/aspect_based_SA_EV`, branch `main`, dan entrypoint `app.py`.

---

## 4. Pengoptimalan Memori & Kinerja Server Cloud

- **Resource Caching (`@st.cache_resource`):** Inisialisasi model di Streamlit wajib di-cache agar model besar tidak di-load berulang kali setiap kali halaman berubah.
- **Micro-Batch Inference:** Jalankan batch inferensi dengan ukuran batch `8` hingga `16` untuk menghemat penggunaan RAM pada instance server gratisan.
