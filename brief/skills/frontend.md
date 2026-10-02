# 🎨 Skill & Design System: Frontend UI/UX Streamlit Web Application

Dokumen ini adalah acuan **Design System & Template Preferensi Tampilan Frontend** untuk pembuatan aplikasi web **Aspect-Based Sentiment Analysis (ABSA) Komentar Mobil Listrik (EV)** berbasis **Streamlit**.
---

## 🛠️ 1. Tema & Palet Warna (Color Palette & Dark/Light Mode)

- **Mode Tampilan Utama:** `Dark Mode`
- **Warna Utama (Primary Accent):** `#00D2FF` 
- **Warna Latar Belakang (Background):** `#0F172A` 
- **Warna Latar Kartu (Card Background):** `#1E293B` 
- **Warna Teks Utama (Text Color):** `#FFFFFF` 

### 🚥 Palet Warna Sentimen per Aspek:
- 🟢 **Sentimen Positif:** `#10B981` *(Emerald Green)*
- 🟡 **Sentimen Netral:** `#F59E0B` *(Amber Yellow)*
- 🔴 **Sentimen Negatif:** `#EF4444` *(Coral Red)*
- ⚪ **Aspek Non-aktif / None:** `#6B7280` *(Muted Gray)*

---

## 🔤 2. Tipografi & Font (Typography)

- **Font Utama Judul (Headings):** `'Inter'`,
- **Font Isi Teks (Body Text):** `Inter`
- **Gaya Judul Dashboard:** *Gradiens Teks (Electric Gradient)* / *Solid Minimalis*

---

## 📐 3. Tata Letak & Layout Halaman (Page Layout)

- **Streamlit Layout Mode:** `st.set_page_config(layout="wide")`
- **Sidebar Status:** `Expanded` *(Terbuka secara default)*
- **Header Banner:** 
  - [x] Tampilkan Hero Banner dengan Judul Proyek & Sub-judul Deskriptif.
  - [x] Tampilkan Logo / Icon EV Otomotif (`⚡`, `🚗`, `🔋`).
- **Struktur Halaman Utama (Multi-Tab):**
  - **Tab 1:** `📊 Dasbor Eksekutif` (KPI Cards + Grouped Charts + Donut Proportion)
  - **Tab 2:** `🎯 Analisis Mendalam Aspek` (Sub-tab: Infra, Ekonomi, Kualitas, Purnajual)
  - **Tab 3:** `⭐ Galeri Komentar Terpopuler` (Top Liked Comments Gallery)
  - **Tab 4:** `📥 Unduh Dataset Terlabel` (Interactive Dataframe & Export CSV)

---

## 🃏 4. Gaya Komponen & Kartu Informasi (Cards & Component Styling)

- **Kartu Metric KPI (Metric Cards):**
  - Menggunakan Custom CSS Card dengan `border-radius: 12px`, subtle border, dan shadow.
  - Menggunakan warna indikator naik/turun yang kontras.
- **Kartu Komentar Terpopuler (Top Liked Comment Cards):**
  - Desain berupa *Hero Quote Bubble* atau *Card Box*.
  - Menampilkan badge jumlah *like* bertuliskan `👍 XXX Likes` dengan warna emas/kuning kontras.
  - Badge tag berwarna untuk nama aspek (`Kualitas`, `Ekonomi`, dll) dan warna sentimen.

---

## 📈 5. Estetika Grafik & Chart (Plotly / Visualizations)

- **Library Chart Utama:** `Plotly` *(Interaktif dengan hover tooltip)*
- **Tema Grafik:** Custom Dark Theme menyesuaikan palet warna di atas.
- **Transparansi Background Chart:** `Transparent` *(Agar menyatu dengan background Streamlit)*.

---

## 🎨 6. Custom CSS Injection (Kode CSS Kustom Preferensi)

```css
/* Contoh Custom CSS yang akan di-inject ke Streamlit */
div[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #00D2FF !important;
}

div.stButton > button {
    border-radius: 8px !important;
    background: linear-gradient(135deg, #00D2FF 0%, #0072FF 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
}
```

---

## 📝 7. Catatan & Keinginan Khusus Pengguna (Custom User Notes)

Contoh referensi desain streamlit
```
import time
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.patches as mpatches
from PIL import Image
from pathlib import Path
from torchvision.models import mobilenet_v3_small
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-VEOLI | Pneumonia Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- CONFIG ---------- */
/* Import Font Inter dari Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stMarkdown h3, [data-testid="stMarkdown"] h3 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
}
            
.stMarkdown p, 
.stMarkdown ul, 
.stMarkdown ol, 
.stMarkdown li,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] ul,
[data-testid="stMarkdownContainer"] ol,
[data-testid="stMarkdownContainer"] li {
    font-family: 'Inter', sans-serif !important;
}

/* Background Utama Halaman */
.stApp {
    background-color: #F1F1F1;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(90deg, #E81A1D 0%, #F7947E 100%);
}
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
    font-size: 15px;
    font-weight: 600;
}
            
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
    display: none !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
    padding: 10px 15px;
    border-radius: 25px;
    border: 2px solid transparent; 
    margin-bottom: 5px;
    transition: all 0.3s ease;
    cursor: pointer;
    width: 100%;
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
    background-color: rgba(255, 255, 255, 0.1);
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    border: 2px solid #FFFFFF !important;
    background-color: rgba(255, 255, 255, 0.15) !important;
}
            
.sidebar-logo-card {
    background-color: #FFFFFF;
    border-radius: 20px;
    padding: 10px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
}
.sidebar-logo-card h2 {
    color: #E81A1D !important;
    margin: 0;
    font-size: 25px;
    font-weight: 800;
}

/* ========== HOME ========== */        
/* ---------- Title & Subtitle ---------- */
.hero-container {
    background: linear-gradient(90deg, #FFDBDB 0%, #FFFFFF 100%);
    border-radius: 30px;
    padding: 20px 20px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
    text-align: center;
    margin-bottom: 40px;
}            

.main-title {
    font-size: 50px;
    font-weight: 800;
    background: linear-gradient(90deg, #E81A1D 0%, #F7947E 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 5px;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 10px;
}
.sub-title {
    font-size: 18px;
    color: #707070;
    text-align: center;
    margin-bottom: 20px;
    font-weight: 600;
}

/* ---------- Badges / Pills ---------- */
.badge-container {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-bottom: 30px;
}
.badge {
    background: linear-gradient(90deg, #E81A1D 0%, #F7947E 100%);
    color: white;
    padding: 8px 20px;
    border-radius: 20px;
    font-size: 15px;
    font-weight: 600;
}

/* ---------- Feature Cards ---------- */
.feature-card {
    background: linear-gradient(360deg, #FFDBDB 0%, #FFFFFF 100%);
    border-radius: 15px;
    padding: 25px 20px;
    text-align: center;
    height: 100%;
    box-shadow: 0 4px 10px rgba(0,0,0,0.03);
}
.feature-card h3 {
    color: #E81A1D;
    font-size: 18px;
    font-weight: 800;
    margin-top: 15px;
    margin-bottom: 10px;
}
.feature-card p {
    font-size: 15px;
    color: #707070;
    line-height: 1.5;
    margin: 0;
}
.feature-icon {
    font-size: 50px;
}

/* ---------- Disclaimer Card ---------- */
.disclaimer-card {
    background-color: #FFFFFF;
    border-radius: 10px;
    padding: 20px;
    margin-top: 40px;
    display: flex;
    align-items: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    border-left: 8px solid #E81A1D; /* Garis merah di sebelah kiri */
}
.disclaimer-content {
    text-align: center;
    width: 100%;
}
.disclaimer-title {
    color: #E81A1D;
    font-weight: 800;
    font-size: 18px;
    margin-bottom: 8px;
}
.disclaimer-text {
    font-size: 15px;
    color: #707070;
}

/* ---------- Footer ---------- */
.custom-footer {
    text-align: center;
    font-size: 12px;
    color: #707070;
    line-height: 1.6;
    margin-top: 50px !important;
    margin-bottom: 10px !important;
    padding-bottom: 10px !important;
}
.custom-footer span {
    color: #61afef; 
    font-weight: 600;
}
            
.block-container {
    padding-bottom: 1rem !important;
}

/* ========== HEADER GLOBAL ========== */  
/* ---------- Header ---------- */
.header-card {
    background: linear-gradient(90deg, #FFDBDB 0%, #FFFFFF 100%);
    border-radius: 30px;
    padding: 20px 30px;
    margin-bottom: 30px;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    justify-content: center;
}
            
.header-card h1 {
    margin: 0;
    font-size: 32px;
    font-weight: 800;
    color: #E81A1D;
    font-family: 'Inter', sans-serif;
}
            
/* ========== PREDICTION ========== */  
/* ---------- Upload ---------- */
.upload-box {
    border: 2px dashed #E81A1D;
    border-radius: 15px;
    padding: 30px;
    background-color: #FFFFFF;
    text-align: center;
}
            
[data-testid="stFileUploader"] {
    border: 2px dashed #E81A1D !important;
    border-radius: 15px !important;
    background-color: #FFFFFF !important;
    padding: 25px 20px !important;
    text-align: center;
    transition: all 0.3s ease;
}
            
[data-testid="stFileUploader"]:hover {
    background-color: #FFF5F5 !important;
}

[data-testid="stFileUploaderDropzone"] {
    border: none !important;
    background-color: transparent !important;
    padding: 10px !important;
}
[data-testid="stFileUploaderDropzone"] svg {
    display: none; 
}

[data-testid="stFileUploader"] label {
    display: flex !important;
    justify-content: center !important;
    width: 100%;
}
            
[data-testid="stFileUploader"] label p {
    font-size: 25px !important;
    font-weight: 600 !important;
    color: #E81A1D !important;
    margin-bottom: 10px;
    font-family: 'Inter', sans-serif;
}

/* ---------- Pred Button ---------- */
div.stButton > button:first-child {
    background: linear-gradient(90deg, #E81A1D 0%, #F7947E 100%) !important;
    border: none !important;
    padding: 10px 25px !important;
    border-radius: 50px !important;
    width: 100% !important;
    transition: transform 0.2s ease !important;
}

div.stButton > button:first-child:hover {
    transform: scale(1.02) !important;
}

div.stButton > button:first-child p {
    color: white !important;
    font-weight: 600 !important;
    font-size: 18px !important;
    margin: 0 !important; 
}

/* ---------- File Path Bar ---------- */
.file-path-bar {
    background-color: #ffffff;
    padding: 10px 20px;
    border-radius: 10px;
    font-size: 15px;
    color: #707070;
    margin: 20px 0;
    font-family: inter, sans-serif;
}

/* ---------- Result Metrics Bar ---------- */
.result-bar {
    background: linear-gradient(90deg, #FFDBDB 0%, #FFFFFF 100%);
    border-radius: 10px;
    padding: 25px;
    display: flex;
    justify-content: space-around;
    align-items: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    margin-top: 20px;
}
.result-item {
    text-align: center;
}
.result-value {
    font-size: 25px;
    font-weight: 800;
    color: #1E293B;
}
.result-label {
    font-size: 15px;
    color: #707070;
    font-weight: 600;
}

/* ========== METRIK MODEL ========== */    
/* Styling Metric Cards */
.metric-grid { 
    display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 30px; font-family: inter, sans-serif;
}
.metric-box { 
    flex: 1; min-width: 150px; background: linear-gradient(90deg, #FFDBDB 0%, #FFFFFF 100%); border-radius: 15px; 
    padding: 20px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.05); 
    border: 1px solid #f1f1f1; transition: transform 0.2s ease;
}
.metric-box:hover { transform: translateY(-5px); }
.metric-title { 
    font-size: 15px; color: #707070; margin-bottom: 8px; 
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; 
}
.metric-value { 
    font-size: 25px; font-weight: 800; color: #E81A1D; 
}

/* Styling Tabel Markdown Streamlit */
.stMarkdown table { 
    width: 100%; border-collapse: collapse; border-radius: 10px; 
    overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.05); 
    margin-bottom: 30px; font-size: 15px; font-family: 'Inter', sans-serif; background: #FFFFFF;
}
.stMarkdown th { 
    background: linear-gradient(90deg, #E81A1D 0%, #F7947E 100%) !important; 
    color: white !important; font-weight: 700; padding: 15px; 
    text-align: left; border: none !important; 
}
.stMarkdown td { 
    padding: 12px 15px; border-bottom: 1px solid #eee; color: #555; 
    border-right: none !important; border-left: none !important; 
}
.stMarkdown tr:last-child td { border-bottom: none; }
.stMarkdown tr:hover td { background-color: #FFF5F5; }

/* ========== INFORMASI PNEUMONIA ========== */    
/* Styling Dasar Kartu Informasi */
.info-card {
    background-color: #FFFFFF;
    padding: 25px 30px;
    border-radius: 15px;
    margin-bottom: 25px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    transition: transform 0.2s ease;
}
.info-card:hover {
    transform: translateX(5px);
}

/* Varian Warna Garis Tepi (Kiri) */
.info-normal { border-left: 6px solid #27ae60; }
.info-bacterial { border-left: 6px solid #e67e22; }
.info-viral { border-left: 6px solid #e74c3c; }

/* Tipografi di dalam kartu */
.info-card h3 {
    margin-top: 0;
    margin-bottom: 15px;
    color: #1E293B;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 22px;
    font-weight: 800;
}
.info-card p {
    color: #707070;
    line-height: 1.7;
    font-size: 15px;
    margin: 0;
    text-align: justify;
}
.citation {
    font-style: italic;
    color: #999;
    font-size: 13px;
}

/* ========== REFERENSI ========== */  
/* Kartu Daftar Pustaka */
.ref-card {
    background-color: #FFFFFF; 
    padding: 30px; 
    border-radius: 15px; 
    box-shadow: 0 4px 15px rgba(0,0,0,0.05); 
    border-left: 6px solid #1E293B;
    margin-bottom: 25px;
}
.ref-list {
    color: #555; 
    line-height: 2; 
    font-size: 15px; 
    padding-left: 20px; 
    margin: 0;
}
.ref-list li {
    margin-bottom: 15px; 
    text-align: justify;
}
.ref-list li:last-child {
    margin-bottom: 0;
}

/* Kartu Lisensi & Keterangan Dataset */
.license-card {
    background: linear-gradient(180deg, #F8FAFC 0%, #FFFFFF 100%);
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #E2E8F0;
    color: #707070;
    font-size: 15px;
    line-height: 1.6;
    text-align: justify;
    box-shadow: 0 4px 15px rgba(0,0,0,0.02);
    margin-bottom: 30px;
}
                
</style>

""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES   = ["bacterial", "normal", "viral"]
CLASS_COLORS  = ["#e67e22", "#27ae60", "#e74c3c"]
IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
MODEL_PATH    = "pneumonia_mobilenetv3_deploy.pth"  

CLASS_INFO = {
    "normal":    {"icon": "🟢", "label": "NORMAL", "sub": "Paru-paru Sehat"},
    "bacterial": {"icon": "🟠", "label": "BACTERIAL", "sub": "Pneumonia Bacterial"},
    "viral":     {"icon": "🔴", "label": "VIRAL", "sub": "Pneumonia Viral"},
}


# ─── Model Loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = mobilenet_v3_small(weights=None)
    in_f   = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_f, len(CLASS_NAMES))

    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, device, ckpt


# ─── Preprocessing ────────────────────────────────────────────────────────────
def segment_lung(img_gray):
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_gray
    top_n = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
    h, w   = img_gray.shape
    xmn, ymn, xmx, ymx = w, h, 0, 0
    for cnt in top_n:
        x, y, cw, ch = cv2.boundingRect(cnt)
        xmn = min(xmn, x); ymn = min(ymn, y)
        xmx = max(xmx, x+cw); ymx = max(ymx, y+ch)
    dx = int((xmx-xmn)*0.05); dy = int((ymx-ymn)*0.05)
    xmn = max(0,xmn-dx); ymn = max(0,ymn-dy)
    xmx = min(w,xmx+dx); ymx = min(h,ymx+dy)
    c = img_gray[ymn:ymx, xmn:xmx]
    return c if c.size > 100 else img_gray


def preprocess(pil_img):
    img = np.array(pil_img.convert("L"))  # grayscale
    img = segment_lung(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img3 = np.stack([img, img, img], axis=2)
    tf  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    return tf(Image.fromarray(img3.astype(np.uint8)))


def denormalize(tensor):
    t = tensor.clone()
    for c, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        t[c] = t[c]*s + m
    return t.clamp(0,1)


def run_gradcam(model, img_tensor, pred_class, device):
    target_layers = [model.features[-1]]
    cam_obj = GradCAM(model=model, target_layers=target_layers)
    input_t = img_tensor.unsqueeze(0).to(device)
    targets = [ClassifierOutputTarget(pred_class)]
    cam_map  = cam_obj(input_tensor=input_t, targets=targets)[0]
    img_dn   = denormalize(img_tensor).permute(1,2,0).numpy()
    img_dn   = np.clip(img_dn, 0, 1)
    overlay  = show_cam_on_image(img_dn, cam_map, use_rgb=True)
    cam_obj.__exit__(None, None, None)
    return overlay, cam_map


# ─── Sidebar Navigation ───────────────────────────────────────────────────────
# Logo Sidebar (Putih)
st.sidebar.markdown('''
<div class="sidebar-logo-card">
    <h2>🫁 AI-VEOLI</h2>
</div>
''', unsafe_allow_html=True)

# Menu Navigasi
menu = st.sidebar.radio(
    "",
    ["🏠 HOME", "🔬 KLASIFIKASI CITRA", "📊 METRIK MODEL", "🦠 INFORMASI PNEUMONIA", "📚 REFERENSI"],
    index=0
)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
if menu == "🏠 HOME":
    
    # --- Top Section ---
    st.markdown('''
    <div class="hero-container">
        <div class="main-title">🫁 AI-VEOLI</div>
        <div class="sub-title">Sistem Klasifikasi Citra X-Ray Pneumonia Berbasis Deep Learning</div>
        <div class="badge-container">
            <div class="badge">MobileNetV3</div>
            <div class="badge">PyTorch</div>
            <div class="badge">Grad-CAM</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    

    # --- Cards Section ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔬</div>
            <h3>KLASIFIKASI CITRA</h3>
            <p>Upload Citra X-Ray dada berformat JPG/ PNG untuk melihat hasil klasifikasi dengan per-class confidence score serta visualisasi Grad-CAM.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3>METRIK MODEL</h3>
            <p>Informasi terkait laporan evaluasi model meliputi: akurasi, recall, precision, ROC-AUC, confusion matrix serta visualisasi riwayat pelatihan.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🦠</div>
            <h3>INFORMASI PNEUMONIA</h3>
            <p>Pelajari lebih lanjut terkait karakteristik pneumonia berdasarkan penyebabnya seperti: virus dan bakteri.</p>
        </div>
        """, unsafe_allow_html=True)

    # --- Disclaimer Section ---
    st.markdown("""
    <div class="disclaimer-card">
        <div class="disclaimer-content">
            <div class="disclaimer-title">⚠️ DISCLAIMER</div>
            <div class="disclaimer-text">Aplikasi ini bersifat alat bantu diagnostik (decision support) dan <b>TIDAK</b> menggantikan diagnosis dokter/radiologis. Hasil harus selalu diinterpretasikan oleh tenaga medis yang berkompeten.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Footer Section ---
    st.markdown("""
    <div class="custom-footer">
        AI-VEOLI | Pneumonia Classifier | Version 1.1<br>
        Created by: <a style="text-decoration: none;" href="https://github.com/wicaksonohanif" target="_blank"><span>@wicaksonohanif_</span></a><br>
        Seluruh referensi termasuk dataset tercantum pada halaman Referensi
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
elif menu == "🔬 KLASIFIKASI CITRA":
    # --- HEADER ---
    st.markdown('''
    <div class="header-card">
        <span style="font-size: 40px;">🔬</span>
        <h1>Klasifikasi Citra</h1>
    </div>
    ''', unsafe_allow_html=True)

    # --- LOAD MODEL ---
    model, device, ckpt = load_model(MODEL_PATH)

    # --- UPLOAD & PREVIEW AREA ---
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        uploaded = st.file_uploader("📂 Upload Citra X-Ray Dada", type=["jpg", "jpeg", "png"])

    with col_right:
        if uploaded:
            pil_img = Image.open(uploaded)
            st.image(pil_img, caption="Preview Citra Input", use_container_width=True)
        else:
            st.markdown('''
            <div style="background-color: #ffffff; height: 175px; border-radius: 15px; display: flex; align-items: center; justify-content: center; color: #707070; font-size: 15px; font-weight: 600; font-family: 'Inter', sans-serif;">
                Upload gambar untuk melakukan preview!
            </div>
            ''', unsafe_allow_html=True)

    # --- FILE PATH BAR ---
    file_name = uploaded.name if uploaded else "Belum ada file yang dipilih"
    st.markdown(f'<div class="file-path-bar">📁 {file_name}</div>', unsafe_allow_html=True)

    # --- PREDICT BUTTON ---
    if st.button("🔮 Prediksi"):
        if uploaded:
            start_time = time.time()
            with st.spinner("Menganalisis Citra..."):
                # Preprocessing & Inference
                img_tensor = preprocess(pil_img)
                with torch.no_grad():
                    logit = model(img_tensor.unsqueeze(0).to(device))
                    probs = F.softmax(logit, dim=1)[0].cpu().numpy()
                
                pred_idx = int(np.argmax(probs))
                pred_class = CLASS_NAMES[pred_idx]
                confidence = float(probs[pred_idx])
                inf_time = time.time() - start_time

                # --- RESULT METRICS BAR ---
                info = CLASS_INFO[pred_class]
                st.markdown(f'''
                <div class="result-bar">
                    <div class="result-item">
                        <div class="result-value">{info["icon"]} {info["label"]}</div>
                        <div class="result-label">{info["sub"]}</div>
                    </div>
                    <div class="result-item">
                        <div class="result-value">{confidence*100:.1f}%</div>
                        <div class="result-label">Confidence Score</div>
                    </div>
                    <div class="result-item">
                        <div class="result-value">{inf_time:.3f}s</div>
                        <div class="result-label">Inference Time</div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                color_map = ['#e67e22', '#27ae60', '#e74c3c'] 
                
                fig = go.Figure(go.Bar(
                    x=probs * 100,
                    y=[c.capitalize() for c in CLASS_NAMES],
                    orientation='h',
                    marker=dict(color=color_map, line=dict(color='#1E293B', width=1)),
                    text=[f"{p*100:.1f}%" for p in probs],
                    textposition='auto',
                ))
                fig.update_layout(
                    title="Confidence per Kelas",
                    xaxis_title="Confidence (%)",
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                
                # Kalkulasi Grad-CAM
                overlay, cam_map = run_gradcam(model, img_tensor, pred_idx, device)

                # Render 2 kolom berdampingan
                col_orig, col_cam = st.columns(2)
                
                with col_orig:
                    # Proses denormalize untuk menampilkan gambar asli
                    orig_np = denormalize(img_tensor).permute(1,2,0).numpy()
                    fig_orig, ax_orig = plt.subplots(figsize=(5, 5))
                    ax_orig.imshow(orig_np[:,:,0], cmap="gray")
                    ax_orig.set_title("Gambar Preprocessed", fontweight="bold")
                    ax_orig.axis("off")
                    st.pyplot(fig_orig)
                    plt.close(fig_orig)

                with col_cam:
                    # Menampilkan overlay Grad-CAM
                    fig_cam, ax_cam = plt.subplots(figsize=(5, 5))
                    ax_cam.imshow(overlay)
                    ax_cam.set_title(f"Grad-CAM — {pred_class.upper()}", fontweight="bold")
                    ax_cam.axis("off")
                    st.pyplot(fig_cam)
                    plt.close(fig_cam)  

        else:
            st.warning("Silakan upload gambar terlebih dahulu!")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL INFO
# ═══════════════════════════════════════════════════════════════════════════════
elif menu == "📊 METRIK MODEL":
    # --- HEADER ---
    st.markdown('''
    <div class="header-card">
        <span style="font-size: 40px;">📊</span>
        <h1>Metrik Model</h1>
    </div>
    ''', unsafe_allow_html=True)

    # --- EKSTRAKSI DATA CHECKPOINT ---
    if Path(MODEL_PATH).exists():
        _, _, ckpt = load_model(MODEL_PATH)
        val_acc = f"{ckpt.get('best_val_acc', 0)*100:.2f}%"
        # Jika nilai test_acc tidak ada di ckpt, gunakan placeholder default sesuai prompt
        test_acc = f"{ckpt.get('test_acc', 0.8928)*100:.2f}%" 
        roc_auc = f"{ckpt.get('roc_auc_macro', 0):.4f}"
        classes_str = ", ".join(ckpt.get('class_names', CLASS_NAMES))
    else:
        st.warning("⚠️ Model belum dimuat. File .pth tidak ditemukan. Menampilkan data placeholder.")
        val_acc, test_acc, roc_auc, classes_str = "88.50%", "89.28%", "0.9450", "Normal, Bacterial, Viral"

    # --- METRIC CARDS ---
    st.markdown(f'''
    <div class="metric-grid">
        <div class="metric-box">
            <div class="metric-title">Arsitektur</div>
            <div class="metric-value" style="font-size: 18px; margin-top: 5px;">MobileNetV3 Small</div>
        </div>
        <div class="metric-box">
            <div class="metric-title">Val Accuracy</div>
            <div class="metric-value">{val_acc}</div>
        </div>
        <div class="metric-box">
            <div class="metric-title">Test Accuracy</div>
            <div class="metric-value">{test_acc}</div>
        </div>
        <div class="metric-box">
            <div class="metric-title">ROC-AUC Macro</div>
            <div class="metric-value">{roc_auc}</div>
        </div>
        <div class="metric-box">
            <div class="metric-title">Kelas</div>
            <div class="metric-value" style="font-size: 14px; margin-top: 5px;">{classes_str}</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # --- VISUALISASI KINERJA ---
    st.markdown("### 📈 Visualisasi Kinerja Model")
    
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        try:
            st.image("./assets/confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)
        except:
            st.info("🖼️ Gambar confusion_matrix.png belum tersedia di direktori.")

    with col_img2:
        try:
            st.image("./assets/roc_auc.png", caption="Kurva ROC-AUC", use_container_width=True)
        except:
            st.info("🖼️ Gambar roc_auc.png belum tersedia di direktori.")

    try:
        st.image("./assets/training_curves.png", caption="Grafik Pelatihan (Loss & Accuracy)", use_container_width=True)
    except:
        st.info("🖼️ Gambar training_curves.png belum tersedia di direktori.")

    st.markdown("---")

    # --- CLASSIFICATION REPORT ---
    st.markdown("### 📑 Classification Report")
    st.markdown("""
    | Kelas | Precision | Recall | F1-Score | Support |
    |-------|-----------|--------|----------|---------|
    | **Bacterial** | 0.8908 | 0.8433 | 0.8664 | 300 |
    | **Normal** | 0.9702 | 0.9939 | 0.9819 | 328 |
    | **Viral** | 0.7457 | 0.7818 | 0.7633 | 165 |
    | **Accuracy** | | | **0.8928** | **793** |
    | **Macro Avg** | 0.8689 | 0.8730 | 0.8706 | 793 |
    | **Weighted Avg** | 0.8935 | 0.8928 | 0.8927 | 793 |
    """)

    st.markdown("---")

    # --- PIPELINE & PARAMETER TABS/COLUMNS ---
    col_pipe, col_param = st.columns(2, gap="large")

    with col_pipe:
        st.markdown("### 🏗️ Arsitektur Pipeline")
        st.markdown("""
        | Tahap | Detail |
        |-------|--------|
        | **Input** | Citra X-Ray Grayscale |
        | **Segmentasi** | Otsu Thresholding + Morphological Ops |
        | **Resize** | 224 × 224 piksel |
        | **Kanal** | Grayscale → 3-Channel (replikasi) |
        | **Normalisasi** | ImageNet Mean/Std |
        | **Backbone** | MobileNetV3 Small (Pretrained) |
        | **Classifier** | Linear(1024, 3) |
        | **Output** | Softmax → 3 kelas |
        """)

    with col_param:
        st.markdown("### ⚙️ Hyperparameter")
        st.markdown("""
        | Parameter | Nilai |
        |-----------|-------|
        | **NUM_EPOCHS** | 30 |
        | **BATCH_SIZE** | 32 |
        | **LR_INIT** | 1e-4 |
        | **WEIGHT_DECAY** | 1e-4 |
        | **LR_MILESTONES**| [20] |
        | **LR_GAMMA** | 0.1 |
        | **NUM_WORKERS** | 2 |
        """)

    st.markdown("---")

    # --- AUGMENTASI & STRATEGI ---
    col_aug, col_strat = st.columns(2, gap="large")

    with col_aug:
        st.markdown("### 🖼️ Augmentasi Training")
        st.markdown("""
        | Teknik | Parameter |
        |--------|-----------|
        | **RandomResizedCrop** | scale=(0.80, 1.00) |
        | **RandomAffine** | degrees=±15, shear=±8° |
        | **GaussianBlur** | sigma=(0.1, 1.5), p=0.3 |
        | **RandomAdjustSharpness**| factor=2, p=0.3 |
        | **RandomPerspective** | distortion=0.15, p=0.3 |
        | **RandomErasing** | scale=(0.02, 0.08), p=0.2 |
        """)

    with col_strat:
        st.markdown("### 🎓 Training Strategy")
        st.markdown("""
        - **1 Fase**: Semua bobot MobileNetV3 dibuka dari awal dengan inisialisasi ImageNet
        - **Optimizer**: AdamW
        - **Loss**: CrossEntropyLoss (weighted)
        - **Scheduler**: MultiStepLR
        - **Early Stopping**: Tanpa *early stopping* untuk memastikan kelas viral tidak *underfitting*.
        """)

        st.markdown("### ⚖️ Class Weighting", unsafe_allow_html=True)
        st.markdown("Penalti kelas minoritas dihitung menggunakan distribusi invers:")
        # Render rumus matematis dengan rapi menggunakan LaTeX
        st.latex(r"w_i = \frac{N}{k \times n_i}")
        st.markdown("<p style='font-size: 12px; color: #707070; text-align: center;'>Di mana: N = total sampel, k = jumlah kelas, n_i = sampel kelas ke-i</p>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: INFORMASI PNEUMONIA
# ═══════════════════════════════════════════════════════════════════════════════
elif menu == "🦠 INFORMASI PNEUMONIA":
    # --- HEADER ---
    st.markdown('''
    <div class="header-card">
        <span style="font-size: 40px;">🦠</span>
        <h1>Informasi Pneumonia</h1>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown("<p style='color: #707070; margin-bottom: 30px; font-size: 16px;'>Karakteristik radiologis dari masing-masing kondisi paru-paru berdasarkan tinjauan literatur.</p>", unsafe_allow_html=True)

    # --- KARTU NORMAL ---
    st.markdown('''
    <div class="info-card info-normal">
        <h3>🟢 Paru-paru Normal</h3>
        <p>
            Kelas "Normal" merepresentasikan kondisi sehat yang secara spesifik mengindikasikan absennya kehadiran pneumonia virus maupun bakteri pada pasien. Dalam proses evaluasi diagnosis klinis, paru-paru normal dinilai berdasarkan ketiadaan penumpukan cairan pneumonia di dalam paru-paru, yang keberadaannya biasanya menjadi fokus pencarian utama oleh para ahli radiologi <span class="citation">(Nguyen et al., 2020)</span>.
        </p>
    </div>
    ''', unsafe_allow_html=True)

    # --- KARTU BACTERIAL ---
    st.markdown('''
    <div class="info-card info-bacterial">
        <h3>🟠 Pneumonia Bacterial</h3>
        <p>
            Bakteri merupakan salah satu agen infeksi primer yang menyebabkan kondisi pneumonia, memicu gejala fisik seperti batuk berdahak atau bernanah, demam, menggigil, serta kesulitan bernapas. Pneumonia bakteri secara khas menunjukkan adanya konsolidasi lobar yang terpusat di satu area tertentu (focal lobar consolidation) <span class="citation">(Nguyen et al., 2020)</span>.
        </p>
    </div>
    ''', unsafe_allow_html=True)

    # --- KARTU VIRAL ---
    st.markdown('''
    <div class="info-card info-viral">
        <h3>🔴 Pneumonia Viral</h3>
        <p>
            Pneumonia virus adalah bentuk infeksi pernapasan yang prevalensinya sangat tinggi, di mana hingga 60% dari total kasus pneumonia memiliki keterkaitan dengan infeksi virus pernapasan ini. Secara visual pada hasil rontgen dada, pneumonia virus menampilkan pola interstisial yang menyebar (diffuse interstitial pattern) dan umumnya memengaruhi kedua belah paru-paru secara bersamaan <span class="citation">(Nguyen et al., 2020)</span>.
        </p>
    </div>
    ''', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: REFERENSI
# ═══════════════════════════════════════════════════════════════════════════════
elif menu == "📚 REFERENSI":
    # --- HEADER ---
    st.markdown('''
    <div class="header-card">
        <span style="font-size: 40px;">📚</span>
        <h1>Referensi</h1>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown("<p style='color: #707070; margin-bottom: 20px; font-size: 16px;'>Berikut adalah daftar pustaka yang menjadi acuan dalam pengembangan sistem klasifikasi diagnostik ini:</p>", unsafe_allow_html=True)

    # --- KARTU REFERENSI UTAMA ---
    st.markdown("""
    <div class="ref-card">
        <ol class="ref-list">
            <li>
                <b>Sait, Unais; Lal KV, Gokul; Prakash Prajapati, Sunny; Bhaumik, Rahul; Kumar, Tarun; Shivakumar, Sanjana; Bhalla , Kriti (2021).</b> "Curated Dataset for COVID-19 Posterior-Anterior Chest Radiography Images (X-Rays).", <i>Mendeley Data</i>, V3, doi: 10.17632/9xkhgts2s6.3.
            </li>
            <li>
                <b>Nguyen, Hai & Tran, Toan & Hoang Luong, Huong & Phuoc, Trung & Cong, Nghi. (2020).</b> Viral and Bacterial Pneumonia Diagnosis via Deep Learning Techniques and Model Explainability. <i>International Journal of Advanced Computer Science and Applications</i>, 11, doi: 10.14569/IJACSA.2020.0110780.
            </li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # --- KARTU LISENSI & KETERANGAN DATASET ---
    st.markdown("""
    <div class="license-card">
        <h4 style="color: #1E293B; margin-top: 0; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 20px;">📝</span> Keterangan Penggunaan Dataset
        </h4>
        <p style="margin: 0;">
            Dataset radiografi dada yang digunakan dalam aplikasi ini bersumber dari Mendeley Data (Sait et al., 2021) dan telah dilakukan proses <i>subsetting</i> menjadi 3 kelas utama: <b>Normal, Bacterial, dan Viral</b>. Seluruh tahapan prapemrosesan data (<i>preprocessing</i>) yang diterapkan pada citra orisinal telah dijelaskan secara rinci pada halaman <b>Metrik Model</b>. Penggunaan, modifikasi, dan distribusi turunan dari dataset ini dilakukan dengan mematuhi ketentuan atribusi dari lisensi <b>Creative Commons Attribution 4.0 International (CC BY 4.0)</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)
```