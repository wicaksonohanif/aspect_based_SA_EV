# 🎨 Skill & Design System: Frontend UI/UX Streamlit Web Application

Dokumen ini adalah acuan **Design System & Template Preferensi Tampilan Frontend** untuk pembuatan aplikasi web **Aspect-Based Sentiment Analysis (ABSA) Komentar Mobil Listrik (EV)** berbasis **Streamlit**.

Silakan isi atau ubah variabel dan preferensi di bawah ini sesuai keinginan Anda. Template ini akan dibaca secara otomatis oleh AI sebagai petunjuk utama dalam mengodekan `app.py` dan komponen kustom CSS.

---

## 🛠️ 1. Tema & Palet Warna (Color Palette & Dark/Light Mode)

- **Mode Tampilan Utama:** `Dark Mode` / `Light Mode` *(Pilih salah satu)*
- **Warna Utama (Primary Accent):** `#00D2FF` *(misal: Electric Cyan / Blue Otomotif)*
- **Warna Latar Belakang (Background):** `#0F172A` *(misal: Deep Slate Navy)*
- **Warna Latar Kartu (Card Background):** `#1E293B` *(misal: Dark Slate Gray)*
- **Warna Teks Utama (Text Color):** `#F8FAFC` *(Pure White / Soft White)*

### 🚥 Palet Warna Sentimen per Aspek:
- 🟢 **Sentimen Positif:** `#10B981` *(Emerald Green)*
- 🟡 **Sentimen Netral:** `#F59E0B` *(Amber Yellow)*
- 🔴 **Sentimen Negatif:** `#EF4444` *(Coral Red)*
- ⚪ **Aspek Non-aktif / None:** `#6B7280` *(Muted Gray)*

---

## 🔤 2. Tipografi & Font (Typography)

- **Font Utama Judul (Headings):** `'Plus Jakarta Sans'`, `'Poppins'`, atau `'Inter'`
- **Font Isi Teks (Body Text):** `'Inter'`, `'Segoe UI'`, atau `'Roboto'`
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
- **Legend Position:** Di bawah grafik (*Bottom Horizontal Legend*).

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

*Tuliskan catatan, keinginan khusus, atau referensi contoh tampilan website yang Anda sukai di sini:*

1. **Keinginan Khusus 1:** *(misal: Saya ingin ada toggle mode atau filter cepat per brand EV BYD, Wuling, Chery)*
2. **Keinginan Khusus 2:** *(misal: Kartu komentar top likes harus memiliki tombol salin teks)*
3. **Keinginan Khusus 3:** *(misal: Tampilan header harus kelihatan futuristik ala otomotif EV)*
