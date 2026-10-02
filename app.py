import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.append(os.path.abspath("."))

from src.deployment.model_loader import XLMInferenceEngine, ASPECTS
from src.deployment.data_processor import DashboardDataProcessor, ASPECT_NAMES, STANDARD_COLUMNS

# Page Configuration
st.set_page_config(
    page_title="EV China ABSA Analytics Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom SaaS Styling (CSS)
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .kpi-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #3498db;
        margin-bottom: 10px;
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #7f8c8d;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
        margin-top: 5px;
    }
    .banner-fallback {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 35px 25px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .banner-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 1px;
    }
    .banner-subtitle {
        font-size: 1.05rem;
        opacity: 0.9;
        margin-top: 8px;
    }
    .comment-card {
        background-color: #ffffff;
        border: 1px solid #e1e8ed;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .badge-positif { background-color: #d4edda; color: #155724; padding: 3px 8px; border-radius: 12px; font-weight: 600; font-size: 0.8rem; }
    .badge-netral { background-color: #fff3cd; color: #856404; padding: 3px 8px; border-radius: 12px; font-weight: 600; font-size: 0.8rem; }
    .badge-negatif { background-color: #f8d7da; color: #721c24; padding: 3px 8px; border-radius: 12px; font-weight: 600; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# Cache Model & Processor Initialization for Streamlit Cloud Compatibility
@st.cache_resource(show_spinner="Memuat Pipeline Model XLM-RoBERTa...")
def get_xlm_engine():
    return XLMInferenceEngine()

@st.cache_resource
def get_data_processor():
    return DashboardDataProcessor()

processor = get_data_processor()


# 1. Top Header Banner Render (4:1 Ratio)
def render_header_banner():
    banner_path = Path("assets/banner.jpg")
    if banner_path.exists():
        st.image(str(banner_path), use_column_width=True)
    else:
        st.markdown("""
        <div class="banner-fallback">
            <h1 class="banner-title">🚗 EV China ABSA Analytics Dashboard</h1>
            <p class="banner-subtitle">Platform Intelijen Sentiment Analysis Komentar Consumer EV China di Indonesia (XLM-RoBERTa Engine)</p>
        </div>
        """, unsafe_allow_html=True)

render_header_banner()


# 2. Main Navigation Mode Selection
st.sidebar.title("🎛️ Navigasi Utama")
mode_selection = st.sidebar.radio(
    "Pilih Fitur Aplikasi:",
    [
        "📊 Pilihan 1: Analisis Data Berlabel (Executive Dashboard)",
        "🤖 Pilihan 2: Lakukan Pelabelan Data (XLM-RoBERTa Inference Engine)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Petunjuk Penggunaan:**\n- **Pilihan 1:** Tampilkan Dasbor SaaS Eksekutif dari data berlabel.\n- **Pilihan 2:** Labeli otomatis komentar mentah dari YouTube API dengan XLM-RoBERTa, lalu unduh hasilnya.")


# ==============================================================================
# MODE 1: ANALISIS DATA BERLABEL (EXECUTIVE DASHBOARD)
# ==============================================================================
if "Pilihan 1" in mode_selection:
    st.title("📊 Executive Dashboard — Analysis Data Berlabel")
    st.markdown("Dasbor intelijen eksekutif untuk menganalisis persepsi konsumen terhadap 4 aspek kendaraan listrik (EV) China.")

    # Data Source Selection
    st.sidebar.subheader("📂 Sumber Data Master")
    uploaded_file = st.sidebar.file_uploader("Unggah CSV Master Berlabel:", type=["csv", "xlsx"])

    df_labeled = None

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".xlsx"):
                df_labeled = pd.read_excel(uploaded_file)
            else:
                df_labeled = pd.read_csv(uploaded_file, encoding="utf-8-sig")
            st.sidebar.success(f"Berhasil memuat: {uploaded_file.name} ({len(df_labeled):,} baris)")
        except Exception as e:
            st.sidebar.error(f"Gagal membaca file: {e}")

    # Fallback to local default dataset
    if df_labeled is None:
        default_paths = [
            Path("data/interim/master_labeled_comments.csv"),
            Path("data/interim/valid_labeled_comments.csv"),
            Path("data/processed/train.csv")
        ]
        for dp in default_paths:
            if dp.exists():
                df_labeled = pd.read_csv(dp, encoding="utf-8-sig")
                st.sidebar.info(f"Menggunakan dataset default: `{dp.name}` ({len(df_labeled):,} baris)")
                break

    if df_labeled is None:
        st.error("❌ Dataset master berlabel tidak ditemukan. Silakan unggah berkas CSV master pada sidebar.")
        st.stop()

    # Calculate SaaS Executive KPI Metrics
    metrics = processor.calculate_saas_metrics(df_labeled)

    # Render SaaS KPI Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Komentar</div>
            <div class="kpi-value">{metrics['total_comments']:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Video YouTube</div>
            <div class="kpi-value">{metrics['videos_analyzed']:,} <span style="font-size: 0.9rem; color: #7f8c8d;">Video</span></div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Cakupan Ber-Aspek</div>
            <div class="kpi-value">{metrics['aspect_coverage_pct']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Aspek Paling Dominan</div>
            <div class="kpi-value" style="font-size: 1.3rem;">{metrics['top_aspect']}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Sentimen Dominan</div>
            <div class="kpi-value" style="font-size: 1.3rem;">{metrics['top_sentiment']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dashboard Tabs
    tab_overview, tab_deepdive, tab_toplikes, tab_dataviewer = st.tabs([
        "📈 Ringkasan Eksekutif",
        "🔍 Aspect Deep-Dive",
        "⭐ Top Liked Comments",
        "📋 Data Viewer"
    ])

    # TAB 1: EXECUTIVE OVERVIEW
    with tab_overview:
        st.subheader("📊 Distribusi Sentimen per 4 Aspek EV China")
        df_aspect_sent = processor.prepare_aspect_sentiment_df(df_labeled)

        col_left, col_right = st.columns([1.2, 0.8])

        with col_left:
            if not df_aspect_sent.empty:
                fig_bar = px.bar(
                    df_aspect_sent,
                    x="Aspek",
                    y="Jumlah",
                    color="Sentimen",
                    barmode="group",
                    color_discrete_map={"Positif": "#2ecc71", "Netral": "#f1c40f", "Negatif": "#e74c3c"},
                    title="Perbandingan Distribusi Sentimen per Aspek",
                    text_auto=True
                )
                fig_bar.update_layout(height=420, legend_title="Sentimen")
                st.plotly_chart(fig_bar, use_container_width=True)

        with col_right:
            aspect_sums = df_aspect_sent.groupby("Aspek")["Jumlah"].sum().reset_index()
            if not aspect_sums.empty and aspect_sums["Jumlah"].sum() > 0:
                fig_donut = px.pie(
                    aspect_sums,
                    names="Aspek",
                    values="Jumlah",
                    hole=0.45,
                    title="Proporsi Diskusi per Aspek",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_donut.update_layout(height=420)
                st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown("---")
        st.markdown(f"**📌 Ringkasan Akumulasi Total Sentiment Label:** `{metrics['total_labels']:,}` Anotasi Sentimen | "
                    f"🟢 **Positif:** `{metrics['pos_count']:,}` | "
                    f"🟡 **Netral:** `{metrics['neu_count']:,}` | "
                    f"🔴 **Negatif:** `{metrics['neg_count']:,}`")

    # TAB 2: ASPECT DEEP-DIVE
    with tab_deepdive:
        st.subheader("🔍 Analisis Kata Kunci & Sentimen per Aspek Spesifik")
        subtab_infra, subtab_ekonomi, subtab_kualitas, subtab_purnajual = st.tabs([
            "🔌 Infrastruktur",
            "💰 Ekonomi & Harga",
            "🚗 Kualitas & Durabilitas",
            "🛠️ Purnajual & Layanan"
        ])

        VISUALIZATION_STOPWORDS = set([
            'yang', 'yg', 'nya', 'di', 'ke', 'dan', 'ini', 'itu', 'ada', 'sudah', 'bisa', 'banyak', 
            'lagi', 'sama', 'kalau', 'kalo', 'akan', 'jadi', 'bikin', 'dari', 'pada', 'buat', 'saja', 
            'aja', 'atau', 'dengan', 'untuk', 'lah', 'pun', 'kan', 'kah', 'deh', 'dong', 'kok', 'juga', 
            'masih', 'belum', 'harus', 'gak', 'ga', 'ngga', 'nggak', 'tidak', 'tak', 'gk', 'apa', 'tapi', 
            'tetap', 'biar', 'pakai', 'pake', 'mau', 'orang', 'sih', 'lu', 'gue', 'gw', 
            'gua', 'dia', 'mereka', 'kita', 'kamu', 'anda', 'saya', 'aku', 'sy', 'om', 'bang', 'min', 
            'bro', 'bos', 'gan', 'sist', 'kak', 'bapak', 'ibu', 'pak', 'bu', 'terus', 
            'seperti', 'karena', 'sampai', 'jika', 'bila', 'semua', 'hal', 'bahkan', 'secara', 'malah'
        ])

        subtabs = [
            ("infra_sentiment", subtab_infra, "Infrastruktur & SPKLU"),
            ("ekonomi_sentiment", subtab_ekonomi, "Ekonomi & Harga EV"),
            ("kualitas_sentiment", subtab_kualitas, "Kualitas & Durabilitas Build Quality"),
            ("purnajual_sentiment", subtab_purnajual, "Purnajual & Layanan Dealer")
        ]

        for col_name, subtab_obj, label_title in subtabs:
            with subtab_obj:
                if col_name in df_labeled.columns:
                    aspect_df = df_labeled[df_labeled[col_name].notnull()]
                    st.write(f"**Total Komentar Membahas {label_title}:** `{len(aspect_df):,}` komentar")

                    c1, c2 = st.columns([1, 1])
                    with c1:
                        s_counts = aspect_df[col_name].astype(str).str.lower().value_counts()
                        df_s = pd.DataFrame({'Sentimen': s_counts.index.str.capitalize(), 'Jumlah': s_counts.values})
                        fig_p = px.pie(df_s, names='Sentimen', values='Jumlah', title=f"Distribusi Sentimen - {label_title}",
                                       color='Sentimen', color_discrete_map={"Positif": "#2ecc71", "Netral": "#f1c40f", "Negatif": "#e74c3c"})
                        st.plotly_chart(fig_p, use_container_width=True)

                    with c2:
                        texts = aspect_df['text_cleaned'].dropna().astype(str).tolist() if 'text_cleaned' in aspect_df.columns else aspect_df['text_original'].dropna().astype(str).tolist()
                        combined_text = " ".join(texts)
                        if len(combined_text.strip()) > 5:
                            wc = WordCloud(width=500, height=300, background_color="white", colormap="Dark2", stopwords=VISUALIZATION_STOPWORDS).generate(combined_text)
                            fig_wc, ax_wc = plt.subplots(figsize=(6, 4))
                            ax_wc.imshow(wc, interpolation="bilinear")
                            ax_wc.axis("off")
                            st.pyplot(fig_wc)
                        else:
                            st.info("Teks tidak cukup untuk membuat WordCloud.")

    # TAB 3: TOP LIKED COMMENTS
    with tab_toplikes:
        st.subheader("⭐ Galeri Komentar Terpopuler (Top Liked Comments)")
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            sel_aspect = st.selectbox("Pilih Aspek:", list(ASPECT_NAMES.values()))
        with f_col2:
            sel_sentiment = st.selectbox("Pilih Sentimen:", ["Positif", "Netral", "Negatif"])
        with f_col3:
            top_n_count = st.slider("Jumlah Komentar Ditampilkan:", 3, 20, 5)

        inv_aspect_names = {v: k for k, v in ASPECT_NAMES.items()}
        target_aspect_col = inv_aspect_names[sel_aspect]

        top_comments = processor.filter_top_liked_comments(df_labeled, aspect_col=target_aspect_col, sentiment=sel_sentiment, top_n=top_n_count)

        if not top_comments.empty:
            for idx, row in top_comments.iterrows():
                like_num = int(row.get('like_count', 0)) if pd.notna(row.get('like_count')) else 0
                comment_txt = row.get('text_original', row.get('text_cleaned', ''))
                vid_id = row.get('video_id', '')

                badge_class = f"badge-{sel_sentiment.lower()}"
                
                st.markdown(f"""
                <div class="comment-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span class="{badge_class}">{sel_sentiment.upper()} — {sel_aspect}</span>
                        <span style="font-weight: 700; color: #e67e22;">👍 {like_num:,} Likes</span>
                    </div>
                    <p style="font-size: 1.05rem; color: #2c3e50; margin: 8px 0;">"{comment_txt}"</p>
                    <div style="font-size: 0.8rem; color: #95a5a6;">Video ID: <code>{vid_id}</code></div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(f"Tidak ada komentar ditemukan untuk Aspek **{sel_aspect}** dengan Sentimen **{sel_sentiment}**.")

    # TAB 4: DATA VIEWER
    with tab_dataviewer:
        st.subheader("📋 Tabel Data Master Berlabel")
        search_query = st.text_input("🔍 Cari Teks Komentar:")
        
        df_display = df_labeled.copy()
        if search_query:
            text_mask = df_display.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)
            df_display = df_display[text_mask]

        st.dataframe(df_display, use_container_width=True)
        
        # Standardized CSV Export
        std_export_df = processor.standardize_output_dataframe(df_labeled)
        csv_data = std_export_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="📥 Unduh Data Master Berlabel (CSV Standard)",
            data=csv_data,
            file_name="master_labeled_comments_export.csv",
            mime="text/csv"
        )


# ==============================================================================
# MODE 2: LAKUKAN PELABELAN DATA (XLM-ROBERTA INFERENCE ENGINE)
# ==============================================================================
else:
    st.title("🤖 Inferensi & Pelabelan Otomatis XLM-RoBERTa")
    st.markdown("Unggah komentar mentah hasil tarikan YouTube API (`yt_comments_raw.csv`) untuk menjalankan inferensi model XLM-RoBERTa PyTorch.")

    raw_file = st.file_uploader("Unggah Komentar Mentah Hasil Tarikan YouTube API (CSV):", type=["csv"])

    if raw_file is not None:
        try:
            df_raw = pd.read_csv(raw_file, encoding="utf-8-sig")
            st.success(f"Berhasil membaca file mentah: `{raw_file.name}` ({len(df_raw):,} baris)")
        except Exception as e:
            st.error(f"Gagal membaca file CSV mentah: {e}")
            df_raw = None
    else:
        st.info("💡 **Belum ada file diunggah.** Anda dapat mengunggah file komentar mentah YouTube API Anda sendiri. "
                "Sebagai contoh uji coba, Anda bisa menggunakan data sampel di bawah ini.")
        
        sample_raw_path = Path("data/raw/yt_comments_additional_20260927_210207.csv")
        if sample_raw_path.exists():
            if st.button("🧪 Gunakan Sample Raw Data"):
                df_raw = pd.read_csv(sample_raw_path, encoding="utf-8-sig").head(50)
                st.success(f"Berhasil memuat 50 baris sampel mentah dari `{sample_raw_path.name}`")
            else:
                df_raw = None
        else:
            df_raw = None

    if df_raw is not None:
        st.subheader("👀 Pratinjau Teks Mentah")
        st.dataframe(df_raw.head(5), use_container_width=True)

        if st.button("🚀 Jalankan Pelabelan Otomatis (XLM-RoBERTa Engine)"):
            with st.spinner("Memuat model XLM-RoBERTa & menyiapkan pipeline..."):
                xlm_engine = get_xlm_engine()

            st.info("⚙️ Memproses pembersihan teks (text cleaner) & inferensi multi-head...")
            
            # Step 1: Text Cleaning
            if "text_cleaned" not in df_raw.columns:
                text_col = "text_original" if "text_original" in df_raw.columns else df_raw.columns[0]
                df_raw["text_cleaned"] = processor.clean_text_series(df_raw[text_col])

            # Step 2: Batch Inference
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(pct):
                progress_bar.progress(pct)
                status_text.text(f"Memproses inferensi XLM-RoBERTa: {int(pct*100)}%")

            texts_to_predict = df_raw["text_cleaned"].tolist()
            predictions_dict = xlm_engine.predict_batch(texts_to_predict, batch_size=16, progress_callback=update_progress)

            # Assign aspect predictions
            for col_name, preds in predictions_dict.items():
                df_raw[col_name] = preds

            st.success("🎉 Pelabelan Otomatis XLM-RoBERTa Selesai!")

            # Standardize output columns WITHOUT human_* or rule_* engine columns (as per Jawaban 3)
            df_final_output = processor.standardize_output_dataframe(df_raw)

            st.subheader("📋 Hasil Pelabelan Otomatis (Cuplikan 10 Baris)")
            st.dataframe(df_final_output.head(10), use_container_width=True)

            # Download Button
            csv_export = df_final_output.to_csv(index=False, encoding="utf-8-sig")
            
            st.markdown("---")
            st.success("💡 **Petunjuk Langkah Selanjutnya:** Unduh berkas CSV berlabel di bawah ini, kemudian Anda dapat beralih ke **Pilihan 1 (Executive Dashboard)** di sidebar untuk langsung menganalisis hasilnya!")
            
            st.download_button(
                label="📥 Unduh CSV Berlabel (Standard 4 Aspek)",
                data=csv_export,
                file_name="yt_comments_xlmroberta_labeled.csv",
                mime="text/csv"
            )
