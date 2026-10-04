import os
import sys
import io
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

def generate_excel_template():
    buffer = io.BytesIO()
    sample_data = [{
        "comment_id": "c_sample_001",
        "video_id": "v_sample_101",
        "user_id_hash": "usr_abc123",
        "text_original": "Mobil listrik ini harganya sangat terjangkau tapi build quality nya bagus banget.",
        "like_count": 15,
        "published_at": "2026-09-01 10:00:00",
        "updated_at": "2026-09-01 10:00:00",
        "extracted_at": "2026-09-02 12:00:00",
        "text_cleaned": "mobil listrik ini harganya sangat terjangkau tapi build quality nya bagus banget",
        "infra_sentiment": None,
        "ekonomi_sentiment": "positif",
        "kualitas_sentiment": "positif",
        "purnajual_sentiment": None
    }]
    template_df = pd.DataFrame(sample_data, columns=STANDARD_COLUMNS)
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        template_df.to_excel(writer, index=False, sheet_name="Master_Template")
    return buffer.getvalue()

# Page Configuration
st.set_page_config(
    page_title="SentyBoard Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Inter Font, Blue Gradient Sidebar Layout matching misc.md reference)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif !important;
    }
    
    h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
    }

    .stMarkdown p, .stMarkdown ul, .stMarkdown ol, .stMarkdown li, [data-testid="stMarkdownContainer"] p {
        font-family: 'Inter', sans-serif !important;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* ---------- Sidebar Blue Gradient Layout (Referenced from misc.md) ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #123fb2 0%, #091459 100%) !important;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
        font-size: 15px;
        font-weight: 600;
    }

    /* Hide radio circle dot */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }

    /* Pill-shaped radio menu options */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
        padding: 12px 20px !important;
        border-radius: 25px !important;
        border: 2px solid transparent !important; 
        margin-bottom: 8px !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
        width: 100% !important;
        background-color: transparent;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
        background-color: rgba(255, 255, 255, 0.15) !important;
        transform: translateX(4px) !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        border: 2px solid #FFFFFF !important;
        background-color: rgba(255, 255, 255, 0.25) !important;
    }

    /* ---------- KPI Cards ---------- */
    .kpi-card {
        background-color: #161b22;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        border-left: 5px solid #3498db;
        margin-bottom: 12px;
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #8b949e;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f0f6fc;
        margin-top: 5px;
    }

    /* ---------- Header Banner Fallback ---------- */
    .banner-fallback {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 35px 25px;
        border-radius: 14px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.12);
    }
    .banner-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 0.5px;
    }
    .banner-subtitle {
        font-size: 1.05rem;
        opacity: 0.9;
        margin-top: 8px;
    }

    /* ---------- Comment Cards ---------- */
    .comment-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.25);
    }
    .badge-positif { background-color: #1b4332; color: #75b798; padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 0.8rem; }
    .badge-netral { background-color: #433511; color: #ffda6a; padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 0.8rem; }
    .badge-negatif { background-color: #4c1d24; color: #ea868f; padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 0.8rem; }
</style>

""", unsafe_allow_html=True)


# Cache Model & Processor Initialization
@st.cache_resource(show_spinner="Memuat Pipeline Model XLM-RoBERTa...")
def get_xlm_engine():
    try:
        return XLMInferenceEngine()
    except Exception as e:
        print(f"[App Warning] Exception in get_xlm_engine: {e}")
        return XLMInferenceEngine(model_dir="non_existent")  # Fallback to rule engine

def get_data_processor():
    return DashboardDataProcessor()

processor = get_data_processor()


# 1. Top Header Banner Render (4:1 Ratio)
def render_header_banner():
    banner_path = Path("assets/banner.jpg")
    if banner_path.exists():
        st.image(str(banner_path), use_container_width=True)
    else:
        st.markdown("""
        <div class="banner-fallback">
            <h1 class="banner-title">🚗 EV China ABSA Analytics Dashboard</h1>
            <p class="banner-subtitle">Platform Intelijen Sentiment Analysis Komentar Consumer EV China di Indonesia (XLM-RoBERTa Engine)</p>
        </div>
        """, unsafe_allow_html=True)

render_header_banner()


# 2. Sidebar Navigation

nav_selection = st.sidebar.radio(
    "",
    ["ANALYTICS", "LABELING"],
    index=0
)

# ==============================================================================
# MODE 1: DASHBOARD (ANALISIS DATA BERLABEL)
# ==============================================================================
if nav_selection == "ANALYTICS":
    st.title("Sentiment Analytics")
    sample_path = Path("data/interim/valid_labeled_comments.csv")
    
    col_up_title, col_up_sample, col_up_dl = st.columns([2, 1.2, 1.2])
    with col_up_title:
        st.markdown("#### 📂 Unggah Data Master Berlabel (CSV / XLSX)")
    with col_up_sample:
        if sample_path.exists():
            if st.button("📊 Gunakan Data Sampel", use_container_width=True, help="Muat data sampel valid_labeled_comments.csv"):
                st.session_state["mode1_use_sample"] = True
    with col_up_dl:
        excel_template_bytes = generate_excel_template()
        st.download_button(
            label="📄 Unduh Template XLSX",
            data=excel_template_bytes,
            file_name="template_master_labeled_comments.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    uploaded_file = st.file_uploader(
        "Pilih file CSV atau XLSX master yang berisi label 4 aspek:",
        type=["csv", "xlsx"],
        key="mode1_main_uploader",
        label_visibility="collapsed"
    )

    df_labeled = None

    if uploaded_file is not None:
        st.session_state["mode1_use_sample"] = False
        try:
            if uploaded_file.name.endswith(".xlsx"):
                df_labeled = pd.read_excel(uploaded_file)
            else:
                df_labeled = pd.read_csv(uploaded_file, encoding="utf-8-sig")
            st.success(f"Berhasil memuat berkas: `{uploaded_file.name}` ({len(df_labeled):,} baris data)")
        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")
            st.stop()
    elif st.session_state.get("mode1_use_sample", False) and sample_path.exists():
        try:
            df_labeled = pd.read_csv(sample_path, encoding="utf-8-sig")
            st.info(f"💡 Menggunakan Data Sampel Bawaan: `valid_labeled_comments.csv` ({len(df_labeled):,} baris data)")
        except Exception as e:
            st.error(f"❌ Gagal membaca file data sampel: {e}")
            st.stop()
    else:
        st.info("👆 Silakan unggah berkas data berlabel Anda atau klik **Gunakan Data Sampel** untuk langsung menganalisis data bawaan.")
        st.stop()

    st.markdown("---")

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
        "📈 Executive Summary",
        "🔍 Aspect Deep-Dive",
        "⭐ Top Liked Comments",
        "📋 Data Viewer"
    ])

    # TAB 1: EXECUTIVE OVERVIEW (Strict 2-Column Grid Layout)
    with tab_overview:
        st.subheader("📈 Ringkasan Visual Distribusi Sentimen & Aspek")
        df_aspect_sent = processor.prepare_aspect_sentiment_df(df_labeled)

        # ROW 1 (2 Columns): Grouped Bar Chart & Overall Sentiment Pie Chart
        row1_col1, row1_col2 = st.columns(2)

        with row1_col1:
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
                fig_bar.update_layout(height=400, legend_title="Sentimen")
                st.plotly_chart(fig_bar, use_container_width=True)

        with row1_col2:
            df_overall_pie = processor.prepare_overall_sentiment_pie_df(df_labeled)
            if not df_overall_pie.empty and df_overall_pie["Jumlah"].sum() > 0:
                fig_overall_pie = px.pie(
                    df_overall_pie,
                    names="Sentimen",
                    values="Jumlah",
                    hole=0.35,
                    title="Persentase Keseluruhan Sentimen",
                    color="Sentimen",
                    color_discrete_map={"Positif": "#2ecc71", "Netral": "#f1c40f", "Negatif": "#e74c3c"}
                )
                fig_overall_pie.update_layout(height=400)
                st.plotly_chart(fig_overall_pie, use_container_width=True)

        st.markdown("---")

        # ROW 2 (2 Columns): Aspect Donut Chart & Aspect Co-occurrence Matrix Heatmap
        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            aspect_sums = df_aspect_sent.groupby("Aspek")["Jumlah"].sum().reset_index() if not df_aspect_sent.empty else pd.DataFrame()
            if not aspect_sums.empty and aspect_sums["Jumlah"].sum() > 0:
                fig_donut = px.pie(
                    aspect_sums,
                    names="Aspek",
                    values="Jumlah",
                    hole=0.45,
                    title="Proporsi Diskusi per Aspek",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_donut.update_layout(height=400)
                st.plotly_chart(fig_donut, use_container_width=True)

        with row2_col2:
            co_matrix, aspect_labels = processor.prepare_cooccurrence_matrix(df_labeled)
            fig_matrix = px.imshow(
                co_matrix,
                x=aspect_labels,
                y=aspect_labels,
                color_continuous_scale="Blues",
                text_auto=True,
                aspect="auto",
                title="Matriks Ko-okurensi Kemunculan Aspek"
            )
            fig_matrix.update_layout(
                height=400,
                coloraxis_showscale=False,
                xaxis_title="Aspek",
                yaxis_title="Aspek"
            )
            st.plotly_chart(fig_matrix, use_container_width=True)

        st.markdown("---")

        # ROW 3 (2 Columns): Word Count Length Distribution per Sentiment & Key Summary Card
        row3_col1, row3_col2 = st.columns(2)

        with row3_col1:
            df_wc_dist = processor.prepare_word_count_distribution_df(df_labeled)
            if not df_wc_dist.empty:
                fig_box = px.box(
                    df_wc_dist,
                    x="Sentimen",
                    y="Jumlah Kata",
                    color="Sentimen",
                    color_discrete_map={"Positif": "#2ecc71", "Netral": "#f1c40f", "Negatif": "#e74c3c"},
                    title="Distribusi Panjang Kata Komentar per Sentimen",
                    points="outliers"
                )
                fig_box.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("Data tidak cukup untuk menampilkan distribusi panjang kata.")

        with row3_col2:
            tot_lbls = metrics['total_labels']
            pos_c = metrics['pos_count']
            neu_c = metrics['neu_count']
            neg_c = metrics['neg_count']
            pos_p = (pos_c / tot_lbls * 100) if tot_lbls > 0 else 0
            neu_p = (neu_c / tot_lbls * 100) if tot_lbls > 0 else 0
            neg_p = (neg_c / tot_lbls * 100) if tot_lbls > 0 else 0

            st.markdown(f"""
            <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.25); height: 400px; display: flex; flex-direction: column; justify-content: center;">
                <h4 style="color: #58a6ff; margin-top: 0;">Akumulasi Ringkasan Label Sentimen</h4>
                <p style="color: #8b949e; font-size: 0.95rem;">Rincian total anotasi sentimen konsumen EV China pada 4 aspek utama (Infrastruktur, Ekonomi, Kualitas, Purnajual):</p>
                <hr style="margin: 15px 0; border-color: #30363d;">
                <div style="font-size: 1.05rem; line-height: 2.2; font-weight: 600;">
                    <div>📝 Total Anotasi Label: <span style="color: #f0f6fc;">{tot_lbls:,}</span></div>
                    <div>🟢 Sentimen Positif: <span style="color: #2ecc71;">{pos_c:,} ({pos_p:.1f}%)</span></div>
                    <div>🟡 Sentimen Netral: <span style="color: #f1c40f;">{neu_c:,} ({neu_p:.1f}%)</span></div>
                    <div>🔴 Sentimen Negatif: <span style="color: #e74c3c;">{neg_c:,} ({neg_p:.1f}%)</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

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
            'ya', 'yang', 'yg', 'nya', 'di', 'ke', 'dan', 'ini', 'itu', 'ada', 'sudah', 'bisa', 'banyak', 
            'lagi', 'sama', 'kalau', 'kalo', 'akan', 'jadi', 'bikin', 'dari', 'pada', 'buat', 'saja', 
            'aja', 'atau', 'dengan', 'untuk', 'lah', 'pun', 'kan', 'kah', 'deh', 'dong', 'kok', 'juga', 
            'masih', 'belum', 'harus', 'gak', 'ga', 'ngga', 'nggak', 'tidak', 'tak', 'gk', 'apa', 'tapi', 
            'tetap', 'biar', 'pakai', 'pake', 'mau', 'orang', 'sih', 'lu', 'gue', 'gw', 
            'gua', 'dia', 'mereka', 'kita', 'kamu', 'anda', 'saya', 'aku', 'sy', 'om', 'bang', 'min', 
            'bro', 'bos', 'gan', 'sist', 'kak', 'bapak', 'ibu', 'pak', 'bu', 'terus', 
            'seperti', 'karena', 'sampai', 'jika', 'bila', 'semua', 'hal', 'bahkan', 'secara', 'malah'
        ])

        subtabs = [
            ("infra_sentiment", subtab_infra, "Infrastruktur, SPKLU, dan Jarak Tempuh"),
            ("ekonomi_sentiment", subtab_ekonomi, "Ekonomi, Harga EV, dan Garansi"),
            ("kualitas_sentiment", subtab_kualitas, "Kualitas, Desain, Durabilitas, dan Performa"),
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
                        st.markdown("**☁️ WordCloud Ringkasan Keseluruhan**")
                        texts = aspect_df['text_cleaned'].dropna().astype(str).tolist() if 'text_cleaned' in aspect_df.columns else aspect_df['text_original'].dropna().astype(str).tolist()
                        combined_text = " ".join(texts)
                        if len(combined_text.strip()) > 5:
                            wc = WordCloud(width=500, height=300, background_color="white", colormap="Dark2", stopwords=VISUALIZATION_STOPWORDS).generate(combined_text)
                            fig_wc, ax_wc = plt.subplots(figsize=(6, 4))
                            ax_wc.imshow(wc, interpolation="bilinear")
                            ax_wc.axis("off")
                            st.pyplot(fig_wc)
                            plt.close(fig_wc)
                        else:
                            st.info("Teks tidak cukup untuk membuat WordCloud.")

                    st.markdown("---")
                    st.markdown(f"#### ☁️ WordCloud Kata Kunci Per-Sentimen — {label_title}")
                    st.caption("Membandingkan topik & kata kunci utama pada sentimen Positif, Netral, dan Negatif.")

                    c_pos, c_neu, c_neg = st.columns(3)
                    sentiment_configs = [
                        ("positif", "Positif", "🟢", "Greens", "#2ecc71", c_pos),
                        ("netral", "Netral", "🟡", "YlOrBr", "#f1c40f", c_neu),
                        ("negatif", "Negatif", "🔴", "Reds", "#e74c3c", c_neg)
                    ]

                    for sent_code, sent_label, icon, cmap_name, text_color, col_obj in sentiment_configs:
                        with col_obj:
                            sub_df = aspect_df[aspect_df[col_name].astype(str).str.lower() == sent_code]
                            count_val = len(sub_df)
                            st.markdown(f"<h5 style='color:{text_color}; margin-bottom: 5px;'>{icon} {sent_label} <span style='font-size: 0.85rem; color: #7f8c8d;'>(N={count_val:,})</span></h5>", unsafe_allow_html=True)
                            
                            if count_val > 0:
                                s_texts = sub_df['text_cleaned'].dropna().astype(str).tolist() if 'text_cleaned' in sub_df.columns else sub_df['text_original'].dropna().astype(str).tolist()
                                s_combined = " ".join(s_texts)
                                if len(s_combined.strip()) > 5:
                                    s_wc = WordCloud(
                                        width=400,
                                        height=280,
                                        background_color="white",
                                        colormap=cmap_name,
                                        max_words=60,
                                        collocations=False,
                                        stopwords=VISUALIZATION_STOPWORDS
                                    ).generate(s_combined)
                                    fig_swc, ax_swc = plt.subplots(figsize=(5, 3.5))
                                    ax_swc.imshow(s_wc, interpolation="bilinear")
                                    ax_swc.axis("off")
                                    st.pyplot(fig_swc)
                                    plt.close(fig_swc)
                                else:
                                    st.info(f"Teks tidak cukup untuk WordCloud {sent_label}.")
                            else:
                                st.info(f"Belum ada data sentimen {sent_label}.")

    # TAB 3: TOP LIKED COMMENTS
    with tab_toplikes:
        st.subheader("⭐ Komentar Terpopuler")
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
                
                st.markdown(f"""<div class="comment-card">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
    <span class="{badge_class}">{sel_sentiment.upper()} — {sel_aspect}</span>
    <span style="font-weight: 700; color: #e67e22;">👍 {like_num:,} Likes</span>
</div>
<p style="font-size: 1.05rem; color: #f0f6fc; margin: 8px 0;">"{comment_txt}"</p>
<div style="font-size: 0.85rem; color: #8b949e;">Video ID: <code style="color: #58a6ff; background-color: rgba(110,118,129,0.4); padding: 2px 6px; border-radius: 4px;">{vid_id}</code></div>
</div>""", unsafe_allow_html=True)

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
# MODE 2: PELABELAN (XLM-ROBERTA INFERENCE ENGINE)
# ==============================================================================
else:
    st.title("Inference & Labeling")
    
    xlm_engine = get_xlm_engine()
    if xlm_engine and xlm_engine.is_weights_loaded:
        st.success(f"⚡ **Inference Engine Ready:** XLM-RoBERTa Fine-Tuned Model Loaded | **Device:** `{xlm_engine.device}`")
    else:
        st.info("⚡ **Serverless Cloud API Engine Active:** Menggunakan Cloud API & Gemini Engine (RAM Optimized untuk Cloud)")

    col_raw_title, col_raw_sample = st.columns([2.5, 1])
    with col_raw_title:
        st.markdown("#### 📂 Unggah Komentar Mentah YouTube API (CSV)")
    with col_raw_sample:
        dummy_sample_path = Path("data/dummy/dummy_inference_sample.csv")
        interim_sample_path = Path("data/interim/valid_labeled_comments.csv")
        target_raw_sample = dummy_sample_path if dummy_sample_path.exists() else interim_sample_path
        
        if target_raw_sample.exists():
            if st.button("🧪 Gunakan Data Sampel", use_container_width=True, help=f"Muat data sampel {target_raw_sample.name}"):
                st.session_state["mode2_use_sample"] = True

    raw_file = st.file_uploader(
        "📂 Unggah Komentar Mentah YouTube API (CSV):",
        type=["csv"],
        key="mode2_main_uploader",
        label_visibility="collapsed"
    )

    df_raw = None

    if raw_file is not None:
        st.session_state["mode2_use_sample"] = False
        try:
            df_raw = pd.read_csv(raw_file, encoding="utf-8-sig")
            st.success(f"✅ Berhasil membaca berkas mentah: `{raw_file.name}` ({len(df_raw):,} baris data)")
        except Exception as e:
            st.error(f"❌ Gagal membaca file CSV mentah: {e}")
            st.stop()
    elif st.session_state.get("mode2_use_sample", False) and target_raw_sample.exists():
        try:
            df_raw = pd.read_csv(target_raw_sample, encoding="utf-8-sig")
            st.info(f"💡 Menggunakan Data Sampel Bawaan: `{target_raw_sample.name}` ({len(df_raw):,} baris data)")
        except Exception as e:
            st.error(f"❌ Gagal membaca file data sampel: {e}")
            st.stop()
    else:
        st.info("👆 Silakan unggah berkas CSV mentah Anda atau klik **Gunakan Data Sampel** untuk pengujian inferensi.")
        st.stop()

    st.subheader("Pratinjau Teks Mentah")
    st.dataframe(df_raw.head(5), use_container_width=True)

    if st.button("Jalankan Pelabelan Otomatis (XLM-RoBERTa Engine)"):
        with st.spinner("Memuat model XLM-RoBERTa & menyiapkan pipeline..."):
            xlm_engine = get_xlm_engine()

        st.info("Memproses pembersihan teks (text cleaner) & inferensi multi-head...")
        
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

        st.success("Pelabelan Otomatis XLM-RoBERTa Selesai!")

        # Standardize output columns WITHOUT human_* or rule_* engine columns
        df_final_output = processor.standardize_output_dataframe(df_raw)

        st.subheader("📋 Hasil Pelabelan Otomatis (Cuplikan 10 Baris)")
        st.dataframe(df_final_output.head(10), use_container_width=True)

        # Download Button
        csv_export = df_final_output.to_csv(index=False, encoding="utf-8-sig")
        
        st.markdown("---")
        st.success("💡 **Petunjuk Langkah Selanjutnya:** Unduh berkas CSV berlabel di bawah ini, kemudian Anda dapat beralih ke **Dashboard** di sidebar untuk langsung menganalisis hasilnya!")
        
        st.download_button(
            label="📥 Unduh CSV Berlabel (Standard 4 Aspek)",
            data=csv_export,
            file_name="yt_comments_xlmroberta_labeled.csv",
            mime="text/csv"
        )
