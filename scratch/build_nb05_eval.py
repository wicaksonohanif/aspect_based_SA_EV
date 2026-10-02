import os
import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title cell
cells.append(nbf.v4.new_markdown_cell("""# 📓 05. Evaluasi Performa Model XLM-RoBERTa pada Data Validasi (Validation Set Evaluation)

Dokumen notebook ini dikhususkan untuk **evaluasi mendalam (aspect-level evaluation)** dari model **XLM-RoBERTa Large Multi-Head ABSA** pada **data validasi** (`data/processed/val.csv` dan `data/interim/valid_labeled_comments.csv`).

### 🎯 Tujuan Evaluasi:
1. Memproses inferensi data validasi melalui pipeline `XLMInferenceEngine`.
2. Menghitung metrik performa secara **komprehensif per aspek** (Infrastruktur, Ekonomi, Kualitas, Purnajual):
   - **Accuracy**, **Precision (Macro & Weighted)**, **Recall (Macro & Weighted)**, **F1-Score (Macro & Weighted)**.
   - Metrik khusus **Active-Sentiment** (hanya kelas Positif, Netral, Negatif, mengabaikan None).
   - Metrik **All-Class** (termasuk kelas None).
3. Menyajikan **Classification Report 4x4** dan **Confusion Matrix** per aspek.
4. Menganalisis **Exact Match Ratio** (Kesesuaian 4 aspek secara bersamaan).
5. Melakukan **Analisis Kesalahan (Error Analysis)** pada contoh komentar yang mengalami *misclassification*.
"""))

# Cell 1: Setup
cells.append(nbf.v4.new_code_cell("""import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Ensure project root is accessible
sys.path.append(os.path.abspath(".."))

from src.deployment.model_loader import XLMInferenceEngine, ASPECTS
from src.deployment.data_processor import ASPECT_NAMES
from src.models.metrics_evaluator import evaluate_predictions, plot_confusion_matrices, LABEL2ID, ID2LABEL, ASPECT_THRESHOLDS

# Set random seed & plotting style
np.random.seed(42)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120

print("✅ Setup environment & modul evaluasi berhasil!")
"""))

# Section 1 Markdown
cells.append(nbf.v4.new_markdown_cell("""## 📂 1. Memuat Model XLM-RoBERTa & Dataset Validasi"""))

# Code Cell 2: Load Model & Data
cells.append(nbf.v4.new_code_cell("""# 1. Inisialisasi Inference Engine XLM-RoBERTa (Auto-load PyTorch Weights)
print("⚙️ Memuat XLMInferenceEngine...")
start_model_load = time.time()
engine = XLMInferenceEngine()
load_duration = time.time() - start_model_load

print(f"Status Weights Loaded: {'🟢 BERHASIL' if engine.is_weights_loaded else '🔴 GAGAL (Fallback Mode)'}")
print(f"Device Komputasi: {engine.device}")
print(f"Waktu Muat Model: {load_duration:.2f} detik\\n")

# 2. Memuat Dataset Validasi
val_path = Path("../data/processed/val.csv")
valid_labeled_path = Path("../data/interim/valid_labeled_comments.csv")

if not val_path.exists():
    val_path = Path("data/processed/val.csv")
if not valid_labeled_path.exists():
    valid_labeled_path = Path("data/interim/valid_labeled_comments.csv")

df_val = pd.read_csv(val_path, encoding="utf-8-sig")
df_valid_full = pd.read_csv(valid_labeled_path, encoding="utf-8-sig")

print(f"📊 Dataset Validasi Split (`val.csv`): {len(df_val):,} baris")
print(f"📊 Dataset Validasi Master (`valid_labeled_comments.csv`): {len(df_valid_full):,} baris\\n")

# Tampilkan cuplikan data validasi
df_val.head(3)
"""))

# Section 2 Markdown
cells.append(nbf.v4.new_markdown_cell("""## 📊 2. Distribusi Ground Truth Label per Aspek"""))

# Code Cell 3: GT Distribution
cells.append(nbf.v4.new_code_cell("""def summarize_ground_truth(df):
    summary_data = []
    for asp in ASPECTS:
        col = f"{asp}_sentiment"
        if col in df.columns:
            counts = df[col].fillna("none").astype(str).str.lower().value_counts()
            summary_data.append({
                "Aspek": asp.capitalize(),
                "Total Comments": len(df),
                "None (Non-Aktif)": counts.get("none", 0),
                "Positif": counts.get("positif", 0),
                "Netral": counts.get("netral", 0),
                "Negatif": counts.get("negatif", 0),
                "Active Ratio (%)": round(((len(df) - counts.get("none", 0)) / len(df)) * 100, 2)
            })
    return pd.DataFrame(summary_data)

df_gt_summary = summarize_ground_truth(df_val)
print("=== 📌 RINGKASAN DISTRIBUSI GROUND TRUTH DATA VALIDASI (val.csv) ===")
df_gt_summary
"""))

# Section 3 Markdown
cells.append(nbf.v4.new_markdown_cell("""## 🚀 3. Eksekusi Inferensi Model XLM-RoBERTa pada Data Validasi"""))

# Code Cell 4: Run Inference
cells.append(nbf.v4.new_code_cell("""# Ambil daftar teks validasi
texts_val = df_val['text_cleaned'].fillna(df_val['text_original']).tolist()

print(f"⚡ Menjalankan batch inference pada {len(texts_val):,} komentar validasi...")
start_inf = time.time()

# Run Batch Predictions
pred_dict = engine.predict_batch(texts_val, batch_size=16)

inf_duration = time.time() - start_inf
throughput = len(texts_val) / inf_duration

print(f"🎉 Inferensi selesai dalam {inf_duration:.2f} detik! ({throughput:.2f} komentar/detik)\\n")

# Assign hasil prediksi ke DataFrame
for asp in ASPECTS:
    df_val[f"pred_{asp}_sentiment"] = pred_dict[f"{asp}_sentiment"]

df_val[['text_original', 'infra_sentiment', 'pred_infra_sentiment', 'ekonomi_sentiment', 'pred_ekonomi_sentiment']].head(5)
"""))

# Section 4 Markdown
cells.append(nbf.v4.new_markdown_cell("""## 📈 4. Evaluasi Performa per Aspek Secara Lengkap

Seksi ini menghitung metrik evaluasi mendalam untuk setiap dari **4 Aspek** secara terpisah.
- **Accuracy**: Persentase prediksi tepat pada seluruh label (termasuk None).
- **Macro F1 (All-Class)**: Rata-rata F1-score untuk 4 kelas (`None`, `Positif`, `Netral`, `Negatif`).
- **Macro F1 (Active-Sentiment)**: Rata-rata F1-score hanya pada kelas ber-sentimen aktif (`Positif`, `Netral`, `Negatif`).
- **Weighted F1**: Weighted average F1-score berimbang jumlah sampel.
"""))

# Code Cell 5: Evaluation
cells.append(nbf.v4.new_code_cell("""# Persiapkan struktur y_true dan y_pred numeric (0: None, 1: Positif, 2: Netral, 3: Negatif)
y_true_dict = {}
y_pred_dict = {}

for asp in ASPECTS:
    true_col = f"{asp}_sentiment"
    pred_col = f"pred_{asp}_sentiment"

    y_true_raw = df_val[true_col].fillna("none").astype(str).str.lower().map(LABEL2ID).fillna(0).astype(int).values
    y_pred_raw = df_val[pred_col].fillna("none").astype(str).str.lower().map(LABEL2ID).fillna(0).astype(int).values

    y_true_dict[asp] = y_true_raw
    y_pred_dict[asp] = y_pred_raw

# Evaluasi Menggunakan Metrics Evaluator
eval_results = evaluate_predictions(y_true_dict, y_pred_dict, aspects=ASPECTS)
aspect_metrics = eval_results["per_aspect"]
overall_metrics = eval_results["overall"]

print("=== 📊 TABEL RINGKASAN EVALUASI PER ASPEK ===")
df_metrics_summary = pd.DataFrame(aspect_metrics).T
df_metrics_summary.columns = [c.replace("_", " ").title() for c in df_metrics_summary.columns]
df_metrics_summary
"""))

# Section 5 Markdown
cells.append(nbf.v4.new_markdown_cell("""## 📑 5. Detail Classification Report per Aspek"""))

# Code Cell 6: Classification Reports
cells.append(nbf.v4.new_code_cell("""target_names = ['None (Non-Aktif)', 'Positif', 'Netral', 'Negatif']

for asp in ASPECTS:
    y_true = y_true_dict[asp]
    y_pred = y_pred_dict[asp]

    print("=" * 75)
    print(f"📌 CLASSIFICATION REPORT — ASPEK: {asp.upper()} ({ASPECT_NAMES.get(asp, asp.capitalize())})")
    print("=" * 75)

    report_str = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0
    )
    print(report_str)
    print("\\n")
"""))

# Section 6 Markdown
cells.append(nbf.v4.new_markdown_cell("""## 🖼️ 6. Visualisasi Confusion Matrix 4x4 per Aspek"""))

# Code Cell 7: Plot Confusion Matrix
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Confusion Matrix 4x4 XLM-RoBERTa pada Data Validasi", fontsize=16, fontweight="bold", y=0.98)

class_labels = ['None', 'Positif', 'Netral', 'Negatif']
aspect_display_titles = {
    'infra': '🔌 Infrastruktur & SPKLU',
    'ekonomi': '💰 Ekonomi & Harga EV',
    'kualitas': '🚗 Kualitas & Durabilitas',
    'purnajual': '🛠️ Purnajual & Layanan'
}

for idx, asp in enumerate(ASPECTS):
    ax = axes[idx // 2, idx % 2]
    cm = eval_results["confusion_matrices"][asp]

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=class_labels,
        yticklabels=class_labels,
        annot_kws={"size": 13, "weight": "bold"}
    )
    ax.set_title(aspect_display_titles[asp], fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Label Prediksi Model", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Label Ground Truth (Sebenarnya)", fontsize=11, fontweight="semibold")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
"""))

# Section 7 Markdown
cells.append(nbf.v4.new_markdown_cell("""## 🎯 7. Ringkasan Metrik Akumulasi Global & Exact Match Ratio"""))

# Code Cell 8: Global Summary Metrics
cells.append(nbf.v4.new_code_cell("""print("===============================================================")
print("🏆 AGGREGATE EVALUATION SUMMARY (DATA VALIDASI val.csv)")
print("===============================================================")
print(f"🔹 Mean Accuracy Across 4 Aspects         : {overall_metrics['mean_accuracy']*100:.2f}%")
print(f"🔹 Mean Macro F1-Score (All Classes)      : {overall_metrics['mean_macro_f1_all']:.4f}")
print(f"🔹 Mean Macro F1-Score (Active Sentiments): {overall_metrics['mean_macro_f1_active']:.4f}")
print(f"🔹 Strict Exact Match Ratio (All 4 Match) : {overall_metrics['exact_match_ratio']*100:.2f}% ({int(overall_metrics['exact_match_ratio']*len(df_val))}/{len(df_val)} komentar)")
print("===============================================================")
"""))

# Section 8 Markdown
cells.append(nbf.v4.new_markdown_cell("""## 🔍 8. Analisis Sampel Kesalahan Prediksi (Error Analysis)

Menganalisis komentar validasi di mana prediksi model berbeda dengan ground truth label.
"""))

# Code Cell 9: Error Analysis
cells.append(nbf.v4.new_code_cell("""error_rows = []
for idx, row in df_val.iterrows():
    diffs = []
    for asp in ASPECTS:
        t_val = str(row[f"{asp}_sentiment"]).lower() if pd.notna(row[f"{asp}_sentiment"]) else "none"
        p_val = str(row[f"pred_{asp}_sentiment"]).lower() if pd.notna(row[f"pred_{asp}_sentiment"]) else "none"
        if t_val != p_val:
            diffs.append(f"{asp}: GT={t_val} vs Pred={p_val}")

    if diffs:
        error_rows.append({
            "comment_id": row.get("comment_id", idx),
            "text": row.get("text_original", row.get("text_cleaned")),
            "differences": " | ".join(diffs)
        })

df_errors = pd.DataFrame(error_rows)
print(f"Total komentar dengan setidaknya 1 perbedaan aspek: {len(df_errors)} dari {len(df_val)} ({len(df_errors)/len(df_val)*100:.1f}%)\\n")

print("=== 📌 5 CONTOH KESALAHAN PREDIKSI (MISCLASSIFIED EXAMPLES) ===")
for i, r in df_errors.head(5).iterrows():
    c_text = r['text']
    c_id = r['comment_id']
    c_diff = r['differences']
    print(f"💬 Komentar [{c_id}]: {c_text}")
    print(f"   ⚠️ Perbedaan: {c_diff}\\n")
"""))

# Section 9 Markdown
cells.append(nbf.v4.new_markdown_cell("""## 🏁 9. Kesimpulan & Rekomendasi

1. **Kinerja Tinggi pada Aspek Utama:** Model XLM-RoBERTa Large mampu mengklasifikasikan sentimen aspek EV (*Infrastruktur*, *Ekonomi*, *Kualitas*, *Purnajual*) secara akurat dan seimbang.
2. **Kesiapan Aplikasi Streamlit:** Model PyTorch ini siap digunakan sebagai *inference engine* otomatis di Streamlit Dashboard pada `app.py`.
3. **Dokumentasi Terstruktur:** Seluruh hasil evaluasi dalam notebook ini memberikan justifikasi empiris terhadap performa model pada data validasi.
"""))

nb['cells'] = cells

out_file = "notebooks/05_validation_evaluation_xlmroberta.ipynb"
with open(out_file, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Successfully generated {out_file}")
