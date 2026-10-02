import pandas as pd
import numpy as np
from pathlib import Path
from src.preprocessing.text_cleaner import TextCleaner

ASPECTS = ['infra', 'ekonomi', 'kualitas', 'purnajual']
ASPECT_NAMES = {
    'infra_sentiment': 'Infrastruktur',
    'ekonomi_sentiment': 'Ekonomi & Harga',
    'kualitas_sentiment': 'Kualitas & Durabilitas',
    'purnajual_sentiment': 'Purnajual & Layanan'
}

STANDARD_COLUMNS = [
    "comment_id", "video_id", "user_id_hash", "text_original", "like_count",
    "published_at", "updated_at", "extracted_at", "text_cleaned",
    "infra_sentiment", "ekonomi_sentiment", "kualitas_sentiment", "purnajual_sentiment"
]


class DashboardDataProcessor:
    def __init__(self):
        self.cleaner = TextCleaner()

    def clean_text_series(self, text_series):
        return text_series.astype(str).apply(self.cleaner.clean_text)

    def calculate_saas_metrics(self, df):
        total_comments = len(df)
        
        # Videos analyzed
        if "video_id" in df.columns:
            videos_analyzed = df["video_id"].nunique(dropna=True)
        else:
            videos_analyzed = 1

        aspect_cols = [f"{a}_sentiment" for a in ASPECTS if f"{a}_sentiment" in df.columns]
        
        if aspect_cols:
            # Check aspect bearing comments
            has_aspect_mask = df[aspect_cols].notnull().any(axis=1)
            aspect_bearing_count = has_aspect_mask.sum()
            aspect_coverage_pct = (aspect_bearing_count / total_comments * 100) if total_comments > 0 else 0.0

            # Count aspects frequency
            aspect_counts = {a_name: df[a_col].notnull().sum() for a_col, a_name in ASPECT_NAMES.items() if a_col in df.columns}
            top_aspect = max(aspect_counts, key=aspect_counts.get) if aspect_counts else "-"

            # Overall sentiment breakdown
            all_sentiments = []
            for col in aspect_cols:
                valid_sents = df[col].dropna().astype(str).str.lower().tolist()
                valid_sents = [s for s in valid_sents if s not in ['none', 'nan', 'null', '']]
                all_sentiments.extend(valid_sents)

            if all_sentiments:
                sent_series = pd.Series(all_sentiments)
                top_sent = sent_series.mode()[0].capitalize() if not sent_series.empty else "Netral"
                pos_count = (sent_series == "positif").sum()
                neu_count = (sent_series == "netral").sum()
                neg_count = (sent_series == "negatif").sum()
                total_labels = len(all_sentiments)
            else:
                top_sent = "N/A"
                pos_count = neu_count = neg_count = total_labels = 0

        else:
            aspect_coverage_pct = 0.0
            top_aspect = "-"
            top_sent = "-"
            pos_count = neu_count = neg_count = total_labels = 0

        return {
            "total_comments": total_comments,
            "videos_analyzed": videos_analyzed,
            "aspect_coverage_pct": aspect_coverage_pct,
            "top_aspect": top_aspect,
            "top_sentiment": top_sent,
            "pos_count": pos_count,
            "neu_count": neu_count,
            "neg_count": neg_count,
            "total_labels": total_labels
        }

    def prepare_aspect_sentiment_df(self, df):
        records = []
        for col, name in ASPECT_NAMES.items():
            if col in df.columns:
                s_counts = df[col].astype(str).str.lower().value_counts()
                for sent in ['positif', 'netral', 'negatif']:
                    records.append({
                        'Aspek': name,
                        'Sentimen': sent.capitalize(),
                        'Jumlah': int(s_counts.get(sent, 0))
                    })
        return pd.DataFrame(records)

    def filter_top_liked_comments(self, df, aspect_col="kualitas_sentiment", sentiment="positif", top_n=5):
        if aspect_col not in df.columns:
            return pd.DataFrame()

        filtered = df[df[aspect_col].astype(str).str.lower() == sentiment.lower()].copy()
        
        if "like_count" in filtered.columns:
            filtered["like_count"] = pd.to_numeric(filtered["like_count"], errors="coerce").fillna(0).astype(int)
            filtered = filtered.sort_values(by="like_count", ascending=False)

        return filtered.head(top_n)

    def standardize_output_dataframe(self, df):
        # Format clean standard dataset output without human_* or rule_* engine columns
        output_df = df.copy()
        
        # Ensure text_cleaned exists
        if "text_cleaned" not in output_df.columns and "text_original" in output_df.columns:
            output_df["text_cleaned"] = self.clean_text_series(output_df["text_original"])

        # Pick only standard columns if present, plus any unknown non-internal columns
        final_cols = [c for c in STANDARD_COLUMNS if c in output_df.columns]
        extra_cols = [c for c in output_df.columns if c not in STANDARD_COLUMNS and not c.startswith("human_") and not c.startswith("rule_") and c not in ["strat_key", "strat_label", "has_aspect", "aspect_count"]]
        
        return output_df[final_cols + extra_cols]

    def prepare_overall_sentiment_pie_df(self, df):
        aspect_cols = [f"{a}_sentiment" for a in ASPECTS if f"{a}_sentiment" in df.columns]
        all_sentiments = []
        for col in aspect_cols:
            valid_sents = df[col].dropna().astype(str).str.lower().tolist()
            valid_sents = [s.capitalize() for s in valid_sents if s in ['positif', 'netral', 'negatif']]
            all_sentiments.extend(valid_sents)
        
        if not all_sentiments:
            return pd.DataFrame([{"Sentimen": "Positif", "Jumlah": 0}, {"Sentimen": "Netral", "Jumlah": 0}, {"Sentimen": "Negatif", "Jumlah": 0}])

        s_counts = pd.Series(all_sentiments).value_counts()
        records = [
            {"Sentimen": "Positif", "Jumlah": int(s_counts.get("Positif", 0))},
            {"Sentimen": "Netral", "Jumlah": int(s_counts.get("Netral", 0))},
            {"Sentimen": "Negatif", "Jumlah": int(s_counts.get("Negatif", 0))},
        ]
        return pd.DataFrame(records)

    def prepare_cooccurrence_matrix(self, df):
        aspect_cols = [f"{a}_sentiment" for a in ASPECTS]
        aspect_display_names = ['Infrastruktur', 'Ekonomi', 'Kualitas', 'Purnajual']
        
        matrix = np.zeros((4, 4), dtype=int)
        for i, col1 in enumerate(aspect_cols):
            for j, col2 in enumerate(aspect_cols):
                if col1 in df.columns and col2 in df.columns:
                    mask1 = df[col1].notnull() & ~df[col1].astype(str).str.lower().isin(['none', 'nan', ''])
                    mask2 = df[col2].notnull() & ~df[col2].astype(str).str.lower().isin(['none', 'nan', ''])
                    matrix[i, j] = int((mask1 & mask2).sum())
                    
        return matrix, aspect_display_names

    def prepare_word_count_distribution_df(self, df):
        text_col = "text_cleaned" if "text_cleaned" in df.columns else ("text_original" if "text_original" in df.columns else None)
        if not text_col:
            return pd.DataFrame()

        df_temp = df.copy()
        df_temp["word_count"] = df_temp[text_col].fillna("").astype(str).apply(lambda x: len(x.split()))
        
        records = []
        for col, name in ASPECT_NAMES.items():
            if col in df_temp.columns:
                sub = df_temp[df_temp[col].notnull()].copy()
                sub["sent"] = sub[col].astype(str).str.lower()
                for _, row in sub.iterrows():
                    if row["sent"] in ["positif", "netral", "negatif"]:
                        records.append({
                            "Sentimen": row["sent"].capitalize(),
                            "Jumlah Kata": int(row["word_count"]),
                            "Aspek": name
                        })
                        
        return pd.DataFrame(records)

