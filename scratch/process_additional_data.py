import sys
import os
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.abspath("."))

from src.extraction.extractor import YouTubeCommentExtractor
from src.preprocessing.text_cleaner import TextCleaner
from src.labeling.weak_supervision import WeakSupervisionEngine

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ProcessAdditional")

NEW_VIDEO_URLS = [
    # tambahan infra
    "https://www.youtube.com/watch?v=jL5p1zZMNEM",
    "https://www.youtube.com/watch?v=O-vAYJNfQEk",
    "https://www.youtube.com/watch?v=4f36QI6CVtc",
    "https://www.youtube.com/watch?v=l8fPSUqdmOM",
    "https://www.youtube.com/watch?v=khAUeXPkWvg",
    # tambahan aftersales
    "https://www.youtube.com/watch?v=AOjceFDyYeM",
    "https://www.youtube.com/watch?v=IMO53AFzSZU",
    "https://www.youtube.com/watch?v=0fHm0HIIJrY",
    "https://www.youtube.com/watch?v=9GTZD3la5qI",
    "https://www.youtube.com/watch?v=W3G8V7xV3-Y",
]

RAW_CSV = "data/raw/yt_comments_additional_20260927_210207.csv"

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Loading / Extracting Raw Comments
    if Path(RAW_CSV).exists():
        logger.info(f"=== STEP 1: Membaca Komentar Mentah dari {RAW_CSV} ===")
        df_raw = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
    else:
        logger.info("=== STEP 1: Ekstraksi Komentar YouTube Baru ===")
        extractor = YouTubeCommentExtractor()
        comments = extractor.extract_from_video_list(NEW_VIDEO_URLS, max_comments_per_video=500)
        df_raw = pd.DataFrame(comments)
        df_raw = df_raw.drop_duplicates(subset=["comment_id"]).reset_index(drop=True)
        raw_csv_path = f"data/raw/yt_comments_additional_{timestamp}.csv"
        df_raw.to_csv(raw_csv_path, index=False, encoding="utf-8-sig")

    logger.info(f"Total komentar mentah unik: {len(df_raw)}")

    # 2. Preprocessing / Text Cleaning
    logger.info("=== STEP 2: Pembersihan Teks (Preprocessing) ===")
    cleaner = TextCleaner()
    df_raw["text_cleaned"] = df_raw["text_original"].astype(str).apply(cleaner.clean_text)
    
    df_clean = df_raw[df_raw["text_cleaned"].str.strip() != ""].copy().reset_index(drop=True)
    logger.info(f"Total komentar setelah cleaning: {len(df_clean)}")
    
    # 3. Labeling dengan Rule-Based ONLY (use_gemini=False)
    logger.info("=== STEP 3: Pelabelan Aspek & Sentimen (Rule-Based Murni) ===")
    engine = WeakSupervisionEngine(use_gemini=False)
    
    results = []
    for idx, row in df_clean.iterrows():
        text = str(row["text_cleaned"])
        labeled_data = engine.label_single_text(text)
        record = row.to_dict()
        record.update(labeled_data)
        results.append(record)
        
    df_labeled = pd.DataFrame(results)
    
    # 4. Tambah kolom Human Audit untuk validasi manual
    for aspect in ["infra", "ekonomi", "kualitas", "purnajual"]:
        df_labeled[f"human_{aspect}"] = None
        
    # 5. Sorting: Prioritaskan komentar yang memiliki setidaknya 1 label aspek
    def check_has_aspect(row):
        aspect_cols = ["infra_sentiment", "ekonomi_sentiment", "kualitas_sentiment", "purnajual_sentiment"]
        for col in aspect_cols:
            val = row.get(col)
            if pd.notna(val) and val is not None and str(val).strip() != "" and str(val).lower() not in ["none", "nan"]:
                return True
        return False
        
    def count_aspects(row):
        aspect_cols = ["infra_sentiment", "ekonomi_sentiment", "kualitas_sentiment", "purnajual_sentiment"]
        cnt = 0
        for col in aspect_cols:
            val = row.get(col)
            if pd.notna(val) and val is not None and str(val).strip() != "" and str(val).lower() not in ["none", "nan"]:
                cnt += 1
        return cnt
        
    df_labeled["has_aspect"] = df_labeled.apply(check_has_aspect, axis=1)
    df_labeled["aspect_count"] = df_labeled.apply(count_aspects, axis=1)
    
    # Sort: has_aspect (True first), aspect_count descending, like_count descending
    df_sorted = df_labeled.sort_values(
        by=["has_aspect", "aspect_count", "like_count"],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    
    # Drop temporary helper columns for final export
    df_export = df_sorted.drop(columns=["has_aspect", "aspect_count"])
    
    desired_columns = [
        "comment_id", "video_id", "user_id_hash", "text_original", "like_count",
        "published_at", "updated_at", "extracted_at", "text_cleaned",
        "infra_sentiment", "ekonomi_sentiment", "kualitas_sentiment", "purnajual_sentiment",
        "label_source", "human_infra", "human_ekonomi", "human_kualitas", "human_purnajual"
    ]
    
    final_cols = [c for c in desired_columns if c in df_export.columns]
    remaining_cols = [c for c in df_export.columns if c not in final_cols]
    df_export = df_export[final_cols + remaining_cols]
    
    excel_path = "data/interim/human_audit_additional.xlsx"
    csv_interim_path = f"data/interim/human_audit_additional_{timestamp}.csv"
    
    df_export.to_excel(excel_path, index=False, engine="openpyxl")
    df_export.to_csv(csv_interim_path, index=False, encoding="utf-8-sig")
    
    logger.info("=== SELESAI ===")
    logger.info(f"Total komentar diekspor ke Excel: {len(df_export)}")
    logger.info(f"Komentar beraspek: {df_labeled['has_aspect'].sum()} | Komentar tanpa aspek: {(~df_labeled['has_aspect']).sum()}")
    logger.info(f"Excel file: {os.path.abspath(excel_path)}")
    logger.info(f"CSV file: {os.path.abspath(csv_interim_path)}")

if __name__ == "__main__":
    main()
