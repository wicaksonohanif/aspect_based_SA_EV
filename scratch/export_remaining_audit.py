import pandas as pd
import numpy as np
from pathlib import Path

def main():
    audit_sample_path = Path("data/interim/human_audit_sample.xlsx")
    master_path = Path("data/interim/master_labeled_comments.csv")
    
    print(f"Reading {audit_sample_path}...")
    sample_df = pd.read_excel(audit_sample_path)
    audited_ids = set(sample_df['comment_id'].dropna())
    
    print(f"Reading {master_path}...")
    master_df = pd.read_csv(master_path, encoding='utf-8-sig')

    # Filter remaining comments
    rem_df = master_df[~master_df['comment_id'].isin(audited_ids)].copy()
    print(f"Total remaining comments to audit: {len(rem_df)}")

    # Add human audit columns
    for col in ['human_infra', 'human_ekonomi', 'human_kualitas', 'human_purnajual']:
        rem_df[col] = np.nan

    # Sort so that aspect-bearing comments are at the top
    aspect_cols = ['infra_sentiment', 'ekonomi_sentiment', 'kualitas_sentiment', 'purnajual_sentiment']
    rem_df['has_aspect'] = rem_df[aspect_cols].notnull().any(axis=1)
    
    # Sort: has_aspect True first, then False
    rem_df = rem_df.sort_values(by=['has_aspect'], ascending=False).drop(columns=['has_aspect'])

    # Reorder columns to match human_audit_sample.xlsx
    target_cols = [
        'comment_id', 'video_id', 'user_id_hash', 'text_original', 'like_count', 
        'published_at', 'updated_at', 'extracted_at', 'text_cleaned', 
        'infra_sentiment', 'ekonomi_sentiment', 'kualitas_sentiment', 'purnajual_sentiment', 
        'label_source', 'human_infra', 'human_ekonomi', 'human_kualitas', 'human_purnajual'
    ]
    
    # Fill missing columns if any
    for col in target_cols:
        if col not in rem_df.columns:
            rem_df[col] = np.nan
            
    rem_df = rem_df[target_cols]

    out_excel = Path("data/interim/human_audit_remaining.xlsx")
    out_csv = Path("data/interim/human_audit_remaining.csv")

    rem_df.to_excel(out_excel, index=False)
    rem_df.to_csv(out_csv, index=False, encoding='utf-8-sig')

    print(f"SUCCESS: Exported remaining audit file to:")
    print(f"  - Excel: {out_excel.resolve()}")
    print(f"  - CSV:   {out_csv.resolve()}")

if __name__ == '__main__':
    main()
