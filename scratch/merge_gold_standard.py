import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from pathlib import Path
from src.labeling.dataset_splitter import DatasetSplitter

def main():
    sample_path = Path('data/interim/human_audit_sample.xlsx')
    remaining_path = Path('data/interim/human_audit_remaining.xlsx')
    auto_path = Path('data/interim/auto_labeled_comments_all.csv')

    aspects = ['infra', 'ekonomi', 'kualitas', 'purnajual']
    typo_map = {
        'psoitif': 'positif',
        'negtaif': 'negatif',
        'positif': 'positif',
        'negatif': 'negatif',
        'netral': 'netral',
        'neutral': 'netral',
        'positive': 'positif',
        'negative': 'negatif'
    }

    print(f"Loading base comments from {auto_path}...")
    auto_df = pd.read_csv(auto_path, encoding='utf-8-sig')

    # Collect human audits from sample and remaining sheets
    audit_dict = {}

    if sample_path.exists():
        print(f"Reading {sample_path}...")
        df1 = pd.read_excel(sample_path)
        for _, r in df1.iterrows():
            audit_dict[r['comment_id']] = r.to_dict()

    if remaining_path.exists():
        print(f"Reading {remaining_path}...")
        df2 = pd.read_excel(remaining_path)
        for _, r in df2.iterrows():
            c_id = r['comment_id']
            # Remaining overrides sample if present
            audit_dict[c_id] = r.to_dict()

    print(f"Total audit records collected: {len(audit_dict)}")

    # Clear machine aspect sentiments first for PURE human gold standard
    for asp in aspects:
        auto_df[f'{asp}_sentiment'] = np.nan
        auto_df[f'{asp}_sentiment'] = auto_df[f'{asp}_sentiment'].astype(object)
    auto_df['label_source'] = np.nan
    auto_df['label_source'] = auto_df['label_source'].astype(object)

    pure_human_count = 0
    total_aspects_count = 0

    for idx, row in auto_df.iterrows():
        c_id = row['comment_id']
        if c_id in audit_dict:
            h_row = audit_dict[c_id]
            
            # Check if user filled any human aspect sentiment
            has_any_human = False
            for asp in aspects:
                val = h_row.get(f'human_{asp}')
                if pd.notnull(val) and str(val).strip() != '' and str(val).strip().lower() != 'nan':
                    has_any_human = True
                    break

            if has_any_human:
                pure_human_count += 1
                auto_df.at[idx, 'label_source'] = 'Human-Audit-GoldStandard'
                for asp in aspects:
                    raw_h = h_row.get(f'human_{asp}')
                    if pd.notnull(raw_h) and str(raw_h).strip() != '' and str(raw_h).strip().lower() != 'nan':
                        clean_str = str(raw_h).strip().lower()
                        norm_h = typo_map.get(clean_str, clean_str)
                        auto_df.at[idx, f'{asp}_sentiment'] = norm_h
                        total_aspects_count += 1
                    else:
                        auto_df.at[idx, f'{asp}_sentiment'] = np.nan

    print(f"\n==================================================")
    print(f"PURE HUMAN GOLD STANDARD PROCESSING COMPLETE:")
    print(f"  - Pure Human Audited Comments: {pure_human_count}")
    print(f"  - Total Human Aspect Sentiment Labels: {total_aspects_count}")
    print(f"==================================================\n")

    master_path = Path('data/interim/master_labeled_comments.csv')
    auto_df.to_csv(master_path, index=False, encoding='utf-8-sig')
    print(f"Saved master labeled dataset to: {master_path.resolve()}")

    # Filter ONLY aspect-bearing comments that have Human Gold Standard labels
    aspect_cols = [f'{asp}_sentiment' for asp in aspects]
    valid_df = auto_df[auto_df['label_source'] == 'Human-Audit-GoldStandard'].copy()
    valid_df = valid_df[valid_df[aspect_cols].notnull().any(axis=1)].copy()
    
    valid_path = Path('data/interim/valid_labeled_comments.csv')
    valid_df.to_csv(valid_path, index=False, encoding='utf-8-sig')
    print(f"Filtered {len(valid_df)} PURE HUMAN GOLD STANDARD aspect-bearing comments out of {len(auto_df)} total comments.")

    # Aspect distribution
    print("\nAspect Sentiment Breakdown (100% Pure Human Gold Standard):")
    for asp in aspects:
        col = f'{asp}_sentiment'
        counts = valid_df[col].value_counts(dropna=False).to_dict()
        print(f"  - Aspect '{asp}': {counts}")

    # Re-split dataset 70/30 (Train / Val)
    print("\nSplitting pure human gold standard dataset into train (70%) and val (30%)...")
    train_path, val_path = DatasetSplitter.split_dataset(
        input_csv_path=str(valid_path),
        output_dir="data/processed",
        train_ratio=0.70,
        val_ratio=0.30,
        random_seed=42
    )

    print("\nCOMPLETE: Updated train.csv (70%) and val.csv (30%) in data/processed/")

if __name__ == '__main__':
    main()
