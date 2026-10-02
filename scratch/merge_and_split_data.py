import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys

sys.path.append(os.path.abspath("."))
from src.labeling.dataset_splitter import DatasetSplitter

def clean_sentiment_label(val):
    if pd.isna(val) or val is None:
        return np.nan
    val_str = str(val).strip().lower()
    if val_str in ['none', 'nan', 'null', '']:
        return np.nan
    # Fix typos
    if val_str in ['psoitif', 'potisif', 'positiif', 'positif']:
        return 'positif'
    if val_str in ['netral', 'netaral']:
        return 'netral'
    if val_str in ['negatif', 'negtif', 'negtaif', 'negatif ']:
        return 'negatif'
    return val_str

def main():
    sample_path = 'data/interim/human_audit_sample.xlsx'
    remaining_path = 'data/interim/human_audit_remaining.xlsx'
    additional_path = 'data/interim/human_audit_additional.xlsx'

    df_sample = pd.read_excel(sample_path)
    df_remaining = pd.read_excel(remaining_path)
    df_additional = pd.read_excel(additional_path)

    print(f"Loaded Sample    : {len(df_sample)} rows")
    print(f"Loaded Remaining : {len(df_remaining)} rows")
    print(f"Loaded Additional: {len(df_additional)} rows")

    # Combine all datasets
    df_all = pd.concat([df_sample, df_remaining, df_additional], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=['comment_id']).reset_index(drop=True)
    print(f"Total Combined Unique Rows: {len(df_all)}")

    aspects = ['infra', 'ekonomi', 'kualitas', 'purnajual']
    
    # Process human labels
    for a in aspects:
        human_col = f'human_{a}'
        sent_col = f'{a}_sentiment'
        if human_col in df_all.columns:
            # Clean human label
            df_all[human_col] = df_all[human_col].apply(clean_sentiment_label)
            # Override sent_col with human label if human label exists
            df_all[sent_col] = df_all[human_col].where(df_all[human_col].notnull(), df_all[sent_col].apply(clean_sentiment_label))

    # Identify rows that have at least 1 human audit label across the 4 aspects
    human_cols = [f'human_{a}' for a in aspects]
    df_all['has_human_audit'] = df_all[human_cols].notnull().any(axis=1)

    # Filter Valid Dataset (100% Pure Human Gold Standard)
    # Only rows with human audit labels
    df_valid = df_all[df_all['has_human_audit']].copy().reset_index(drop=True)
    
    # Update label_source for human audited ones
    df_valid['label_source'] = 'Human-Audit-GoldStandard'

    # Filter aspect sentiments in valid dataset: only keep aspect sentiments that came from human audit
    for a in aspects:
        human_col = f'human_{a}'
        sent_col = f'{a}_sentiment'
        df_valid[sent_col] = df_valid[human_col]

    print(f"Total Master Dataset Rows : {len(df_all)}")
    print(f"Total Valid (Pure Human) : {len(df_valid)}")

    for a in aspects:
        print(f"Valid {a}_sentiment counts:\n{df_valid[f'{a}_sentiment'].value_counts(dropna=False)}")

    # Save master and valid CSVs
    master_csv = 'data/interim/master_labeled_comments.csv'
    valid_csv = 'data/interim/valid_labeled_comments.csv'

    df_all.to_csv(master_csv, index=False, encoding='utf-8-sig')
    df_valid.to_csv(valid_csv, index=False, encoding='utf-8-sig')

    print(f"Saved master dataset to {master_csv}")
    print(f"Saved valid dataset to {valid_csv}")

    # Re-run train/val split 70/30
    train_path, val_path = DatasetSplitter.split_dataset(
        input_csv_path=valid_csv,
        output_dir="data/processed",
        train_ratio=0.70,
        val_ratio=0.30,
        random_seed=42,
    )
    print(f"Split done: {train_path}, {val_path}")

if __name__ == '__main__':
    main()
