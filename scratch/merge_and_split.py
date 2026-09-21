import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from pathlib import Path
from src.labeling.dataset_splitter import DatasetSplitter

def main():
    audit_path = Path('data/interim/human_audit_sample.xlsx')
    auto_path = Path('data/interim/auto_labeled_comments_all.csv')
    
    print(f"Loading {audit_path}...")
    audit_df = pd.read_excel(audit_path)
    print(f"Loading {auto_path}...")
    auto_df = pd.read_csv(auto_path, encoding='utf-8-sig')

    aspects = ['infra', 'ekonomi', 'kualitas', 'purnajual']
    typo_map = {
        'psoitif': 'positif',
        'negtaif': 'negatif',
        'positif': 'positif',
        'negatif': 'negatif',
        'netral': 'netral'
    }

    # Standardize machine labels in auto_df first
    for asp in aspects:
        col = f'{asp}_sentiment'
        if col in auto_df.columns:
            auto_df[col] = auto_df[col].apply(
                lambda x: typo_map.get(str(x).strip().lower(), str(x).strip().lower()) 
                if pd.notnull(x) and str(x).strip() != '' and str(x).lower() != 'nan' else np.nan
            )

    audit_dict = audit_df.set_index('comment_id').to_dict('index')
    
    overridden_comments = 0
    total_aspects_overridden = 0

    for idx, row in auto_df.iterrows():
        c_id = row['comment_id']
        if c_id in audit_dict:
            h_row = audit_dict[c_id]
            has_any_human = any(
                pd.notnull(h_row.get(f'human_{asp}')) and str(h_row.get(f'human_{asp}')).strip() != '' and str(h_row.get(f'human_{asp}')).lower() != 'nan'
                for asp in aspects
            )
            if has_any_human:
                overridden_comments += 1
                auto_df.at[idx, 'label_source'] = 'Human-Audit-GoldStandard'
                for asp in aspects:
                    raw_h = h_row.get(f'human_{asp}')
                    if pd.notnull(raw_h) and str(raw_h).strip() != '' and str(raw_h).lower() != 'nan':
                        norm_h = typo_map.get(str(raw_h).strip().lower(), str(raw_h).strip().lower())
                        auto_df.at[idx, f'{asp}_sentiment'] = norm_h
                        total_aspects_overridden += 1
                    else:
                        auto_df.at[idx, f'{asp}_sentiment'] = np.nan

    print(f"Overrode {overridden_comments} comments ({total_aspects_overridden} aspect annotations) with Human Audit Gold Standard.")

    master_path = Path('data/interim/master_labeled_comments.csv')
    auto_df.to_csv(master_path, index=False, encoding='utf-8-sig')
    print(f"Saved master labeled comments to {master_path}")

    # Filter aspect-bearing comments (where at least 1 aspect is not null)
    aspect_cols = [f'{asp}_sentiment' for asp in aspects]
    valid_df = auto_df[auto_df[aspect_cols].notnull().any(axis=1)].copy()
    valid_path = Path('data/interim/valid_labeled_comments.csv')
    valid_df.to_csv(valid_path, index=False, encoding='utf-8-sig')
    print(f"Filtered {len(valid_df)} valid aspect-bearing comments out of {len(auto_df)} total comments.")

    # Print aspect distribution
    print("\nAspect Sentiment Breakdown (Valid Dataset):")
    for asp in aspects:
        col = f'{asp}_sentiment'
        counts = valid_df[col].value_counts(dropna=False).to_dict()
        print(f"  - Aspect '{asp}': {counts}")

    # Perform dataset split (70% train, 15% val, 15% test)
    print("\nSplitting dataset into train (70%), val (15%), test (15%)...")
    train_path, val_path, test_path = DatasetSplitter.split_dataset(
        input_csv_path=str(valid_path),
        output_dir="data/processed",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )

    # Read back to verify
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print("\nVerification of split data:")
    print(f"  Train: {len(train_df)} rows ({len(train_df)/len(valid_df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} rows ({len(val_df)/len(valid_df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} rows ({len(test_df)/len(valid_df)*100:.1f}%)")

if __name__ == '__main__':
    main()
