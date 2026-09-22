"""
Dataset Splitter Module (Train / Val Split 70/30 with Stratified Multi-Aspect Splitting)
Spec Compliance: specs/02_preprocessing_labeling.spec.md & specs/03_model_training.spec.md
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DatasetSplitter")


class DatasetSplitter:
    """
    Modul Pembagi Dataset ke Train (70%) dan Validation (30%)
    menggunakan Stratified Multi-Aspect Splitting dengan Random Seed = 42.
    """

    @staticmethod
    def split_dataset(
        input_csv_path: str,
        output_dir: str = "data/processed",
        train_ratio: float = 0.70,
        val_ratio: float = 0.30,
        random_seed: int = 42,
    ) -> tuple[str, str]:
        """
        Membaca CSV terlabel final 100% Pure Human Gold Standard,
        membagi 70/30 dengan Stratified Splitting, dan menyimpan train.csv dan val.csv.
        """
        assert abs((train_ratio + val_ratio) - 1.0) < 1e-5, "Rasio pembagian harus berjumlah 1.0!"

        input_path = Path(input_csv_path)
        if not input_path.exists():
            raise FileNotFoundError(f"File CSV input tidak ditemukan: {input_csv_path}")

        df = pd.read_csv(input_path, encoding="utf-8-sig")
        logger.info(f"Membaca {len(df)} total baris dataset terlabel...")

        # Membuat Composite Stratification Key dari 4 Aspek
        aspects = ['infra', 'ekonomi', 'kualitas', 'purnajual']
        aspect_cols = [f'{a}_sentiment' for a in aspects if f'{a}_sentiment' in df.columns]

        if len(aspect_cols) == 4:
            df['strat_key'] = (
                df[aspect_cols[0]].fillna('none').astype(str) + '_' +
                df[aspect_cols[1]].fillna('none').astype(str) + '_' +
                df[aspect_cols[2]].fillna('none').astype(str) + '_' +
                df[aspect_cols[3]].fillna('none').astype(str)
            )
            
            # Bin rare combination keys (< 3 samples) into 'rare_comb' for stable stratification
            counts = df['strat_key'].value_counts()
            rare_keys = set(counts[counts < 3].index)
            strat_labels = df['strat_key'].apply(lambda x: 'rare_comb' if x in rare_keys else x)
        else:
            strat_labels = None

        # Split: Train (70%) vs Val (30%)
        train_df, val_df = train_test_split(
            df,
            test_size=val_ratio,
            random_state=random_seed,
            shuffle=True,
            stratify=strat_labels,
        )

        # Drop temporary strat_key column
        for d in [train_df, val_df]:
            if 'strat_key' in d.columns:
                d.drop(columns=['strat_key'], inplace=True, errors='ignore')

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        train_path = out_dir / "train.csv"
        val_path = out_dir / "val.csv"
        test_path = out_dir / "test.csv"

        # Remove old test.csv if exists to prevent confusion
        if test_path.exists():
            test_path.unlink()

        train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
        val_df.to_csv(val_path, index=False, encoding="utf-8-sig")

        logger.info("SUKSES: Pembagian Dataset Stratified 70/30 Selesai Tanpa Data Leakage!")
        logger.info(f"- Train Set (70%): {len(train_df)} baris -> {train_path.resolve()}")
        logger.info(f"- Val Set   (30%): {len(val_df)} baris -> {val_path.resolve()}")

        return str(train_path.resolve()), str(val_path.resolve())


if __name__ == "__main__":
    print("Stratified Dataset Splitter 70/30 Module Loaded Successfully!")
