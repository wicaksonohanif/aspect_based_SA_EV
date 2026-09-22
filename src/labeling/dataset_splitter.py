"""
Dataset Splitter Module (Train / Val / Test 70/15/15 with Stratified Multi-Aspect Splitting)
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
    Modul Pembagi Dataset ke Train (70%), Validation (15%), dan Test (15%)
    menggunakan Stratified Multi-Aspect Splitting dengan Random Seed = 42.
    """

    @staticmethod
    def split_dataset(
        input_csv_path: str,
        output_dir: str = "data/processed",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ) -> tuple[str, str, str]:
        """
        Membaca CSV terlabel final, membagi 70/15/15 dengan Stratified Splitting,
        dan menyimpan file train.csv, val.csv, test.csv.
        """
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Rasio pembagian harus berjumlah 1.0!"

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
            
            # Bin rare combination keys (< 10 samples) into 'rare_comb' for stable stratification
            counts = df['strat_key'].value_counts()
            rare_keys = set(counts[counts < 10].index)
            strat_labels = df['strat_key'].apply(lambda x: 'rare_comb' if x in rare_keys else x)
        else:
            strat_labels = None

        # Split 1: Train (70%) vs (Val + Test 30%)
        temp_ratio = val_ratio + test_ratio
        train_df, temp_df = train_test_split(
            df,
            test_size=temp_ratio,
            random_state=random_seed,
            shuffle=True,
            stratify=strat_labels,
        )

        # Split 2: Val (15%) vs Test (15%)
        if strat_labels is not None:
            temp_strat = strat_labels.loc[temp_df.index]
            # If any class in temp_strat has < 2 samples, fallback to rare_comb for temp split
            t_counts = temp_strat.value_counts()
            t_rare = set(t_counts[t_counts < 2].index)
            if t_rare:
                temp_strat = temp_strat.apply(lambda x: 'rare_comb_temp' if x in t_rare else x)
        else:
            temp_strat = None

        relative_test_ratio = test_ratio / temp_ratio
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_ratio,
            random_state=random_seed,
            shuffle=True,
            stratify=temp_strat,
        )

        # Drop temporary strat_key column
        for d in [train_df, val_df, test_df]:
            if 'strat_key' in d.columns:
                d.drop(columns=['strat_key'], inplace=True, errors='ignore')

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        train_path = out_dir / "train.csv"
        val_path = out_dir / "val.csv"
        test_path = out_dir / "test.csv"

        train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
        val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
        test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

        logger.info("SUKSES: Pembagian Dataset Stratified Selesai Tanpa Data Leakage!")
        logger.info(f"- Train Set (70%): {len(train_df)} baris -> {train_path.resolve()}")
        logger.info(f"- Val Set   (15%): {len(val_df)} baris -> {val_path.resolve()}")
        logger.info(f"- Test Set  (15%): {len(test_df)} baris -> {test_path.resolve()}")

        return str(train_path.resolve()), str(val_path.resolve()), str(test_path.resolve())


if __name__ == "__main__":
    print("Stratified Dataset Splitter Module Loaded Successfully!")
