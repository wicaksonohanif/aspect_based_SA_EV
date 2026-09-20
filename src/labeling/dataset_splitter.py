"""
Dataset Splitter Module (Train / Val / Test 70/15/15)
Spec Compliance: specs/02_preprocessing_labeling.spec.md
"""

import logging
from pathlib import Path
import pandas as pd
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
    dengan penguncian Random Seed = 42 dan pencegahan Data Leakage.
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
        Membaca CSV terlabel final, membagi 70/15/15, dan menyimpan file train.csv, val.csv, test.csv.
        """
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Rasio pembagian harus berjumlah 1.0!"

        input_path = Path(input_csv_path)
        if not input_path.exists():
            raise FileNotFoundError(f"File CSV input tidak ditemukan: {input_csv_path}")

        df = pd.read_csv(input_path, encoding="utf-8-sig")
        logger.info(f"Membaca {len(df)} total baris dataset terlabel...")

        # Split 1: Train vs (Val + Test)
        temp_ratio = val_ratio + test_ratio
        train_df, temp_df = train_test_split(
            df,
            test_size=temp_ratio,
            random_state=random_seed,
            shuffle=True,
        )

        # Split 2: Val vs Test
        relative_test_ratio = test_ratio / temp_ratio
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_ratio,
            random_state=random_seed,
            shuffle=True,
        )

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        train_path = out_dir / "train.csv"
        val_path = out_dir / "val.csv"
        test_path = out_dir / "test.csv"

        train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
        val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
        test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

        logger.info("SUKSES: Pembagian Dataset Selesai Tanpa Data Leakage!")
        logger.info(f"- Train Set (70%): {len(train_df)} baris -> {train_path.resolve()}")
        logger.info(f"- Val Set   (15%): {len(val_df)} baris -> {val_path.resolve()}")
        logger.info(f"- Test Set  (15%): {len(test_df)} baris -> {test_path.resolve()}")

        return str(train_path.resolve()), str(val_path.resolve()), str(test_path.resolve())


if __name__ == "__main__":
    print("Dataset Splitter Module Loaded Successfully!")
