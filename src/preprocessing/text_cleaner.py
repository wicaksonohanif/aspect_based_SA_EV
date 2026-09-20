"""
Text Cleaning Pipeline Module for Indonesian Social Media Comments
Spec Compliance: specs/02_preprocessing_labeling.spec.md
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TextCleaner")


class TextCleaner:
    """
    Modul Pembersihan & Normalisasi Teks Informal Bahasa Indonesia.
    """

    def __init__(self, slang_dict_path: str = "data/slang_dict.json"):
        """Inisialisasi kamus slang dan pola regex."""
        self.slang_dict = self._load_slang_dict(slang_dict_path)
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.mention_pattern = re.compile(r"@\w+")
        self.repeat_char_pattern = re.compile(r"(.)\1{2,}")
        self.whitespace_pattern = re.compile(r"\s+")

    def _load_slang_dict(self, path_str: str) -> dict[str, str]:
        """Membaca kamus slang dari berkas JSON."""
        path = Path(path_str)
        if not path.exists():
            logger.warning(f"Kamus slang '{path_str}' tidak ditemukan! Pembersihan slang dilewati.")
            return {}

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def clean_text(self, text: str) -> str:
        """
        Pembersihan teks bertahap:
        1. Lowercasing
        2. Hapus URL
        3. Hapus Mention
        4. Reduksi huruf berulang >2
        5. Normalisasi Slang / Kata Gaul
        6. Pembersihan Whitespace
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # 1. Lowercase
        text = text.lower()

        # 2. Remove URL
        text = self.url_pattern.sub(" ", text)

        # 3. Remove Mention
        text = self.mention_pattern.sub(" ", text)

        # 4. Reduksi karakter berulang (>2 kali) contoh: "baguuuus" -> "baguus"
        text = self.repeat_char_pattern.sub(r"\1\1", text)

        # 5. Normalisasi Slang / Kata Gaul
        words = text.split()
        cleaned_words = [self.slang_dict.get(word, word) for word in words]
        text = " ".join(cleaned_words)

        # 6. Pembersihan Whitespace
        text = self.whitespace_pattern.sub(" ", text).strip()

        return text

    def process_csv(self, input_csv_path: str, output_csv_path: str | None = None) -> str:
        """
        Membaca data mentah dari CSV input, membersihkan teks, dan menyimpan ke data/interim/.
        """
        input_path = Path(input_csv_path)
        if not input_path.exists():
            raise FileNotFoundError(f"File CSV input tidak ditemukan: {input_csv_path}")

        logger.info(f"Membaca data mentah dari: {input_path.name}")
        df = pd.read_csv(input_path, encoding="utf-8-sig")

        if "text_original" not in df.columns:
            raise KeyError("Kolom 'text_original' tidak ditemukan dalam file CSV!")

        logger.info(f"Memproses pembersihan pada {len(df)} baris komentar...")
        df["text_cleaned"] = df["text_original"].apply(self.clean_text)

        # Tentukan path output di data/interim/
        if not output_csv_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv_path = f"data/interim/cleaned_comments_{timestamp}.csv"

        out_path = Path(output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"SUKSES: Teks terbersihkan disimpan di: [link]({out_path.resolve()})")
        return str(out_path.resolve())


if __name__ == "__main__":
    # Test sederhana
    cleaner = TextCleaner()
    sample = "Gak pake SPKLU mobkas ini baguuusss bgt! Cek https://ev.id @user"
    cleaned = cleaner.clean_text(sample)
    print("Sebelum:", sample)
    print("Sesudah:", cleaned)
