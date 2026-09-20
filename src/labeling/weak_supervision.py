"""
Hybrid Weak Supervision Engine (Rule-Based + Google Gemini API)
Spec Compliance: specs/02_preprocessing_labeling.spec.md
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .gemini_labeler import GeminiAspectLabeler

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("WeakSupervisionEngine")


# Kamus Aturan Kata Kunci Lokal (Rule-Based Stage)
ASPECT_KEYWORDS = {
    "infra": ["spklu", "charging", "cas", "ngecas", "isi daya", "pln", "jarak tempuh", "baterai habis", "stasiun pengisian"],
    "ekonomi": ["harga", "murah", "mahal", "bekas", "mobkas", "depresiasi", "resale", "pajak", "irit", "boncos", "garansi"],
    "kualitas": ["baterai", "blade", "lfp", "suspensi", "build quality", "interior", "fitur", "adas", "rem", "rusak", "awet", "nyaman"],
    "purnajual": ["dealer", "bengkel", "sparepart", "suku cadang", "indent", "sales", "layanan", "servis", "service"],
}

POSITIVE_WORDS = ["bagus", "keren", "mantap", "awet", "irit", "murah", "layak", "worth", "suka", "puas", "nyaman", "rekomended", "gacor"]
NEGATIVE_WORDS = ["rusak", "jelek", "mahal", "boncos", "rugi", "kecewa", "lambat", "susah", "parah", "riwet", "kacau", "batal"]


class WeakSupervisionEngine:
    """
    Mesin Pelabelan Otomatis Hybrid (Rule-Based Filter -> Google Gemini API Fallback)
    """

    def __init__(self, use_gemini: bool = True, gemini_api_key: str | None = None):
        """Inisialisasi engine pelabelan."""
        self.use_gemini = use_gemini
        self.gemini_labeler = None
        if self.use_gemini:
            try:
                self.gemini_labeler = GeminiAspectLabeler(api_key=gemini_api_key)
                logger.info("Gemini API Labeler berhasil diaktifkan.")
            except Exception as e:
                logger.warning(f"Gagal mengaktifkan Gemini API Labeler ({e}). Sistem akan menggunakan Rule-Based murni.")
                self.use_gemini = False

    def _rule_based_check(self, text: str) -> tuple[dict[str, str | None], bool]:
        """
        Pemeriksaan aturan lokal berbasis kata kunci.
        Returns: (labels_dict, is_ambiguous)
        """
        text_lower = text.lower()
        labels = {
            "infra_sentiment": None,
            "ekonomi_sentiment": None,
            "kualitas_sentiment": None,
            "purnajual_sentiment": None,
        }

        matched_aspects = 0
        has_clear_polarity = False

        for aspect, keywords in ASPECT_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                matched_aspects += 1
                pos_match = any(pw in text_lower for pw in POSITIVE_WORDS)
                neg_match = any(nw in text_lower for nw in NEGATIVE_WORDS)

                if pos_match and not neg_match:
                    labels[f"{aspect}_sentiment"] = "Positif"
                    has_clear_polarity = True
                elif neg_match and not pos_match:
                    labels[f"{aspect}_sentiment"] = "Negatif"
                    has_clear_polarity = True
                elif not pos_match and not neg_match:
                    labels[f"{aspect}_sentiment"] = "Netral"
                    has_clear_polarity = True

        # Komentar dianggap ambigu jika tidak ada kata kunci aspek yang cocok, atau polaritas bertolak belakang
        is_ambiguous = (matched_aspects == 0) or (not has_clear_polarity)
        return labels, is_ambiguous

    def label_single_text(self, text: str) -> dict[str, Any]:
        """
        Proses pelabelan hybrid:
        1. Coba Rule-Based lokal.
        2. Jika ambigu & Gemini aktif -> panggil Gemini API.
        """
        labels, is_ambiguous = self._rule_based_check(text)

        if not is_ambiguous:
            labels["label_source"] = "Rule-Based"
            return labels

        # Fallback ke Gemini API jika ambigu
        if self.use_gemini and self.gemini_labeler:
            gemini_labels = self.gemini_labeler.label_comment(text)
            gemini_labels["label_source"] = "Gemini-API"
            return gemini_labels

        labels["label_source"] = "Rule-Based (Ambiguous)"
        return labels

    def process_cleaned_csv(
        self,
        input_cleaned_csv: str,
        output_csv_path: str | None = None,
        max_rows: int | None = None,
    ) -> str:
        """
        Membaca CSV terbersihkan dari data/interim/, menjalankan pelabelan hybrid,
        dan menyimpan data terlabel ke data/interim/auto_labeled_comments_<timestamp>.csv.
        """
        input_path = Path(input_cleaned_csv)
        if not input_path.exists():
            raise FileNotFoundError(f"File CSV input tidak ditemukan: {input_cleaned_csv}")

        logger.info(f"Membaca data terbersihkan dari: {input_path.name}")
        df = pd.read_csv(input_path, encoding="utf-8-sig")

        if "text_cleaned" not in df.columns:
            raise KeyError("Kolom 'text_cleaned' tidak ditemukan dalam file CSV!")

        if max_rows:
            df = df.head(max_rows)

        logger.info(f"Memulai pelabelan Hybrid pada {len(df)} komentar...")

        results = []
        rule_count = 0
        gemini_count = 0

        for idx, row in df.iterrows():
            text = str(row["text_cleaned"])
            labeled_data = self.label_single_text(text)

            if labeled_data.get("label_source") == "Rule-Based":
                rule_count += 1
            else:
                gemini_count += 1

            record = row.to_dict()
            record.update(labeled_data)
            results.append(record)

            if (idx + 1) % 50 == 0 or (idx + 1) == len(df):
                logger.info(f"Progress Pelabelan: [{idx + 1}/{len(df)}] (Rule-Based: {rule_count}, Gemini API: {gemini_count})")
            
            # Delay singkat jika memanggil Gemini API
            if labeled_data.get("label_source") == "Gemini-API":
                time.sleep(0.2)

        out_df = pd.DataFrame(results)

        if not output_csv_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv_path = f"data/interim/auto_labeled_comments_{timestamp}.csv"

        out_path = Path(output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"SUKSES: Data terlabel otomatis disimpan di: [link]({out_path.resolve()})")
        logger.info(f"Statistik Pelabelan: Total = {len(out_df)} | Rule-Based = {rule_count} | Gemini API = {gemini_count}")
        return str(out_path.resolve())


if __name__ == "__main__":
    engine = WeakSupervisionEngine(use_gemini=False)
    sample_text = "spklu di tol sedikit banget, tapi batrenya awet"
    print("Test Sample:", engine.label_single_text(sample_text))
