"""
Cohen's Kappa Evaluation & Human Audit Sample Generator (CSV & Excel XLSX Support)
Spec Compliance: specs/02_preprocessing_labeling.spec.md
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import cohen_kappa_score

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CohenKappaEvaluator")


class CohenKappaEvaluator:
    """
    Modul Generator Sampel Human Audit (20-30%) dan Pengukur Koefisien Cohen's Kappa.
    Mendukung format CSV dan Excel XLSX.
    """

    ASPECTS = ["infra", "ekonomi", "kualitas", "purnajual"]

    @staticmethod
    def generate_human_audit_sample(
        input_auto_labeled_csv: str,
        sample_ratio: float = 0.20,
        random_seed: int = 42,
        output_base_path: str = "data/interim/human_audit_sample",
    ) -> tuple[str, str]:
        """
        Mengambil sampel acak terstratifikasi sebesar sample_ratio (default 20%)
        dan mengespor file CSV serta Excel (.xlsx) dengan kolom audit manusia kosong.
        """
        input_path = Path(input_auto_labeled_csv)
        if not input_path.exists():
            raise FileNotFoundError(f"File CSV tidak ditemukan: {input_auto_labeled_csv}")

        df = pd.read_csv(input_path, encoding="utf-8-sig")
        logger.info(f"Membaca {len(df)} baris dari {input_path.name}")

        sample_size = int(len(df) * sample_ratio)
        sample_df = df.sample(n=sample_size, random_state=random_seed).copy()

        # Tambahkan kolom verifikasi manusia (human_label_<aspek>)
        for aspect in CohenKappaEvaluator.ASPECTS:
            sample_df[f"human_{aspect}"] = ""  # Kolom kosong untuk diisi 1 auditor manusia

        csv_path = Path(f"{output_base_path}.csv")
        xlsx_path = Path(f"{output_base_path}.xlsx")
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Simpan CSV
        sample_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        
        # Simpan XLSX Excel
        sample_df.to_excel(xlsx_path, index=False)

        logger.info(f"SUKSES: Sampel Audit Manusia ({len(sample_df)} baris / {int(sample_ratio*100)}%) dibuat:")
        logger.info(f"- CSV:   [link]({csv_path.resolve()})")
        logger.info(f"- Excel: [link]({xlsx_path.resolve()})")
        logger.info("Silakan buka file Excel (.xlsx) lalu isi kolom 'human_infra', 'human_ekonomi', 'human_kualitas', 'human_purnajual'.")
        return str(csv_path.resolve()), str(xlsx_path.resolve())

    @staticmethod
    def calculate_kappa_scores(human_audited_file: str) -> dict[str, float]:
        """
        Menghitung koefisien Cohen's Kappa Antara Label Mesin vs Verifikasi Manusia.
        Mendukung file input CSV maupun XLSX.
        """
        input_path = Path(human_audited_file)
        if not input_path.exists():
            raise FileNotFoundError(f"File audit manusia tidak ditemukan: {human_audited_file}")

        if input_path.suffix.lower() == ".xlsx":
            df = pd.read_excel(input_path)
        else:
            df = pd.read_csv(input_path, encoding="utf-8-sig")

        logger.info(f"Mengevaluasi Cohen's Kappa dari {input_path.name} ({len(df)} sampel ter-audit)...")

        kappa_results = {}
        all_kappas = []

        print("\n" + "=" * 60)
        print("COHEN'S KAPPA INTER-ANNOTATOR AGREEMENT EVALUATION")
        print("=" * 60)
        print(f"{'Aspek':<20} | {'Cohen Kappa (k)':<18} | {'Status KPI (>= 0.61)':<18}")
        print("-" * 60)

        for aspect in CohenKappaEvaluator.ASPECTS:
            machine_col = f"{aspect}_sentiment"
            human_col = f"human_{aspect}"

            if machine_col not in df.columns or human_col not in df.columns:
                logger.warning(f"Kolom '{machine_col}' atau '{human_col}' tidak ditemukan.")
                continue

            # Filter data yang sudah diisi oleh manusia (tidak kosong)
            valid_df = df[df[human_col].notnull() & (df[human_col].astype(str).str.strip() != "") & (df[human_col].astype(str).str.lower() != "nan")].copy()

            if len(valid_df) == 0:
                print(f"{aspect:<20} | {'Belum Diisi':<18} | {'-':<18}")
                continue

            y_machine = valid_df[machine_col].fillna("None").astype(str).str.strip()
            y_human = valid_df[human_col].fillna("None").astype(str).str.strip()

            kappa = cohen_kappa_score(y_machine, y_human)
            kappa_results[aspect] = kappa
            all_kappas.append(kappa)

            status = "PASSED [OK]" if kappa >= 0.61 else "NEEDS REFINEMENT"
            print(f"{aspect:<20} | {kappa:<18.4f} | {status:<18}")

        if all_kappas:
            avg_kappa = sum(all_kappas) / len(all_kappas)
            kappa_results["average"] = avg_kappa
            print("-" * 60)
            overall_status = "KPI PRD ACHIEVED (k >= 0.61)" if avg_kappa >= 0.61 else "KPI PRD NOT MET (k < 0.61)"
            print(f"{'RATA-RATA (AVERAGE)':<20} | {avg_kappa:<18.4f} | {overall_status:<18}")
            print("=" * 60 + "\n")

        return kappa_results


if __name__ == "__main__":
    print("Cohen's Kappa Evaluator Module (CSV & XLSX) Loaded Successfully!")
