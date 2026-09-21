"""
Google Gemini API Labeler for Aspect-Based Sentiment Analysis
Spec Compliance: specs/02_preprocessing_labeling.spec.md
"""

import json
import logging
import os
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("GeminiLabeler")


class GeminiAspectLabeler:
    """
    Interface Google Gemini API Pro/Flash untuk Pelabelan Aspect-Based Sentiment Analysis
    menggunakan Structured JSON Response Mode.
    """

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-flash-latest"):
        """Inisialisasi Gemini API Client (default model: gemini-flash-latest)."""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key or self.api_key == "YOUR_GEMINI_API_KEY_HERE":
            raise ValueError(
                "GEMINI_API_KEY tidak ditemukan! "
                "Harap isi API Key Anda di berkas '.env' atau lewatkan ke parameter 'api_key'."
            )

        self.model_name = model_name
        self.model = self._build_model()

    def _build_model(self) -> Any:
        """Inisialisasi GenerativeModel dengan response_mime_type application/json."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            return genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={"response_mime_type": "application/json"},
            )
        except ImportError:
            raise ImportError(
                "Pustaka 'google-generativeai' belum terinstal. "
                "Silakan jalankan: pip install -r requirements.txt"
            )

    def label_comment(self, text: str, max_retries: int = 3) -> dict[str, str | None]:
        """
        Mengirimkan komentar ke Gemini API dengan retry backoff jika terkena rate limit 429.
        """
        prompt = f"""
        Kamu adalah pakar Aspect-Based Sentiment Analysis (ABSA) otomotif Indonesia.
        Analisis komentar berikut dan berikan nilai polaritas sentimen ('Positif', 'Netral', 'Negatif', atau null jika tidak dibahas) untuk 4 aspek kendaraan listrik (EV) China:
        1. infra (Infrastruktur & Jangkauan: SPKLU, charging station, baterai habis, durasi cas, PLN)
        2. ekonomi (Ekonomi & Finansial: Harga baru/bekas, depresiasi resale value, pajak, efisiensi biaya)
        3. kualitas (Kualitas & Durabilitas: Durabilitas baterai Blade/LFP, suspensi, interior, build quality, software ADAS)
        4. purnajual (Purna Jual & Ekosistem: Dealer resmi, sparepart, suku cadang, bengkel, respon sales)

        Teks Komentar: "{text}"

        Output WAJIB berupa format JSON persis seperti berikut tanpa tambahan teks lain:
        {{
          "infra_sentiment": "Positif" | "Netral" | "Negatif" | null,
          "ekonomi_sentiment": "Positif" | "Netral" | "Negatif" | null,
          "kualitas_sentiment": "Positif" | "Netral" | "Negatif" | null,
          "purnajual_sentiment": "Positif" | "Netral" | "Negatif" | null
        }}
        """

        for attempt in range(1, max_retries + 1):
            try:
                response = self.model.generate_content(prompt)
                result = json.loads(response.text)

                return {
                    "infra_sentiment": result.get("infra_sentiment"),
                    "ekonomi_sentiment": result.get("ekonomi_sentiment"),
                    "kualitas_sentiment": result.get("kualitas_sentiment"),
                    "purnajual_sentiment": result.get("purnajual_sentiment"),
                }
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower():
                    sleep_time = 4 * attempt
                    logger.warning(f"Rate Limit 429 pada percobann {attempt}/{max_retries}. Menunggu {sleep_time} detik...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Error Gemini API pada percobaan {attempt}/{max_retries}: {e}")
                    time.sleep(1)

        return {
            "infra_sentiment": None,
            "ekonomi_sentiment": None,
            "kualitas_sentiment": None,
            "purnajual_sentiment": None,
        }
