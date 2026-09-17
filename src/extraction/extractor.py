"""
YouTube Comment Data Extractor
Spec Compliance: specs/01_data_extraction.spec.md
"""

import csv
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("YouTubeExtractor")

# Load environment variables from .env file if available
load_dotenv()


def parse_youtube_video_id(url_or_id: str) -> str:
    """
    Mengekstrak 11-karakter video_id dari berbagai format URL YouTube atau ID mentah.

    Format URL yang didukung:
    - Standard: https://www.youtube.com/watch?v=dQw4w9WgXcQ
    - Shortened: https://youtu.be/dQw4w9WgXcQ
    - Shorts: https://www.youtube.com/shorts/dQw4w9WgXcQ
    - Embed: https://www.youtube.com/embed/dQw4w9WgXcQ
    - Raw ID: dQw4w9WgXcQ
    """
    url_or_id = url_or_id.strip()
    if not url_or_id or url_or_id.startswith("#"):
        return ""

    # Check if raw 11-char ID
    if re.match(r"^[0-9A-Za-z_-]{11}$", url_or_id):
        return url_or_id

    # Regex for Youtube URLs
    patterns = [
        r"(?:v=|\/([0-9A-Za-z_-]{11}).*|youtu\.be\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})",
        r"v=([0-9A-Za-z_-]{11})",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"shorts\/([0-9A-Za-z_-]{11})",
        r"embed\/([0-9A-Za-z_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            # Pick the group that captured the 11-char ID
            for group in match.groups():
                if group and len(group) == 11 and re.match(r"^[0-9A-Za-z_-]{11}$", group):
                    return group

    logger.warning(f"Gagal mengurai video_id dari URL/string: '{url_or_id}'")
    return ""


class YouTubeCommentExtractor:
    """
    Modul Ekstraksi Komentar YouTube Resmi (Top-Level Comments Only)
    Mematuhi Spesifikasi specs/01_data_extraction.spec.md
    """

    def __init__(self, api_key: str | None = None, salt: str | None = None):
        """
        Inisialisasi YouTube Data API v3 client & konfigurasi PII masking.
        """
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY")
        if not self.api_key or self.api_key == "YOUR_YOUTUBE_API_KEY_HERE":
            raise ValueError(
                "YOUTUBE_API_KEY tidak ditemukan! "
                "Harap isi API Key Anda di berkas '.env' atau lewatkan ke parameter 'api_key'."
            )

        self.salt = salt or os.getenv("PII_SALT", "usb2026_salt_secret")
        self.youtube_client = self._build_youtube_client()

    def _build_youtube_client(self) -> Any:
        """Inisialisasi client Google API."""
        try:
            from googleapiclient.discovery import build
            return build("youtube", "v3", developerKey=self.api_key)
        except ImportError:
            raise ImportError(
                "Pustaka 'google-api-python-client' belum terinstal. "
                "Silakan jalankan: pip install -r requirements.txt"
            )

    def _hash_user_id(self, channel_id_or_name: str) -> str:
        """Generasi hash SHA-256 anonim untuk PII Masking."""
        raw_string = f"{channel_id_or_name}_{self.salt}"
        return hashlib.sha256(raw_string.encode("utf-8")).hexdigest()

    def extract_comments_from_video(
        self,
        video_url_or_id: str,
        max_comments: int = 1000,
        order: str = "relevance",
    ) -> list[dict[str, Any]]:
        """
        Mengekstrak top-level komentar dari satu video YouTube menggunakan pagination.

        Args:
            video_url_or_id: URL YouTube atau ID Video 11 karakter.
            max_comments: Batas maksimum komentar yang diambil untuk video ini.
            order: 'relevance' atau 'time'.

        Returns:
            Daftar dict komentar dengan PII ter-masking.
        """
        video_id = parse_youtube_video_id(video_url_or_id)
        if not video_id:
            logger.error(f"URL/ID Video tidak valid: '{video_url_or_id}'")
            return []

        logger.info(f"Memulai ekstraksi komentar untuk video_id: {video_id} (max_comments: {max_comments})")

        comments: list[dict[str, Any]] = []
        page_token: str | None = None
        extracted_count = 0

        while extracted_count < max_comments:
            # Hitung jumlah item yang perlu diminta (maxResults max 100)
            batch_size = min(100, max_comments - extracted_count)

            try:
                request = self.youtube_client.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=batch_size,
                    pageToken=page_token,
                    textFormat="plainText",
                    order=order,
                )
                response = request.execute()
            except Exception as e:
                error_str = str(e)
                if "commentsDisabled" in error_str:
                    logger.warning(f"Komentar dimatikan pada video {video_id}. Melewati video ini.")
                elif "quotaExceeded" in error_str:
                    logger.error("KUOTA YOUTUBE API HABIS (quotaExceeded)! Menghentikan ekstraksi aman.")
                else:
                    logger.error(f"Error HTTP/API pada video {video_id}: {e}")
                break

            items = response.get("items", [])
            if not items:
                logger.info(f"Tidak ada komentar lebih lanjut untuk video {video_id}.")
                break

            for item in items:
                try:
                    snippet = item["snippet"]["topLevelComment"]["snippet"]
                    comment_id = item.get("id", "")

                    # PII Masking: Gunakan channelId pengarang jika ada, atau name
                    author_channel = snippet.get("authorChannelId", {}).get("value", "")
                    author_name = snippet.get("authorDisplayName", "anonymous")
                    user_identifier = author_channel if author_channel else author_name
                    user_id_hash = self._hash_user_id(user_identifier)

                    text_original = snippet.get("textOriginal", "")
                    like_count = snippet.get("likeCount", 0)
                    published_at = snippet.get("publishedAt", "")
                    updated_at = snippet.get("updatedAt", "")
                    now_utc = datetime.now(timezone.utc).isoformat()

                    record = {
                        "comment_id": comment_id,
                        "video_id": video_id,
                        "user_id_hash": user_id_hash,
                        "text_original": text_original,
                        "like_count": like_count,
                        "published_at": published_at,
                        "updated_at": updated_at,
                        "extracted_at": now_utc,
                    }
                    comments.append(record)
                    extracted_count += 1

                    if extracted_count >= max_comments:
                        break
                except KeyError as ke:
                    logger.warning(f"Struktur item komentar tidak lengkap: {ke}")
                    continue

            logger.info(f"Video {video_id}: Total komentar terekstrak sejauh ini = {extracted_count}")

            # Cek token halaman berikutnya
            page_token = response.get("nextPageToken")
            if not page_token:
                logger.info(f"Pagination selesai (halaman terakhir) untuk video {video_id}.")
                break

            # Sleep 0.5 detik untuk menjaga rate limit
            time.sleep(0.5)

        return comments

    def extract_from_video_list(
        self,
        urls_or_ids: list[str],
        max_comments_per_video: int = 1000,
    ) -> list[dict[str, Any]]:
        """
        Iterasi ekstraksi dari daftar URL/ID video.
        """
        all_comments: list[dict[str, Any]] = []

        for idx, url_or_id in enumerate(urls_or_ids, start=1):
            url_or_id = url_or_id.strip()
            if not url_or_id or url_or_id.startswith("#"):
                continue

            logger.info(f"Processing Video [{idx}/{len(urls_or_ids)}]: {url_or_id}")
            comments = self.extract_comments_from_video(
                video_url_or_id=url_or_id,
                max_comments=max_comments_per_video,
            )
            all_comments.extend(comments)

        return all_comments

    @staticmethod
    def load_urls_from_file(file_path: str) -> list[str]:
        """Membaca daftar URL/ID dari file teks."""
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File tidak ditemukan: {file_path}")
            return []

        urls = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(line)
        return urls

    def save_to_csv(
        self,
        comments: list[dict[str, Any]],
        output_filepath: str | None = None,
    ) -> str:
        """
        Menyimpan hasil ekstraksi komentar ke berkas CSV (utf-8-sig & QUOTE_ALL).
        """
        if not comments:
            logger.warning("Daftar komentar kosong. Tidak ada file CSV yang ditulis.")
            return ""

        if not output_filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filepath = f"data/raw/yt_comments_{timestamp}.csv"

        output_path = Path(output_filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(comments)
        df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
            quoting=csv.QUOTE_ALL,
        )
        logger.info(f"SUKSES: {len(comments)} komentar berhasil disimpan ke CSV: [link]({output_path.resolve()})")
        return str(output_path.resolve())

    def save_to_jsonl(
        self,
        comments: list[dict[str, Any]],
        output_filepath: str | None = None,
    ) -> str:
        """
        Menyimpan hasil ekstraksi komentar ke berkas JSONL.
        """
        if not comments:
            logger.warning("Daftar komentar kosong. Tidak ada file JSONL yang ditulis.")
            return ""

        if not output_filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filepath = f"data/raw/yt_comments_{timestamp}.jsonl"

        output_path = Path(output_filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for comment in comments:
                f.write(json.dumps(comment, ensure_ascii=False) + "\n")

        logger.info(f"SUKSES: {len(comments)} komentar berhasil disimpan ke JSONL: [link]({output_path.resolve()})")
        return str(output_path.resolve())


if __name__ == "__main__":
    print("=" * 60)
    print("YouTube Comment Extractor - Aspect-Based SA EV Project 2026")
    print("=" * 60)

    # 1. Cek file video_list.txt
    video_list_file = "data/video_list.txt"
    urls = YouTubeCommentExtractor.load_urls_from_file(video_list_file)

    if not urls:
        print(f"\n[INFO] File '{video_list_file}' belum berisi tautan video YouTube.")
        print("Silakan tambahkan tautan video YouTube ke dalam 'data/video_list.txt'.")
        print("Atau jalankan skrip ini secara terprogram dengan mengimpor YouTubeCommentExtractor.")
    else:
        print(f"\nDitemukan {len(urls)} tautan video di '{video_list_file}'.")
        try:
            extractor = YouTubeCommentExtractor()
            comments = extractor.extract_from_video_list(urls, max_comments_per_video=500)

            if comments:
                csv_path = extractor.save_to_csv(comments)
                jsonl_path = extractor.save_to_jsonl(comments)
                print(f"\n[BERHASIL] Ekstraksi selesai! Total {len(comments)} komentar disimpan.")
                print(f"- CSV: {csv_path}")
                print(f"- JSONL: {jsonl_path}")
            else:
                print("\n[INFO] Tidak ada komentar yang berhasil diekstraksi.")
        except ValueError as ve:
            print(f"\n[PERHATIAN] {ve}")
