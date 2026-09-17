# Technical Specification: YouTube Comment Data Extraction

- **Spec ID:** SPEC-01
- **Feature Title:** Ekstraksi Komentar YouTube Resmi (Top-Level Comments Only, CSV/JSONL Export, Auto URL Parser)
- **Target Component:** `src/extraction/`
- **Related PRD Section:** FR-1 (Data Extraction Module)
- **Status:** Approved / In Implementation

---

## 1. Overview & Objectives

Dokumen spesifikasi ini mengatur modul ekstraksi data komentar YouTube menggunakan **YouTube Data API v3** resmi melalui pustaka `google-api-python-client`. Modul ini bertujuan mengumpulkan korpus opini publik warganet Indonesia mengenai mobil listrik (EV) pabrikan China (misal: Wuling, BYD, Chery, Seres, Neta) secara inklusif, etis, dan scalable.

### Key Mandates & Scope Boundaries:
1. **Top-Level Comments Only:** Hanya mengambil komentar tingkat teratas (`topLevelComment`). Balasan komentar (*replies*) **diabaikan** secara eksplisit untuk efisiensi kuota dan fokus pada opini utama.
2. **Flexible Input & Auto URL Parser:** Pengguna dapat memasukkan tautan YouTube lengkap (`https://www.youtube.com/watch?v=...`, `https://youtu.be/...`, `youtube.com/shorts/...`) maupun ID video mentah melalui berkas `data/video_list.txt` atau variabel script.
3. **Dual Output Support (CSV & JSONL):** Hasil ekstraksi disimpan secara default ke berkas **CSV** (`utf-8-sig`, `QUOTE_ALL`) agar mudah dibuka di Microsoft Excel, Google Sheets, dan Pandas, serta opsional ke **JSONL**.
4. **Credential Management:** Mendukung pembacaan `YOUTUBE_API_KEY` dari berkas `.env`, variabel lingkungan OS, argumen CLI, atau variabel skrip.
5. **Strict PII Masking:** Data identitas pribadi (*Personally Identifiable Information* / PII) seperti nama akun dan ID kanal YouTube dihapus/disamarkan secara seketika (*real-time masking*) menggunakan hashing SHA-256 sebelum disimpan ke disk.
6. **Quota Guardrail:** Penggunaan kuota harian dijaga agar tidak melebihi 10.000 unit/hari (setiap panggilan `commentThreads.list` mengonsumsi 1 unit kuota).

---

## 2. YouTube URL Parser & Credential Specifications

### 2.1 YouTube URL & Video ID Parsing
Modul harus mendukung ekstraksi `video_id` 11-karakter dari berbagai format URL berikut menggunakan ekspresi reguler (Regex):
- Standard URL: `https://www.youtube.com/watch?v=dQw4w9WgXcQ` $\rightarrow$ `dQw4w9WgXcQ`
- Shortened URL: `https://youtu.be/dQw4w9WgXcQ` $\rightarrow$ `dQw4w9WgXcQ`
- Shorts URL: `https://www.youtube.com/shorts/dQw4w9WgXcQ` $\rightarrow$ `dQw4w9WgXcQ`
- Embed URL: `https://www.youtube.com/embed/dQw4w9WgXcQ` $\rightarrow$ `dQw4w9WgXcQ`
- Raw Video ID: `dQw4w9WgXcQ` $\rightarrow$ `dQw4w9WgXcQ`

**Regex Pattern Target:**
`r"(?:v=|\/([0-9A-Za-z_-]{11}).*|youtu\.be\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})"`

### 2.2 Input Source Tautan Video
Modul `src/extraction/extractor.py` membaca daftar tautan video dari:
1. **File Input:** Berkas teks `data/video_list.txt` (satu tautan per baris, baris dengan awalan `#` diabaikan sebagai komentar).
2. **List Python / CLI Arg:** Parameter `video_urls` pada fungsi/class `YouTubeCommentExtractor`.

### 2.3 Management Kredensial API Key
Order prioritas pembacaan API Key:
1. Parameter langsung pada inisialisasi class/fungsi (`api_key="..."`).
2. Argumen CLI (`--api-key="..."`).
3. Variabel lingkungan dari berkas `.env` (`YOUTUBE_API_KEY="..."`).
4. Variabel lingkungan OS (`os.getenv("YOUTUBE_API_KEY")`).

---

## 3. API Contract & Parameters

### 3.1 YouTube Data API v3 Endpoint
- **Service:** `youtube.commentThreads()`
- **Method:** `list()`
- **Quota Cost:** 1 unit per API call.

### 3.2 Parameter Konfigurasi Request
| Parameter | Tipe | Nilai / Format | Deskripsi |
|---|---|---|---|
| `part` | `str` | `"snippet"` | Meminta metadata komentar (tidak menyertakan `"replies"`). |
| `videoId` | `str` | `[ALPHANUMERIC_ID]` | ID unik video YouTube target (contoh: `dQw4w9WgXcQ`). |
| `maxResults` | `int` | `100` | Jumlah maksimum thread komentar per panggilan API (maksimum API = 100). |
| `textFormat` | `str` | `"plainText"` | Mengambil teks komentar mentah tanpa tag HTML. |
| `order` | `str` | `"relevance"` atau `"time"` | Urutan pengambilan komentar (default: `relevance`). |
| `pageToken` | `str` | `[NEXT_PAGE_TOKEN]` | Token pagination (kosong/`None` pada panggilan pertama). |

---

## 4. Functional Requirements (FR)

### FR-1.1: Autentikasi API Key
- API Key diambil dari variabel lingkungan `YOUTUBE_API_KEY` via `python-dotenv` atau parameter eksplisit.
- Jika `YOUTUBE_API_KEY` tidak ditemukan, modul harus melemparkan *exception* `ValueError` yang informatif dengan petunjuk membuat berkas `.env`.

### FR-1.2: Pagination Loop & Top-Level Extraction
- Ekstraksi dijalankan secara berulang dalam *loop* pagination:
  1. Inisialisasi request pertama dengan `pageToken = None`.
  2. Ekstrak objek `items` dari respon API.
  3. Iterasi setiap objek `item` dalam `items`, ambil bagian `item["snippet"]["topLevelComment"]`.
  4. Ambil nilai `nextPageToken` dari respon API.
  5. Jika `nextPageToken` ada dan belum mencapai batas maksimum komentar/kuota, set `pageToken = nextPageToken` dan ulang request.
  6. Jika `nextPageToken` tidak ada (`None`), ekstraksi untuk video tersebut selesai.

### FR-1.3: Eksklusi Balasan Komentar (No Replies)
- Modul **dilarang** menyertakan parameter `part="replies"` atau memanggil endpoint `comments().list()`.
- Properti `replies` dalam respon API (jika ada) diabaikan sepenuhnya dan tidak dimasukkan ke dalam skema data mentah.

### FR-1.4: Strict PII Masking & Hashing
Sebelum data dimasukkan ke dalam dataframe/CSV untuk disimpan:
- Hapus field asli `authorDisplayName`, `authorChannelId`, dan `authorProfileImageUrl`.
- Generate `user_id_hash` menggunakan **SHA-256**:
  $$\text{user\_id\_hash} = \text{SHA256}(\text{authorChannelId} + \text{salt})$$
  *(Salt dapat diatur di .env atau default di script).*

### FR-1.5: Format & Immutability Penyimpanan Data CSV & JSONL
- Data hasil ekstraksi disimpan ke **`data/raw/`**.
- Penamaan file mengikuti konvensi:
  - CSV: `data/raw/yt_comments_<timestamp>.csv` atau `data/raw/yt_comments_<video_id>_<timestamp>.csv`
  - JSONL (opsional): `data/raw/yt_comments_<timestamp>.jsonl`
- Standar Penulisan CSV:
  - Encoding: `utf-8-sig` (Excel BOM compatible).
  - Quoting: `csv.QUOTE_ALL` (mencegah korupsi baris akibat *newline* atau koma pada komentar).
- File di `data/raw/` bersifat *read-only / immutable* setelah ditulis. Modul tidak boleh menimpa file yang sudah ada tanpa izin eksplisit (*append-only* atau buat file baru).

---

## 5. Raw Data Schema (CSV & JSONL Output)

Setiap baris data komentar pada CSV atau JSONL harus memiliki atribut berikut:

| Nama Kolom | Tipe Data | Contoh Nilai | Deskripsi |
|---|---|---|---|
| `comment_id` | `str` | `"UgwX1Y2Z3A4B5C6D7E8"` | ID unik komentar dari YouTube. |
| `video_id` | `str` | `"dQw4w9WgXcQ"` | ID video YouTube asal. |
| `user_id_hash` | `str` | `"a8f5f167f44f4964e6c998dee827110c"` | Hash SHA-256 identitas pengguna (PII Masked). |
| `text_original` | `str` | `"Baterai BYD Blade ini tahan lama!"` | Teks asli komentar (plainText). |
| `like_count` | `int` | `42` | Jumlah suka pada komentar. |
| `published_at` | `str` | `"2026-08-15T10:30:00Z"` | Tanggal publikasi (ISO 8601). |
| `updated_at` | `str` | `"2026-08-15T10:30:00Z"` | Tanggal suntingan (ISO 8601). |
| `extracted_at` | `str` | `"2026-09-17T11:20:00Z"` | Timestamp ekstraksi data. |

---

## 6. Non-Functional & Guardrail Requirements

### 6.1 Quota & Rate Limit Management
- Modul wajib mencatat (*log*) perkiraan jumlah panggilan API yang dilakukan.
- Setiap *batch request* diberikan jeda (*sleep*) singkat (`0.5` detik) untuk mencegah penolakan rate limit.
- Batas komentar per video dapat dikonfigurasi via parameter `max_comments_per_video` (default: `1000` komentar per video).

### 6.2 Error Handling & Resilience
- **`commentsDisabled` (HttpError 403 / 404):** Jika komentar dimatikan, catat *warning log* dan lewati (*skip*) video tersebut tanpa menghentikan program.
- **`quotaExceeded` (HttpError 403):** Simpan data yang sudah berhasil diekstraksi ke disk, tampilkan pesan error yang jelas, dan hentikan eksekusi dengan aman (*graceful shutdown*).
- **Network / Transient Errors:** Terapkan *exponential backoff retry* (maksimal 3 kali percobaan) untuk error koneksi sementara.

---

## 7. Proposed Implementation Architecture (`src/extraction/extractor.py`)

### 7.1 Key Functions & Class Structure

```python
def parse_youtube_video_id(url_or_id: str) -> str:
    """Mengekstrak 11-karakter video_id dari URL YouTube berbagai format."""
    ...

class YouTubeCommentExtractor:
    def __init__(self, api_key: str | None = None, salt: str = "usb2026_salt"):
        """Inisialisasi YouTube Data API v3 client & konfigurasi PII masking."""
        ...

    def extract_comments_from_video(
        self, 
        video_url_or_id: str, 
        max_comments: int = 1000
    ) -> list[dict]:
        """Mengekstrak top-level comments dari single video via pagination loop."""
        ...

    def extract_from_video_list(
        self, 
        urls_or_ids: list[str], 
        max_comments_per_video: int = 1000
    ) -> list[dict]:
        """Iterasi ekstraksi dari daftar video."""
        ...

    def save_to_csv(self, comments: list[dict], output_filepath: str) -> str:
        """Menyimpan hasil ekstraksi ke CSV dengan utf-8-sig & QUOTE_ALL."""
        ...

    def save_to_jsonl(self, comments: list[dict], output_filepath: str) -> str:
        """Menyimpan hasil ekstraksi ke format JSONL."""
        ...
```

---

## 8. Acceptance Criteria & Verification Plan

| No | Kriteria Pengujian | Cara Verifikasi | Expectation / Target |
|---|---|---|---|
| **AC-1** | Ekstraksi CSV Berhasil & Rapi | Buka file CSV hasil di Excel/Pandas | Karakter/emoji terbaca, tidak ada baris corrupt akibat koma/newline. |
| **AC-2** | URL Parser Fleksibel | Pass berbagai format URL YouTube | `video_id` 11 karakter terekstrak dengan benar dari semua format URL. |
| **AC-3** | Tanpa Balasan Komentar (No Replies) | Audit data output | Hanya `topLevelComment` yang diambil, tidak ada balasan thread. |
| **AC-4** | Strict PII Masking | Audit atribut CSV | Nama/ID akun asli tidak ada, `user_id_hash` terisi SHA-256. |
| **AC-5** | Kredensial Terbaca dari `.env` | Run tanpa argumen API key | Modul otomatis membaca `YOUTUBE_API_KEY` dari file `.env`. |
