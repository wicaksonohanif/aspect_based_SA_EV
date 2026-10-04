import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, login

sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()
token = os.getenv("HUGGINGFACE_API_KEY")

if not token:
    print("❌ Error: HUGGINGFACE_API_KEY tidak ditemukan di file .env")
    sys.exit(1)

print("🔑 Authenticating with Hugging Face...")
login(token=token)

HF_USERNAME = "wicaksonohanif"
REPO_NAME = "xlm-roberta-ev-absa"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"
MODEL_DIR = Path("models/xlmroberta_local_absa")

if not MODEL_DIR.exists():
    print(f"❌ Folder model '{MODEL_DIR}' tidak ditemukan.")
    sys.exit(1)

print(f"🚀 Memastikan repositori '{REPO_ID}' tersedia di Hugging Face Hub...")
create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

api = HfApi()

print(f"📦 Mengunggah berkas model dari '{MODEL_DIR}' ke '{REPO_ID}'...")
print("  File yang diunggah:")
for f in MODEL_DIR.iterdir():
    if f.is_file():
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.2f} MB)")

api.upload_folder(
    folder_path=str(MODEL_DIR),
    repo_id=REPO_ID,
    repo_type="model"
)

print(f"\n🎉 SUKSES: Model XLM-RoBERTa berhasil diunggah ke Hugging Face Hub!")
print(f"🔗 URL Model: https://huggingface.co/{REPO_ID}")
