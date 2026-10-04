import os
import sys

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath("."))

LABEL2ID = {'None': 0, 'none': 0, 'positif': 1, 'netral': 2, 'negatif': 3}
ID2LABEL = {0: 'None', 1: 'positif', 2: 'netral', 3: 'negatif'}
ASPECTS = ['infra', 'ekonomi', 'kualitas', 'purnajual']

ASPECT_THRESHOLDS = {
    "infra": 0.40,
    "ekonomi": 0.50,
    "kualitas": 0.50,
    "purnajual": 0.35,
}

# Try importing torch & transformers
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Try importing weak supervision fallback
try:
    from src.labeling.weak_supervision import WeakSupervisionEngine
    HAS_WEAK_SUPERVISION = True
except ImportError:
    HAS_WEAK_SUPERVISION = False


if HAS_TORCH:
    class XLMRoBERTaMultiHeadClassifier(nn.Module):
        def __init__(self, model_name="xlm-roberta-large", num_classes=4, dropout_prob=0.1, aspects=ASPECTS):
            super().__init__()
            self.aspects = aspects
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden_size = self.encoder.config.hidden_size

            self.heads = nn.ModuleDict({
                aspect: nn.Sequential(
                    nn.Linear(hidden_size * 2, 512),
                    nn.LayerNorm(512),
                    nn.GELU(),
                    nn.Dropout(dropout_prob),
                    nn.Linear(512, num_classes)
                ) for aspect in aspects
            })

        def forward(self, input_ids, attention_mask):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state

            cls_token = last_hidden[:, 0, :]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            pooled_features = torch.cat([cls_token, mean_pooled], dim=-1)

            logits = {}
            for aspect in self.aspects:
                logits[aspect] = self.heads[aspect](pooled_features)

            return logits


    def predict_with_threshold(logits, none_threshold=ASPECT_THRESHOLDS, aspect=None):
        thresh = none_threshold.get(aspect, 0.50) if isinstance(none_threshold, dict) else float(none_threshold)

        if isinstance(logits, torch.Tensor):
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        else:
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        preds = []
        for i in range(probs.shape[0]):
            p_none = probs[i, 0]
            if p_none > thresh:
                preds.append(None)
            else:
                active_probs = probs[i, 1:]
                best_active_idx = int(np.argmax(active_probs)) + 1
                preds.append(ID2LABEL.get(best_active_idx, None))

        return preds


class XLMInferenceEngine:
    def __init__(self, model_dir="models/xlmroberta_local_absa", model_name="xlm-roberta-large"):
        self.device = torch.device("cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu") if HAS_TORCH else "cpu"
        self.model_dir = Path(model_dir)
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.is_weights_loaded = False
        self.rule_engine = None

        if HAS_WEAK_SUPERVISION:
            self.rule_engine = WeakSupervisionEngine(use_gemini=False)

        if HAS_TORCH:
            self._load_pytorch_pipeline()

    def _load_pytorch_pipeline(self):
        try:
            project_root = Path(__file__).resolve().parent.parent.parent
            candidate_paths = [
                self.model_dir / "pytorch_model.bin",
                project_root / "models/xlmroberta_local_absa/pytorch_model.bin",
                project_root / "models/xlmroberta_absa/pytorch_model.bin",
                project_root / "outputs/iter-03/pytorch_model.bin",
                Path("models/xlmroberta_local_absa/pytorch_model.bin")
            ]
            weights_path = None
            for p in candidate_paths:
                if p.exists():
                    weights_path = p
                    break

            if weights_path is None:
                zip_path = project_root / "outputs/iter-03/xlmroberta_absa_model.zip"
                if not zip_path.exists():
                    zip_path = Path("outputs/iter-03/xlmroberta_absa_model.zip")
                if zip_path.exists():
                    import zipfile
                    extract_dir = project_root / "models/xlmroberta_local_absa"
                    extract_dir.mkdir(parents=True, exist_ok=True)
                    print(f"[ModelLoader] Extracting model archive {zip_path} to {extract_dir}...")
                    with zipfile.ZipFile(zip_path, "r") as z:
                        z.extractall(extract_dir)
                    weights_path = extract_dir / "pytorch_model.bin"

            # Hugging Face Hub Fallback for Cloud Deployment
            if weights_path is None or not weights_path.exists():
                try:
                    from huggingface_hub import hf_hub_download
                    hf_repo = getattr(self, "hf_repo_id", "wicaksonohanif/xlm-roberta-ev-absa")
                    print(f"[ModelLoader] Mengunduh pytorch_model.bin dari Hugging Face Hub: {hf_repo}...")
                    downloaded_weights = hf_hub_download(repo_id=hf_repo, filename="pytorch_model.bin")
                    weights_path = Path(downloaded_weights)
                except Exception as hf_err:
                    print(f"[ModelLoader Warning] Gagal mengunduh dari HF Hub: {hf_err}")

            model_dir_to_use = weights_path.parent if weights_path else self.model_dir
            if (model_dir_to_use / "tokenizer_config.json").exists():
                self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir_to_use))
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            self.model = XLMRoBERTaMultiHeadClassifier(model_name=self.model_name).to(self.device)

            if weights_path and weights_path.exists():
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.is_weights_loaded = True
                print(f"[ModelLoader] Berhasil memuat PyTorch checkpoint XLM-RoBERTa dari {weights_path}")
            else:
                print(f"[ModelLoader Warning] Weights bin belum tersedia. Menggunakan Rule-Based fallback.")

            self.model.eval()
        except Exception as e:
            print(f"[ModelLoader Warning] Gagal memuat pipeline PyTorch: {e}")

    def predict_batch(self, texts, batch_size=16, progress_callback=None):
        all_results = {f"{asp}_sentiment": [] for asp in ASPECTS}
        total_items = len(texts)

        # PyTorch Inference path
        if HAS_TORCH and self.model and self.tokenizer and self.is_weights_loaded:
            num_batches = (total_items + batch_size - 1) // batch_size
            for b_idx in range(num_batches):
                start_idx = b_idx * batch_size
                end_idx = min(start_idx + batch_size, total_items)
                batch_texts = [str(t) if pd.notna(t) else "" for t in texts[start_idx:end_idx]]

                encoding = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=128,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    logits_dict = self.model(encoding["input_ids"], encoding["attention_mask"])

                for asp in ASPECTS:
                    preds = predict_with_threshold(logits_dict[asp], aspect=asp)
                    all_results[f"{asp}_sentiment"].extend(preds)

                if progress_callback:
                    progress_callback((b_idx + 1) / num_batches)

            return all_results

        # Fallback Rule-Based Engine path (Fast CPU Compatible)
        print("[ModelLoader] Menggunakan Rule-Based Engine Fallback untuk pelabelan batch...")
        for idx, text in enumerate(texts):
            text_str = str(text) if pd.notna(text) else ""
            if self.rule_engine:
                res = self.rule_engine.label_single_text(text_str)
                for asp in ASPECTS:
                    val = res.get(f"{asp}_sentiment")
                    all_results[f"{asp}_sentiment"].append(val if val and str(val).lower() != "none" else None)
            else:
                for asp in ASPECTS:
                    all_results[f"{asp}_sentiment"].append(None)

            if progress_callback and (idx + 1) % max(1, total_items // 20) == 0:
                progress_callback((idx + 1) / total_items)

        if progress_callback:
            progress_callback(1.0)

        return all_results
