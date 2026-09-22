"""
SahabatAI-8B Multi-Head Classifier with QLoRA for ABSA (Discriminative LLM Mode)
Spec Compliance: specs/03_model_training.spec.md
Model Backbone: Sahabat-AI/SahabatAI-Instruct-8B (8B Parameter Decoder)
Quantization & Fine-Tuning: 4-bit NF4 QLoRA Partial Fine-Tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

LABEL2ID = {'None': 0, 'none': 0, 'positif': 1, 'netral': 2, 'negatif': 3}
ID2LABEL = {0: 'None', 1: 'positif', 2: 'netral', 3: 'negatif'}
ASPECTS = ['infra', 'ekonomi', 'kualitas', 'purnajual']


class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss untuk menekan loss dari sampel mayoritas (None).
    """
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(inputs, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = torch.clamp(pt, min=1e-8, max=1.0)
        
        focal_weight = (1.0 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_weight = self.alpha.to(inputs.device)[targets]
            focal_weight = focal_weight * alpha_weight

        loss = -focal_weight * torch.log(pt)
        return loss.mean()


class SahabatAIMultiHeadClassifier(nn.Module):
    """
    SahabatAI-8B Multi-Head Classifier (Discriminative Mode) untuk ABSA.
    Menggunakan LLM Decoder 8B (Sahabat-AI/SahabatAI-Instruct-8B) 4-bit NF4 QLoRA.
    Extrak hidden state token terakhir + GELU MLP classification head.
    """

    def __init__(
        self,
        model_name: str = "Sahabat-AI/SahabatAI-Instruct-8B",
        num_classes: int = 4,
        dropout_prob: float = 0.1,
        aspects: list[str] | None = None,
        load_in_4bit: bool = True,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.model_name = model_name
        self.aspects = aspects if aspects is not None else ASPECTS
        self.num_classes = num_classes
        self.focal_gamma = focal_gamma

        # Quantization Config (4-bit NF4 for Kaggle T4 GPU 16GB VRAM)
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        # Try loading model with fallback options if HF identifier differs
        try:
            self.backbone = AutoModel.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        except Exception as e:
            fallback_name = "GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct"
            print(f"⚠️ Warning: Failed to load '{model_name}'. Trying fallback '{fallback_name}'... Error: {e}")
            self.backbone = AutoModel.from_pretrained(
                fallback_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )

        hidden_size = self.backbone.config.hidden_size

        if load_in_4bit and use_lora:
            self.backbone = prepare_model_for_kbit_training(self.backbone)

        if use_lora:
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.backbone = get_peft_model(self.backbone, peft_config)

        # 4 Multi-Task GELU MLP Classification Heads
        self.heads = nn.ModuleDict({
            aspect: nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(256, num_classes)
            )
            for aspect in self.aspects
        })

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: dict[str, torch.Tensor] | torch.Tensor | None = None,
        class_weights: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass melalui SahabatAI-8B backbone dan 4 MLP classification heads.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        last_token_hidden = last_hidden_state[torch.arange(batch_size), sequence_lengths]

        logits = {}
        total_loss = 0.0

        for aspect in self.aspects:
            head = self.heads[aspect].to(last_token_hidden.device)
            aspect_logits = head(last_token_hidden)
            logits[aspect] = aspect_logits

            if labels is not None:
                if isinstance(labels, dict) and aspect in labels:
                    target = labels[aspect].to(aspect_logits.device)
                elif isinstance(labels, torch.Tensor) and labels.ndim == 2:
                    aspect_idx = self.aspects.index(aspect)
                    target = labels[:, aspect_idx].to(aspect_logits.device)
                else:
                    target = None

                if target is not None:
                    alpha = class_weights[aspect].to(aspect_logits.device) if class_weights and aspect in class_weights else None
                    criterion = FocalLoss(alpha=alpha, gamma=self.focal_gamma)
                    loss_aspect = criterion(aspect_logits, target)
                    total_loss += loss_aspect

        output_dict = {"logits": logits}
        if labels is not None:
            output_dict["loss"] = total_loss

        return output_dict


if __name__ == "__main__":
    print("SahabatAI Multi-Head Classifier with Focal Loss initialized successfully!")
