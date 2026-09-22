"""
SahabatAI-8B Multi-Head Classifier with QLoRA for ABSA (Discriminative LLM Mode)
Spec Compliance: specs/03_model_training.spec.md
Model Backbone: SahabatAI/SahabatAI-Instruct-8B (8B Parameter Decoder)
Quantization & Fine-Tuning: 4-bit NF4 QLoRA Partial Fine-Tuning
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

LABEL2ID = {'None': 0, 'none': 0, 'positif': 1, 'netral': 2, 'negatif': 3}
ID2LABEL = {0: 'None', 1: 'positif', 2: 'netral', 3: 'negatif'}
ASPECTS = ['infra', 'ekonomi', 'kualitas', 'purnajual']


class SahabatAIMultiHeadClassifier(nn.Module):
    """
    SahabatAI-8B Multi-Head Classifier (Discriminative Mode) untuk ABSA.
    Menggunakan LLM Decoder 8B yang di-quantize ke 4-bit NF4 dan di-fine-tune via QLoRA.
    Extrak hidden state dari token terakhir untuk diklasifikasikan oleh 4 Classification Head terpisah.
    """

    def __init__(
        self,
        model_name: str = "SahabatAI/SahabatAI-Instruct-8B",
        num_classes: int = 4,
        dropout_prob: float = 0.1,
        aspects: list[str] | None = None,
        load_in_4bit: bool = True,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.model_name = model_name
        self.aspects = aspects if aspects is not None else ASPECTS
        self.num_classes = num_classes

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

        # Load Base AutoModel (Hidden states output)
        self.backbone = AutoModel.from_pretrained(
            model_name,
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

        # 4 Multi-Task Classification Heads
        self.heads = nn.ModuleDict({
            aspect: nn.Sequential(
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_size, num_classes)
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
        Forward pass melalui SahabatAI-8B backbone dan 4 classification heads.
        Mengambil hidden state token terakhir (last non-padded token).
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Ambil representation dari token non-padded terakhir per baris batch
        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        last_token_hidden_states = last_hidden_state[torch.arange(batch_size), sequence_lengths]

        logits = {}
        total_loss = 0.0

        for aspect in self.aspects:
            # Pindahkan head ke device yang sama dengan hidden state
            head = self.heads[aspect].to(last_token_hidden_states.device)
            aspect_logits = head(last_token_hidden_states)
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
                    weight = class_weights[aspect].to(aspect_logits.device) if class_weights and aspect in class_weights else None
                    criterion = nn.CrossEntropyLoss(weight=weight)
                    loss_aspect = criterion(aspect_logits, target)
                    total_loss += loss_aspect

        output_dict = {"logits": logits}
        if labels is not None:
            output_dict["loss"] = total_loss

        return output_dict


if __name__ == "__main__":
    print("SahabatAI Multi-Head Classifier module initialized successfully!")
