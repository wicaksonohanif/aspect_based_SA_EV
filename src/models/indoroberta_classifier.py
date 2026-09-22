"""
IndoRoBERTa Multi-Head Classification Model for ABSA
Spec Compliance: specs/03_model_training.spec.md
Model Backbone: indolem/indobert-base-uncased (110M Parameter Encoder)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

LABEL2ID = {'None': 0, 'none': 0, 'positif': 1, 'netral': 2, 'negatif': 3}
ID2LABEL = {0: 'None', 1: 'positif', 2: 'netral', 3: 'negatif'}
ASPECTS = ['infra', 'ekonomi', 'kualitas', 'purnajual']


class IndoRoBERTaMultiHeadClassifier(nn.Module):
    """
    IndoRoBERTa Multi-Head Classifier untuk Aspect-Based Sentiment Analysis.
    Berisi 1 shared Transformer Encoder (110M) dan 4 Classification Heads terpisah
    untuk aspek: 'infra', 'ekonomi', 'kualitas', 'purnajual'.
    """

    def __init__(
        self,
        model_name: str = "indolem/indobert-base-uncased",
        num_classes: int = 4,
        dropout_prob: float = 0.2,
        aspects: list[str] | None = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.aspects = aspects if aspects is not None else ASPECTS
        self.num_classes = num_classes

        # Base Transformer Encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # 4 Multi-Task Classification Heads (Linear + Dropout)
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
        Forward pass melalui encoder dan 4 classification heads.

        Args:
            input_ids: Tensor shape [batch_size, max_seq_len]
            attention_mask: Tensor shape [batch_size, max_seq_len]
            labels: Dict mapping aspect -> Tensor shape [batch_size] atau Tensor [batch_size, 4]
            class_weights: Dict mapping aspect -> Tensor shape [4] (Opsional untuk menangani imbalance)

        Returns:
            Dict berisi 'logits' (dict aspect -> Tensor [batch_size, 4])
            dan 'loss' (Tensor scalar) jika labels diberikan.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extrak CLS token representation [batch_size, hidden_size]
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            cls_output = outputs.pooler_output
        else:
            cls_output = outputs.last_hidden_state[:, 0, :]

        logits = {}
        total_loss = 0.0

        for aspect in self.aspects:
            aspect_logits = self.heads[aspect](cls_output)
            logits[aspect] = aspect_logits

            if labels is not None:
                if isinstance(labels, dict) and aspect in labels:
                    target = labels[aspect]
                elif isinstance(labels, torch.Tensor) and labels.ndim == 2:
                    aspect_idx = self.aspects.index(aspect)
                    target = labels[:, aspect_idx]
                else:
                    target = None

                if target is not None:
                    weight = class_weights[aspect] if class_weights and aspect in class_weights else None
                    criterion = nn.CrossEntropyLoss(weight=weight)
                    loss_aspect = criterion(aspect_logits, target)
                    total_loss += loss_aspect

        output_dict = {"logits": logits}
        if labels is not None:
            output_dict["loss"] = total_loss

        return output_dict


if __name__ == "__main__":
    print("IndoRoBERTa Multi-Head Classifier module initialized successfully!")
