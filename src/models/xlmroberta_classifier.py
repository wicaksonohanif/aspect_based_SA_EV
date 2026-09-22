"""
XLM-RoBERTa Large Multi-Head Classification Model for ABSA
Spec Compliance: specs/03_model_training.spec.md
Model Backbone: xlm-roberta-large (550M Parameter Multilingual Encoder)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

LABEL2ID = {'None': 0, 'none': 0, 'positif': 1, 'netral': 2, 'negatif': 3}
ID2LABEL = {0: 'None', 1: 'positif', 2: 'netral', 3: 'negatif'}
ASPECTS = ['infra', 'ekonomi', 'kualitas', 'purnajual']


class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss dengan gamma=1.5 untuk penyeimbangan sampel mayoritas (None).
    """
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 1.5):
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


class XLMRoBERTaMultiHeadClassifier(nn.Module):
    """
    XLM-RoBERTa-Large Multi-Head Classifier (550M Parameter Encoder) untuk ABSA.
    Menggunakan Concatenated CLS + Mean Pooling (2048 dim) dan 4 Multi-Task GELU MLP Heads.
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-large",
        num_classes: int = 4,
        dropout_prob: float = 0.1,
        aspects: list[str] | None = None,
        focal_gamma: float = 1.5,
    ):
        super().__init__()
        self.model_name = model_name
        self.aspects = aspects if aspects is not None else ASPECTS
        self.num_classes = num_classes
        self.focal_gamma = focal_gamma

        # Base XLM-RoBERTa Large Transformer Encoder (550M)
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 1024 for xlm-roberta-large

        # 4 Multi-Task Classification Heads (2-Layer MLP dengan GELU)
        # Input hidden size x 2 (Concatenated CLS + Mean Pooling: 1024 x 2 = 2048)
        self.heads = nn.ModuleDict({
            aspect: nn.Sequential(
                nn.Linear(hidden_size * 2, 256),
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
        Forward pass melalui XLM-RoBERTa Large encoder dan 4 GELU MLP classification heads.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # 1. CLS Token Representation
        cls_token = last_hidden[:, 0, :]

        # 2. Mean Pooling Representation
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        # 3. Concatenate CLS + Mean Pooled Features [batch_size, 2048]
        pooled_features = torch.cat([cls_token, mean_pooled], dim=-1)

        logits = {}
        total_loss = 0.0

        for aspect in self.aspects:
            aspect_logits = self.heads[aspect](pooled_features)
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
                    alpha = class_weights[aspect] if class_weights and aspect in class_weights else None
                    criterion = FocalLoss(alpha=alpha, gamma=self.focal_gamma)
                    loss_aspect = criterion(aspect_logits, target)
                    total_loss += loss_aspect

        output_dict = {"logits": logits}
        if labels is not None:
            output_dict["loss"] = total_loss

        return output_dict


if __name__ == "__main__":
    print("XLM-RoBERTa Large Multi-Head Classifier initialized successfully!")
