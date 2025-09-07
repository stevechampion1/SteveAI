# SteveAI - Knowledge Distillation Loss Functions

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining soft and hard targets.

    This loss function implements the standard knowledge distillation approach
    where the student model learns from both the teacher's soft predictions
    and the ground truth hard labels.
    """

    def __init__(self, reduction: str = "mean"):
        """
        Initialize the distillation loss.

        Args:
            reduction: Specifies the reduction to apply to the output.
                      Options: 'none', 'mean', 'sum'
        """
        super(DistillationLoss, self).__init__()
        self.reduction = reduction
        self.kl_div = nn.KLDivLoss(reduction="none")
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 4.0,
        alpha: float = 0.7,
        beta: float = 0.3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute distillation loss.

        Args:
            student_logits: Student model logits [batch_size, seq_len, vocab_size]
            teacher_logits: Teacher model logits [batch_size, seq_len, vocab_size]
            labels: Ground truth labels [batch_size, seq_len]
            temperature: Temperature for softmax scaling
            alpha: Weight for distillation loss
            beta: Weight for hard loss (should be 1 - alpha)

        Returns:
            Tuple of (distillation_loss, hard_loss)
        """
        # Validate inputs
        if not torch.is_tensor(student_logits):
            raise TypeError("student_logits must be a tensor")
        if not torch.is_tensor(teacher_logits):
            raise TypeError("teacher_logits must be a tensor")
        if not torch.is_tensor(labels):
            raise TypeError("labels must be a tensor")

        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                f"Student and teacher logits must have same shape. "
                f"Got {student_logits.shape} and {teacher_logits.shape}"
            )

        if student_logits.shape[:2] != labels.shape:
            raise ValueError(
                f"Logits and labels must have same batch and sequence dimensions. "
                f"Got {student_logits.shape[:2]} and {labels.shape}"
            )

        # Apply temperature scaling
        student_soft = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

        # Compute distillation loss (KL divergence)
        distillation_loss = self.kl_div(student_soft, teacher_soft)
        distillation_loss = distillation_loss.sum(dim=-1) * (temperature**2)

        # Compute hard loss (cross entropy)
        # Flatten for cross entropy
        student_logits_flat = student_logits.view(-1, student_logits.size(-1))
        labels_flat = labels.view(-1)

        # Mask out padding tokens (assuming -100 is the ignore index)
        mask = labels_flat != -100
        if mask.sum() > 0:
            hard_loss = self.ce_loss(student_logits_flat, labels_flat)
            hard_loss = hard_loss * mask.float()
            hard_loss = hard_loss.view(labels.shape)
        else:
            hard_loss = torch.zeros_like(labels, dtype=torch.float)

        # Apply reduction
        if self.reduction == "mean":
            distillation_loss = distillation_loss.mean()
            hard_loss = hard_loss.mean()
        elif self.reduction == "sum":
            distillation_loss = distillation_loss.sum()
            hard_loss = hard_loss.sum()
        # 'none' reduction: return as is

        return distillation_loss, hard_loss


class AdvancedDistillationLoss(nn.Module):
    """
    Advanced distillation loss with additional features.

    This includes:
    - Attention transfer
    - Hidden state matching
    - Layer-wise distillation
    """

    def __init__(self, reduction: str = "mean"):
        super(AdvancedDistillationLoss, self).__init__()
        self.reduction = reduction
        self.kl_div = nn.KLDivLoss(reduction="none")
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self.mse_loss = nn.MSELoss(reduction="none")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_hidden_states: Optional[torch.Tensor] = None,
        teacher_hidden_states: Optional[torch.Tensor] = None,
        student_attention: Optional[torch.Tensor] = None,
        teacher_attention: Optional[torch.Tensor] = None,
        temperature: float = 4.0,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.1,
        delta: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute advanced distillation loss.

        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            labels: Ground truth labels
            student_hidden_states: Student hidden states [batch_size, seq_len, hidden_size]
            teacher_hidden_states: Teacher hidden states [batch_size, seq_len, hidden_size]
            student_attention: Student attention weights [batch_size, num_heads, seq_len, seq_len]
            teacher_attention: Teacher attention weights [batch_size, num_heads, seq_len, seq_len]
            temperature: Temperature for softmax scaling
            alpha: Weight for distillation loss
            beta: Weight for hard loss
            gamma: Weight for hidden state loss
            delta: Weight for attention loss

        Returns:
            Tuple of (distillation_loss, hard_loss, hidden_loss, attention_loss)
        """
        # Standard distillation loss
        distillation_loss, hard_loss = DistillationLoss(self.reduction)(
            student_logits, teacher_logits, labels, temperature, alpha, beta
        )

        hidden_loss = torch.tensor(0.0, device=student_logits.device)
        attention_loss = torch.tensor(0.0, device=student_logits.device)

        # Hidden state matching loss
        if student_hidden_states is not None and teacher_hidden_states is not None:
            if student_hidden_states.shape != teacher_hidden_states.shape:
                # Handle dimension mismatch by projection
                if student_hidden_states.size(-1) != teacher_hidden_states.size(-1):
                    projection = nn.Linear(
                        student_hidden_states.size(-1), teacher_hidden_states.size(-1)
                    ).to(student_hidden_states.device)
                    student_hidden_states = projection(student_hidden_states)

                hidden_loss = self.mse_loss(
                    student_hidden_states, teacher_hidden_states
                )
                if self.reduction == "mean":
                    hidden_loss = hidden_loss.mean()
                elif self.reduction == "sum":
                    hidden_loss = hidden_loss.sum()

        # Attention transfer loss
        if student_attention is not None and teacher_attention is not None:
            if student_attention.shape != teacher_attention.shape:
                # Handle dimension mismatch
                if student_attention.size(1) != teacher_attention.size(1):
                    # Average pool attention heads if number of heads differ
                    student_attention = student_attention.mean(dim=1, keepdim=True)
                    teacher_attention = teacher_attention.mean(dim=1, keepdim=True)

            attention_loss = self.mse_loss(student_attention, teacher_attention)
            if self.reduction == "mean":
                attention_loss = attention_loss.mean()
            elif self.reduction == "sum":
                attention_loss = attention_loss.sum()

        return distillation_loss, hard_loss, hidden_loss, attention_loss


class FocalDistillationLoss(nn.Module):
    """
    Focal Knowledge Distillation Loss.

    This loss focuses on hard examples by down-weighting easy examples
    and focusing on hard examples during distillation.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        """
        Initialize focal distillation loss.

        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Specifies the reduction to apply to the output
        """
        super(FocalDistillationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.kl_div = nn.KLDivLoss(reduction="none")
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 4.0,
        alpha_distill: float = 0.7,
        beta: float = 0.3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute focal distillation loss.

        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            labels: Ground truth labels
            temperature: Temperature for softmax scaling
            alpha_distill: Weight for distillation loss
            beta: Weight for hard loss

        Returns:
            Tuple of (distillation_loss, hard_loss)
        """
        # Standard distillation loss
        student_soft = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

        distillation_loss = self.kl_div(student_soft, teacher_soft)
        distillation_loss = distillation_loss.sum(dim=-1) * (temperature**2)

        # Compute focal weights based on teacher confidence
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        teacher_max_probs = teacher_probs.max(dim=-1)[0]

        # Focal weight: (1 - p_t)^gamma
        focal_weights = (1 - teacher_max_probs) ** self.gamma

        # Apply focal weighting
        distillation_loss = distillation_loss * focal_weights

        # Hard loss with focal weighting
        student_logits_flat = student_logits.view(-1, student_logits.size(-1))
        labels_flat = labels.view(-1)

        mask = labels_flat != -100
        if mask.sum() > 0:
            hard_loss = self.ce_loss(student_logits_flat, labels_flat)
            hard_loss = hard_loss * mask.float()
            hard_loss = hard_loss.view(labels.shape)

            # Apply focal weighting to hard loss
            focal_weights_flat = focal_weights.view(-1)
            hard_loss = hard_loss * focal_weights_flat
        else:
            hard_loss = torch.zeros_like(labels, dtype=torch.float)

        # Apply reduction
        if self.reduction == "mean":
            distillation_loss = distillation_loss.mean()
            hard_loss = hard_loss.mean()
        elif self.reduction == "sum":
            distillation_loss = distillation_loss.sum()
            hard_loss = hard_loss.sum()

        return distillation_loss, hard_loss


def compute_attention_loss(
    student_attention: torch.Tensor,
    teacher_attention: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute attention transfer loss.

    Args:
        student_attention: Student attention weights [batch_size, num_heads, seq_len, seq_len]
        teacher_attention: Teacher attention weights [batch_size, num_heads, seq_len, seq_len]
        reduction: Reduction method

    Returns:
        Attention loss
    """
    if student_attention.shape != teacher_attention.shape:
        raise ValueError(
            f"Attention shapes must match. "
            f"Got {student_attention.shape} and {teacher_attention.shape}"
        )

    # Compute MSE loss between attention weights
    loss = F.mse_loss(student_attention, teacher_attention, reduction="none")

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def compute_hidden_state_loss(
    student_hidden: torch.Tensor, teacher_hidden: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute hidden state matching loss.

    Args:
        student_hidden: Student hidden states [batch_size, seq_len, hidden_size]
        teacher_hidden: Teacher hidden states [batch_size, seq_len, hidden_size]
        reduction: Reduction method

    Returns:
        Hidden state loss
    """
    if student_hidden.shape != teacher_hidden.shape:
        # Handle dimension mismatch
        if student_hidden.size(-1) != teacher_hidden.size(-1):
            projection = nn.Linear(student_hidden.size(-1), teacher_hidden.size(-1)).to(
                student_hidden.device
            )
            student_hidden = projection(student_hidden)

    # Compute MSE loss between hidden states
    loss = F.mse_loss(student_hidden, teacher_hidden, reduction="none")

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


# Example usage and testing functions
def test_distillation_loss():
    """Test the distillation loss functions."""
    logger.info("Testing distillation loss functions...")

    # Create dummy data
    batch_size, seq_len, vocab_size = 2, 10, 1000
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Test basic distillation loss
    loss_fn = DistillationLoss()
    dist_loss, hard_loss = loss_fn(student_logits, teacher_logits, labels)

    logger.info(f"Distillation loss: {dist_loss.item():.4f}")
    logger.info(f"Hard loss: {hard_loss.item():.4f}")

    # Test advanced distillation loss
    hidden_size = 512
    student_hidden = torch.randn(batch_size, seq_len, hidden_size)
    teacher_hidden = torch.randn(batch_size, seq_len, hidden_size)

    num_heads = 8
    student_attention = torch.randn(batch_size, num_heads, seq_len, seq_len)
    teacher_attention = torch.randn(batch_size, num_heads, seq_len, seq_len)

    adv_loss_fn = AdvancedDistillationLoss()
    dist_loss, hard_loss, hidden_loss, attention_loss = adv_loss_fn(
        student_logits,
        teacher_logits,
        labels,
        student_hidden,
        teacher_hidden,
        student_attention,
        teacher_attention,
    )

    logger.info(f"Advanced - Distillation loss: {dist_loss.item():.4f}")
    logger.info(f"Advanced - Hard loss: {hard_loss.item():.4f}")
    logger.info(f"Advanced - Hidden loss: {hidden_loss.item():.4f}")
    logger.info(f"Advanced - Attention loss: {attention_loss.item():.4f}")

    # Test focal distillation loss
    focal_loss_fn = FocalDistillationLoss()
    dist_loss, hard_loss = focal_loss_fn(student_logits, teacher_logits, labels)

    logger.info(f"Focal - Distillation loss: {dist_loss.item():.4f}")
    logger.info(f"Focal - Hard loss: {hard_loss.item():.4f}")

    logger.info("All distillation loss tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_distillation_loss()
