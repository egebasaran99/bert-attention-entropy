# src/entropy.py
# Person C — Entropy computation
# Called by inference.py. Computes mean attention entropy per layer
# from BERT's raw attention tensors.
 
import torch
 
 
def attention_entropy(attn_matrix: torch.Tensor) -> float:
    """
    Computes mean Shannon entropy over all attention heads for one layer.
 
    Args:
        attn_matrix: tensor of shape (1, num_heads, seq_len, seq_len)
            Each row in the (seq_len x seq_len) matrix is a probability
            distribution over tokens that sums to 1.
 
    Returns:
        A single float: mean entropy across all heads and token positions.
 
    Notes:
        - High entropy = attention is spread uniformly across tokens
          (BERT is uncertain / not focusing on any particular token).
        - Low entropy = attention is concentrated on a few tokens
          (BERT has found relevant structure to attend to).
        - We clamp values to 1e-9 before taking the log to avoid log(0).
    """
    attn = attn_matrix.squeeze(0)           # (num_heads, seq_len, seq_len)
    attn = attn.clamp(min=1e-9)             # numerical stability
    entropy = -torch.sum(attn * torch.log(attn), dim=-1)  # (num_heads, seq_len)
    return entropy.mean().item()            # scalar
 
 
def compute_entropy_per_layer(attentions: tuple) -> list:
    """
    Applies attention_entropy to each of BERT's 12 layers.
 
    Args:
        attentions: tuple of 12 tensors, one per layer,
            each of shape (1, num_heads, seq_len, seq_len).
 
    Returns:
        List of 12 floats — one entropy value per layer.
    """
    return [attention_entropy(layer_attn) for layer_attn in attentions]