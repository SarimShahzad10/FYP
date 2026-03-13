import torch
from typing import Optional, Union

from fyp_modules.prosody_extractor import extract_prosody
from fyp_modules.emotion_encoder import EmotionEncoder
from fyp_modules.style_fusion import StyleFusion


def _prepare_tensor(value: Union[torch.Tensor, float], device: torch.device) -> torch.Tensor:
    """Ensure input is a 2D tensor on the desired device."""
    tensor = value if torch.is_tensor(value) else torch.tensor(value, dtype=torch.float32)
    tensor = tensor.to(device).float()
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def build_style_embedding(
    model,
    audio_in_ids: torch.Tensor,
    ref_audio_path: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Build the fused style embedding used by the Higgs model.

    Args:
        model: HiggsAudioModel instance.
        audio_in_ids: torch.Tensor of shape (num_codebooks, seq_len) representing reference audio tokens.
        ref_audio_path: Path to the reference audio file.
        device: torch.device where the model runs.

    Returns:
        torch.Tensor of shape (hidden_size,) representing the fused style embedding.
    """

    if audio_in_ids is None:
        raise ValueError("audio_in_ids is required to compute the speaker embedding.")

    device = torch.device(device)

    # 1. Extract prosody
    prosody_vec = extract_prosody(ref_audio_path)  # numpy (4,)
    prosody = _prepare_tensor(prosody_vec, device)

    # 2. Extract emotion embedding
    emo_model = EmotionEncoder()
    emo_emb_np = emo_model.extract(ref_audio_path)  # numpy (128,)
    emotion_emb = _prepare_tensor(emo_emb_np, device)

    # 3. Compute speaker embedding from Higgs audio_in_ids
    audio_in_ids = audio_in_ids.to(device)
    if audio_in_ids.dim() == 3:
        audio_in_ids = audio_in_ids.squeeze(0)

    with torch.no_grad():
        spk_seq = model._embed_audio_ids(audio_in_ids)  # (seq_len, hidden)
        speaker_emb = spk_seq.mean(dim=0, keepdim=True)  # (1, hidden)

    # 4. Fuse them
    fusion = StyleFusion()
    fusion = fusion.to(device)
    with torch.no_grad():
        style_emb = fusion(speaker_emb, emotion_emb, prosody)  # (1, hidden)

    return style_emb.squeeze(0)


if __name__ == "__main__":
    raise SystemExit(
        "This module provides build_style_embedding(). Import and use it inside your inference pipeline."
    )


