import torch
import numpy as np

from fyp_modules.style_fusion import StyleFusion
from fyp_modules.emotion_encoder import EmotionEncoder
from fyp_modules.prosody_extractor import extract_prosody

# fake speaker embedding (from Higgs — 256 dim)
speaker_emb = torch.rand(1, 256)

# real emotion embedding
emo = EmotionEncoder()
emotion_emb_np = emo.extract("your_audio.wav")
emotion_emb = torch.tensor(emotion_emb_np).float()

# real prosody
prosody_vec = extract_prosody("your_audio.wav")

fusion = StyleFusion()

style_emb = fusion(speaker_emb, emotion_emb, prosody_vec)
print("Style Embedding Shape:", style_emb.shape)
print(style_emb[0][:10])   # show first 10 values

