"""Two-stream 3D-CNN and dictionary-semantic logistic label embedding."""
import torch
from torch import Tensor, nn
import torch.nn.functional as F

class VisualEncoder(nn.Module):
    """Independent body/hand 3D CNNs; input B,T,3,H,W; output unit B,d_v."""
    def __init__(self, d_v: int = 512, width: int = 32) -> None:
        super().__init__()
        def backbone() -> nn.Sequential:
            """Build a spatial-temporal feature extractor with adaptive pooling."""
            layers: list[nn.Module] = []
            channels = 3
            for out in (width, width * 2, width * 4):
                layers.extend([nn.Conv3d(channels, out, 3, stride=(1,2,2), padding=1),
                               nn.GroupNorm(1, out), nn.ReLU()])
                channels = out
            return nn.Sequential(*layers, nn.AdaptiveAvgPool3d(1), nn.Flatten())
        self.body = backbone()
        self.hand = backbone()
        self.projection = nn.Linear(width * 8, d_v)

    def forward(self, body_frames: Tensor, hand_frames: Tensor) -> Tensor:
        """Fuse synchronized streams, which may have different spatial resolutions."""
        for x in (body_frames, hand_frames):
            if x.ndim != 5 or x.shape[2] != 3 or not torch.isfinite(x).all():
                raise ValueError('Expected finite B,T,3,H,W frames')
        if body_frames.shape[:2] != hand_frames.shape[:2]:
            raise ValueError('Body and hands must share B,T')
        b = self.body(body_frames.transpose(1,2))
        h = self.hand(hand_frames.transpose(1,2))
        return F.normalize(self.projection(torch.cat((b,h), dim=-1)), dim=-1)

class SemanticEncoder(nn.Module):
    """Project cached BERT embeddings 768→64 and concatenate 53 binary attributes."""
    def __init__(self) -> None:
        super().__init__()
        self.text_projection = nn.Linear(768,64)

    def forward(self, text_embeddings: Tensor, attributes: Tensor) -> Tensor:
        """Return K,117 normalized class prototypes; reject unknown/nonbinary values."""
        if text_embeddings.ndim != 2 or text_embeddings.shape[1] != 768:
            raise ValueError('Expected K,768 text embeddings')
        if attributes.shape != (text_embeddings.shape[0],53):
            raise ValueError('Expected matching K,53 attributes')
        if not torch.isfinite(text_embeddings).all() or not ((attributes == 0) | (attributes == 1)).all():
            raise ValueError('Embeddings must be finite; attributes must be binary')
        return F.normalize(torch.cat((self.text_projection(text_embeddings),
                                     attributes.to(text_embeddings.dtype)),dim=-1),dim=-1)

class ZSSLRCompatibilityModel(nn.Module):
    """LLE score v @ W @ c.T, with a learnable d_v × 117 compatibility matrix."""
    def __init__(self, d_v: int = 512, width: int = 32) -> None:
        super().__init__()
        self.visual = VisualEncoder(d_v,width)
        self.semantic = SemanticEncoder()
        self.W = nn.Parameter(torch.empty(d_v,117))
        nn.init.xavier_uniform_(self.W)

    def score_features(self, v: Tensor, text: Tensor, attributes: Tensor) -> Tensor:
        """Score precomputed visual features against ordered candidate classes."""
        return v @ self.W @ self.semantic(text,attributes).T

    def forward(self, body_frames: Tensor, hand_frames: Tensor,
                text_embeddings: Tensor, attributes: Tensor) -> Tensor:
        """Return B,K raw compatibility logits, without softmax."""
        return self.score_features(self.visual(body_frames,hand_frames),text_embeddings,attributes)
