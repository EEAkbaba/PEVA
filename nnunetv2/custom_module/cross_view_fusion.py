import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureProjector(nn.Module):
    def __init__(self, dim=768, bottleneck=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, dim),
        )

    def forward(self, x):
        return self.net(x)  # [B,D,H,W,C]

class CrossViewFusion(nn.Module):
    def __init__(self, dim=768, bottleneck=192):
        super().__init__()
        self.dim = dim

        self.proj_s = FeatureProjector(dim, bottleneck)
        self.proj_c = FeatureProjector(dim, bottleneck)

    @staticmethod
    def _infer_square_hw(n_tokens: int) -> int:
        hw = int(math.isqrt(n_tokens))
        if hw * hw != n_tokens:
            raise ValueError(f"Axial token count must be a square number, got N_ax={n_tokens}")
        return hw

    def forward(self, feat_ax, feat_sag, feat_cor):
        B, D_ax, N_ax, C = feat_ax.shape
        B2, D_sag, N_sag, C2 = feat_sag.shape 
        B3, D_cor, N_cor, C3 = feat_cor.shape

        if B != B2 or B != B3: 
            raise ValueError(f"Batch mismatch: ax={feat_ax.shape}, sag={feat_sag.shape}, cor={feat_cor.shape}")
        if C != C2 or C != C3:
            raise ValueError(f"Channel mismatch: ax={feat_ax.shape}, sag={feat_sag.shape}, cor={feat_cor.shape}")
        
        axial_depth = D_ax
        axial_hw = self._infer_square_hw(N_ax)
        if N_sag % axial_hw != 0:
            raise ValueError(f"N_sag={N_sag} is not divisible by axial_hw={axial_hw}.")
        if N_cor % axial_hw != 0:
            raise ValueError(f"N_cor={N_cor} is not divisible by axial_hw={axial_hw}.")
        
        sag_dp = N_sag // axial_hw
        cor_dp = N_cor // axial_hw

        F_ax = feat_ax.view(B, axial_depth, axial_hw, axial_hw, C)

        # sagittal: [B,224,28,C] -> [B,224,2,14,C] -> rotate -> resample -> [B,32,14,14,C]
        #given shapes in the comments are for BTCV
        F_sag = feat_sag.view(B, D_sag, sag_dp, axial_hw, C)
        F_sag = F_sag.permute(0, 2, 3, 1, 4)      # [B, sag_dp, axial_hw, D_sag, C]: [B,2,14,224,C]
        F_sag = F_sag.permute(0, 4, 1, 2, 3)      # [B, C, sag_dp, axial_hw, D_sag]: [B,C,2,14,224]
        F_sag = F.interpolate(
            F_sag,
            size=(axial_depth, axial_hw, axial_hw),
            mode="trilinear",
            align_corners=False,
        )
        F_sag = F_sag.permute(0, 2, 3, 4, 1)      # [B, axial_depth, axial_hw, axial_hw, C]: [B,32,14,14,C]

        # coronal: [B,224,28,C] -> [B,224,2,14,C] -> rotate -> resample -> [B,32,14,14,C]
        F_cor = feat_cor.view(B, D_cor, cor_dp, axial_hw, C)
        F_cor = F_cor.permute(0, 2, 1, 3, 4)      # [B, cor_dp, D_cor, axial_hw, C]: [B,2,224,14,C]
        F_cor = F_cor.permute(0, 4, 1, 2, 3)      # [B, C, cor_dp, D_cor, axial_hw]: [B,C,2,224,14]
        F_cor = F.interpolate(
            F_cor,
            size=(axial_depth, axial_hw, axial_hw),
            mode="trilinear",
            align_corners=False,
        )
        F_cor = F_cor.permute(0, 2, 3, 4, 1)      # [B, axial_depth, axial_hw, axial_hw, C]: [B,32,14,14,C]

        F_s_proj = self.proj_s(F_sag)
        F_c_proj = self.proj_c(F_cor)

        F_out = F_ax + F_s_proj + F_c_proj

        return {
            "fused": F_out.view(B, axial_depth, axial_hw * axial_hw, C),
            "axial_vol": F_ax,
            "sag_proj_vol": F_s_proj,
            "cor_proj_vol": F_c_proj,
        }
