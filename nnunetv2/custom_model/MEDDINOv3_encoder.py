import os
import sys
import torch
import torch.nn as nn


class MedDINOv3Encoder(nn.Module):
    def __init__(
        self,
        checkpoint_path="/ari/users/eakbaba/Med_Project/PEVA/pretrained_models/MedDINOv3_pretrained.pth",
        interaction_indices=(2, 5, 8, 11),
        freeze=True,
        drop_path_rate=0.2,
        layerscale_init=1.0e-5,
    ):
        super().__init__()

        self.interaction_indices = list(interaction_indices)

        dino_root = "/ari/users/eakbaba/Med_Project/PEVA/nnunetv2/training/nnUNetTrainer/dinov3"
        if dino_root not in sys.path:
            sys.path.append(dino_root)

        from dinov3.models.vision_transformer import vit_base

        self.encoder = vit_base(
            drop_path_rate=drop_path_rate,
            layerscale_init=layerscale_init,
            n_storage_tokens=4,
            qkv_bias=False,
            mask_k_bias=True,
        )

        chkpt = torch.load(checkpoint_path, map_location="cpu")

        state_dict = chkpt['teacher']
        state_dict = {
            k.replace('backbone.', ''): v
            for k, v in state_dict.items()
            if 'ibot' not in k and 'dino_head' not in k
        }
        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=True)

        print("Loaded MedDINOv3 encoder")
        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.freeze = freeze

    def forward(self, x):

        features = self.encoder.get_intermediate_layers(x, n=self.interaction_indices, reshape=False)

        return list(features)

if __name__ == "__main__":

    enc = MedDINOv3Encoder(freeze=True)
    enc.eval()

    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        feats = enc(x)

    print(type(feats))
    print("num features:", len(feats))

    for i, f in enumerate(feats):
        print(f"Feature {i}: {f.shape}")