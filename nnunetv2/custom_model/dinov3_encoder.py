import torch
import numpy as np
import cv2

import torch.nn as nn
from transformers import AutoModel, BeitModel, BlipModel, CLIPModel, SamModel
from albumentations.pytorch import ToTensorV2
import albumentations as A
from argparse import Namespace


class Dinov3Encoder(nn.Module):
    def __init__(self, freeze = True, model_name = 'facebook/dinov3-vitb16-pretrain-lvd1689m'):
        super(Dinov3Encoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.freeze = freeze

        # get number of register token from config (HF DINOv3)
        self.n_register = getattr(self.model.config, "num_register_tokens", 0)

        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def forward(self, x):
        """
        x: (B, 3, H, W)
        returns: (B, T, dim)  # T = CLS + (register tokens) + patch tokens
        """
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            out = self.model(x, return_dict=True)
            z = out.last_hidden_state              # (B, 1 + R + Npatch, dim)
            z = z[:, 1 + self.n_register:, :]      # patch-only
            return z
        

class Dinov3Encoder_multilayer(nn.Module):
    def __init__(self, freeze=True, model_name='facebook/dinov3-vitb16-pretrain-lvd1689m'):
        super(Dinov3Encoder_multilayer, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.freeze = freeze

        # HF DINOv3 config
        self.n_register = getattr(self.model.config, "num_register_tokens", 0)

        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def forward(self, x):
        """
        x: (B, 3, H, W)

        returns:
            outputs = [z3, z6, z9, z12]
            each z: (B, Npatch, dim)
        """
        outputs = []
        
        if self.freeze:
            with torch.no_grad():
                pixel_values = x
                position_embeddings = self.model.rope_embeddings(pixel_values)
                x = self.model.embeddings(pixel_values)

                for i, layer in enumerate(self.model.model.layer):
                    out = layer(x, position_embeddings=position_embeddings)
                    x = out[0] if isinstance(out, tuple) else out

                    if i + 1 in [3, 6, 9, 12]:
                        outputs.append(x[:, 1 + self.n_register:, :])
        else:
            pixel_values = x
            position_embeddings = self.model.rope_embeddings(pixel_values)
            x = self.model.embeddings(pixel_values)

            for i, layer in enumerate(self.model.model.layer):
                out = layer(x, position_embeddings=position_embeddings)
                x = out[0] if isinstance(out, tuple) else out

                if i + 1 in [3, 6, 9, 12]:
                    outputs.append(x[:, 1 + self.n_register:, :])

        return outputs