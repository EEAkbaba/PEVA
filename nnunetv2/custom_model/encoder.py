import torch
import numpy as np
import cv2
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModel
from albumentations.pytorch import ToTensorV2
import albumentations as A
from argparse import Namespace
import os

class Dinov2Encoder(nn.Module):
    def __init__(self, freeze = True, model_name = 'facebook/dinov2-base'):
        super(Dinov2Encoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.freeze = freeze

    def forward(self, x):
        outputs = []
        if self.freeze:
            with torch.no_grad(): 
                x = self.model.embeddings(x) #patch로 나누고 embedding (positional encoding도 하도록 수정)
                for i, layer in enumerate(self.model.encoder.layer):
                    x = layer(x)[0] if isinstance(layer(x), tuple) else layer(x)
                    if i+1 in [3, 6, 9, 12]:
                        outputs.append(x)
        else:
            x = self.model.embeddings(x) #patch로 나누고 embedding (positional encoding도 하도록 수정)
            for i, layer in enumerate(self.model.encoder.layer):
                x = layer(x)[0] if isinstance(layer(x), tuple) else layer(x)
                if i+1 in [3, 6, 9, 12]:
                    outputs.append(x)
                    
        return outputs

if __name__ == '__main__':

    dummy_image = torch.randn(1, 3, 1024, 1024)  # (batch_size, channels, height, width)
    encoder = SamEncoder()
    features = encoder(dummy_image)

    print(len(features))
    print(features[0].shape)
