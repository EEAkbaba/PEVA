import torch
import torch.nn as nn
import os
import torch.nn.functional as F

class Conv2DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        p = (k - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)
    

def reshape_transformer_output_2d(z, embed_dim, h, w, d, patch_size):
    batch, D, num, dim = z.shape
    num_patches = (h // patch_size) * (w // patch_size)
    if num == num_patches + 1:
        z = z[:, :, 1:, :]

    h_patches = h // patch_size
    w_patches = w // patch_size
    z = z.reshape(batch * D, num_patches, dim)
    z = z.permute(0, 2, 1).contiguous()
    z = z.view(batch * D, dim, h_patches, w_patches)
    return z


class Decoder2D_16(nn.Module):

    def __init__(self, embed_dim=768, patch_size=16, output_dim=14, input_dim=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.in_ch = embed_dim * input_dim # for BraTS
        self.patch_size = patch_size
        self.output_dim = output_dim

        self.pre_reduce = nn.Sequential(
            nn.Conv2d(self.in_ch, 64, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=64),
            nn.ReLU(inplace=True),
        )

        self.refine = nn.Sequential(
            Conv2DBlock(64, 64, k=3),
            Conv2DBlock(64, 64, k=3),
        )
        self.head = nn.Conv2d(64, output_dim, kernel_size=1, bias=True)

    def forward(self, x, z):
        B, D, C, H, W = x.shape
        assert C == self.input_dim, f"Model input_dim={self.input_dim} but given volume C={C}. BraTS=4, Synapse=1."

        z = reshape_transformer_output_2d(z, embed_dim=self.in_ch, h=H, w=W, d=D, patch_size=self.patch_size) # (B*D, in_ch, H_p, W_p)

        z = self.pre_reduce(z)
        z = nn.functional.interpolate(z, size=(H, W), mode="bilinear", align_corners=False)

        z = self.refine(z)        
        logits2d = self.head(z)   

        logits = logits2d.view(B, D, self.output_dim, H, W).permute(0, 2, 1, 3, 4).contiguous()
        return logits

class SingleDeconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose2d(
            in_planes,
            out_planes,
            kernel_size=2,
            stride=2,
            padding=0
        )

    def forward(self, x):
        return self.block(x)

class SingleConv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv2d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2
        )

    def forward(self, x):
        return self.block(x)
    
class Deconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv2DBlock(in_planes, out_planes),
            SingleConv2DBlock(out_planes, out_planes, kernel_size),
            nn.GroupNorm(num_groups=16, num_channels=out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class Decoder2D_16_multilayer(nn.Module):
    def __init__(self, embed_dim=768, patch_size=16, output_dim=14, input_dim=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.in_ch = embed_dim * input_dim
        self.patch_size = patch_size
        self.output_dim = output_dim

        # raw image branch
        self.decoder0 = nn.Sequential(
            Conv2DBlock(self.input_dim, 32, 3),
            Conv2DBlock(32, 64, 3)
        )

        self.decoder3 = nn.Sequential(
            Deconv2DBlock(self.in_ch, 512), 
            Deconv2DBlock(512, 256),
            Deconv2DBlock(256, 128) 
        )

        self.decoder6 = nn.Sequential(
            Deconv2DBlock(self.in_ch, 512), 
            Deconv2DBlock(512, 256)
        )

        self.decoder9 = Deconv2DBlock(self.in_ch, 512)

        self.decoder12_upsampler = SingleDeconv2DBlock(self.in_ch, 512) 

        self.decoder9_upsampler = nn.Sequential(
            Conv2DBlock(1024, 512),
            Conv2DBlock(512, 512),
            SingleDeconv2DBlock(512, 256) 
        )

        self.decoder6_upsampler = nn.Sequential(
            Conv2DBlock(512, 256),
            Conv2DBlock(256, 256),
            SingleDeconv2DBlock(256, 128)
        )

        self.decoder3_upsampler = nn.Sequential(
            Conv2DBlock(256, 128),
            Conv2DBlock(128, 128),
            SingleDeconv2DBlock(128, 64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )

        self.decoder0_header = nn.Sequential(
            Conv2DBlock(128, 64),
            Conv2DBlock(64, 64),
            SingleConv2DBlock(64, output_dim, 1)
        )

    def forward(self, x, features):
        z0, z3, z6, z9, z12 = x, *features
        B, D, C, H, W = x.shape

        assert C == self.input_dim, f"Model input_dim={self.input_dim} but given volume C={C}"
        z0 = z0.reshape(B * D, C, H, W).contiguous()
        z3 = reshape_transformer_output_2d(z3,  embed_dim=self.in_ch, h=H, w=W, d=D, patch_size=self.patch_size)
        z6 = reshape_transformer_output_2d(z6,  embed_dim=self.in_ch, h=H, w=W, d=D, patch_size=self.patch_size)
        z9 = reshape_transformer_output_2d(z9,  embed_dim=self.in_ch, h=H, w=W, d=D, patch_size=self.patch_size)
        z12 = reshape_transformer_output_2d(z12, embed_dim=self.in_ch, h=H, w=W, d=D, patch_size=self.patch_size)

        z12 = self.decoder12_upsampler(z12)                     # 14 -> 28, ch 512
        z9  = self.decoder9(z9)                                # 14 -> 28, ch 512
        z9  = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))   # 28 -> 56, ch 256

        z6  = self.decoder6(z6)                                # 14 -> 56, ch 256
        z6  = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))    # 56 -> 112, ch 128

        z3  = self.decoder3(z3)                                # 14 -> 112, ch 128
        z3  = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))    # 112 -> 224, ch 64

        z0  = self.decoder0(z0)                                # full-res, ch 64
        logits2d = self.decoder0_header(torch.cat([z0, z3], dim=1))  # (B*D, out, H, W)

        logits = logits2d.view(B, D, self.output_dim, H, W).permute(0, 2, 1, 3, 4).contiguous()
        return logits

if __name__ == "__main__":
    B, D, C, H, W = 1, 32, 1, 224, 224
    patch = 16
    Hp, Wp = H // patch, W // patch
    N = Hp * Wp + 1
    dim = 768
    out_ch = 14
    x = torch.randn(B, D, C, H, W)
    dummy_tokens = torch.randn(B, D, N, dim)
    dec = Decoder2D_16(embed_dim=dim, patch_size=patch, output_dim=out_ch)
    y = dec(x, dummy_tokens)
    print("x:", x.shape)
    print("tokens:", dummy_tokens.shape)
    print("y:", y.shape)  # (B, out_ch, D, H, W)
