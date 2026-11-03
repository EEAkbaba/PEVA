import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(
            in_planes, 
            out_planes, 
            kernel_size=(1, 2, 2),  # (H, W, D)
            stride=(1, 2, 2),       # Stride matches kernel size
            padding=(0, 0, 0)       # No padding
        )

    def forward(self, x):
        return self.block(x)
    
class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv3d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2
        )

    def forward(self, x):
        return self.block(x)

class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.GroupNorm(num_groups = 16, num_channels = out_planes),
            #nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.GroupNorm(num_groups = 16, num_channels = out_planes),
            #nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

def reshape_transformer_output_3d(z, embed_dim, h, w, d, patch_size):
    batch, D, num, dim = z.shape  
    num_patches = (h // patch_size) * (w // patch_size)
    if num == num_patches + 1:  # CLS 토큰이 있는 경우
        z = z[:, :, 1:, :]  # CLS 토큰 제거
    h_patches = h // patch_size 
    w_patches = w // patch_size 
    return z.permute(0, 3, 1, 2).reshape(batch, embed_dim, D, h_patches, w_patches) #num => h * w

class UNETRDecoder3D_14(nn.Module):
    def __init__(self, embed_dim=768, patch_size=14, input_dim=4, output_dim=4, ds = True):
        super().__init__()
        self.ds = ds
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # Decoder layers
        self.decoder0 = nn.Sequential(
            Conv3DBlock(self.input_dim, 32, 3),
            Conv3DBlock(32, 64, 3)
        )

        self.decoder3 = nn.Sequential(
            Deconv3DBlock(embed_dim * self.input_dim, 512),
            Deconv3DBlock(512, 256),
            Deconv3DBlock(256, 128)
        )

        self.decoder6 = nn.Sequential(
            Deconv3DBlock(embed_dim * self.input_dim, 512),
            Deconv3DBlock(512, 256)
        )

        self.decoder9 = Deconv3DBlock(embed_dim * self.input_dim, 512)

        self.decoder12_upsampler = SingleDeconv3DBlock(embed_dim * self.input_dim, 512)

        self.decoder9_upsampler = nn.Sequential(
            Conv3DBlock(1024, 512),
            Conv3DBlock(512, 512),
            SingleDeconv3DBlock(512, 256)
        )

        self.decoder6_upsampler = nn.Sequential(
            Conv3DBlock(512, 256),
            Conv3DBlock(256, 256),
            SingleDeconv3DBlock(256, 128)
        )

        self.decoder3_upsampler = nn.Sequential(
            Conv3DBlock(256, 128),
            Conv3DBlock(128, 128),
            SingleDeconv3DBlock(128, 64),
            #nn.Upsample(size=(32,224,224), mode='trilinear', align_corners=True), #orj
            #nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1) #orj
        )
        self.decoder3_upsampler_final_conv = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1) #--EEA

        self.decoder0_header = nn.Sequential(
            Conv3DBlock(128, 64),
            Conv3DBlock(64, 64),
            SingleConv3DBlock(64, output_dim, 1)
        )
        
        # self.decoder3_header = nn.Sequential(
        #     Conv3DBlock(256, 64),
        #     Conv3DBlock(64, 64),
        #     SingleConv3DBlock(64, output_dim, 1)
        # )
        
        # self.decoder6_header = nn.Sequential(
        #     Conv3DBlock(512, 64),
        #     Conv3DBlock(64, 64),
        #     SingleConv3DBlock(64, output_dim, 1)
        # )
        
        # self.decoder9_header = nn.Sequential(
        #     Conv3DBlock(1024, 64),
        #     Conv3DBlock(64, 64),
        #     SingleConv3DBlock(64, output_dim, 1)
        # )
        
        # self.decoder12_header = nn.Sequential(
        #     Conv3DBlock(3072, 64),
        #     Conv3DBlock(64, 64),
        #     SingleConv3DBlock(64, output_dim, 1)
        # )
        
        
    def forward(self, x, features):
        z0, z3, z6, z9, z12 = x, *features
        B, D, C, H, W = x.shape
        z0 = z0.permute(0,2,1,3,4)
        z3 = reshape_transformer_output_3d(z3, self.embed_dim*C, H, W, D, self.patch_size)
        z6 = reshape_transformer_output_3d(z6, self.embed_dim*C, H, W, D, self.patch_size)
        z9 = reshape_transformer_output_3d(z9, self.embed_dim*C, H, W, D, self.patch_size)
        z12 = reshape_transformer_output_3d(z12, self.embed_dim*C, H, W, D, self.patch_size)
        
        # Decoder operations
        #seg3 = self.decoder12_header(z12)
        
        z12 = self.decoder12_upsampler(z12)  # 512
        z9 = self.decoder9(z9)  # 512p
        #seg2 = self.decoder9_header(torch.cat([z9, z12], dim=1))
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
    
        
        z6 = self.decoder6(z6)  # 256
        #seg1 = self.decoder6_header(torch.cat([z6, z9], dim=1))
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        
        z3 = self.decoder3(z3)  # 128
        #seg0 = self.decoder3_header(torch.cat([z3, z6], dim=1))
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z3 = F.interpolate(z3, size=(D, H, W), mode='trilinear', align_corners=True)
        z3 = self.decoder3_upsampler_final_conv(z3)

        z0 = self.decoder0(z0)  # 64

        logits = self.decoder0_header(torch.cat([z0, z3], dim=1))
        
        # if self.ds:
        #     output = [logits, seg0, seg1, seg2, seg3]
        # else:
        #     output = logits
            
        output = logits
        
        return output



class UNETRDecoder3D(nn.Module):
    def __init__(self, embed_dim=768, patch_size=16, input_dim=3, output_dim=3, ds=True):
        super().__init__()
        self.ds = ds
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # Decoder layers
        self.decoder0 = nn.Sequential(
            Conv3DBlock(input_dim, 32, 3),
            Conv3DBlock(32, 64, 3)
        )

        self.decoder3 = nn.Sequential(
            Deconv3DBlock(embed_dim * input_dim, 512),
            Deconv3DBlock(512, 256),
            Deconv3DBlock(256, 128)
        )

        self.decoder6 = nn.Sequential(
            Deconv3DBlock(embed_dim * input_dim, 512),
            Deconv3DBlock(512, 256)
        )

        self.decoder9 = Deconv3DBlock(embed_dim * input_dim, 512)

        self.decoder12_upsampler = SingleDeconv3DBlock(embed_dim * input_dim, 512)

        self.decoder9_upsampler = nn.Sequential(
            Conv3DBlock(1024, 512),
            Conv3DBlock(512, 512),
            SingleDeconv3DBlock(512, 256)
        )

        self.decoder6_upsampler = nn.Sequential(
            Conv3DBlock(512, 256),
            Conv3DBlock(256, 256),
            SingleDeconv3DBlock(256, 128)
        )

        self.decoder3_upsampler = nn.Sequential(
            Conv3DBlock(256, 128),
            Conv3DBlock(128, 128),
            SingleDeconv3DBlock(128, 64),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        )

        self.decoder0_header = nn.Sequential(
            Conv3DBlock(128, 64),
            Conv3DBlock(64, 64),
            SingleConv3DBlock(64, output_dim, 1)
        )

    def forward(self, x, features):
        z0, z3, z6, z9, z12 = x, *features
        B, D, C, H, W = x.shape
        z0 = z0.permute(0,2,1,3,4)
        z3 = reshape_transformer_output_3d(z3, self.embed_dim*C, H, W, D, self.patch_size)
        z6 = reshape_transformer_output_3d(z6, self.embed_dim*C, H, W, D, self.patch_size)
        z9 = reshape_transformer_output_3d(z9, self.embed_dim*C, H, W, D, self.patch_size)
        z12 = reshape_transformer_output_3d(z12, self.embed_dim*C, H, W, D, self.patch_size)
        # print(f"z0 : {z0.shape}, z3 : {z3.shape}, z6 : {z6.shape}, z9 : {z9.shape}, z12 : {z12.shape}")
        
        # Decoder operations
        z12 = self.decoder12_upsampler(z12)  # 512
        z9 = self.decoder9(z9)  # 512
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1)) 
        z6 = self.decoder6(z6)  # 256
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1)) 
        z3 = self.decoder3(z3)  # 128
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1)) 
        z0 = self.decoder0(z0)  # 64
        output = self.decoder0_header(torch.cat([z0, z3], dim=1)) 
        return output


class SAMDecoder3D(nn.Module):
    def __init__(self, embed_dim=768, patch_size=16, input_dim=3, output_dim=3, ds=True):
        super().__init__()
        self.ds = ds
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # Decoder layers
        self.decoder0 = nn.Sequential(
            Conv3DBlock(input_dim, 32, 3),
            Conv3DBlock(32, 64, 3)
        )

        self.decoder3 = nn.Sequential(
            Deconv3DBlock(embed_dim * input_dim, 512),
            Deconv3DBlock(512, 256),
            Deconv3DBlock(256, 128)
        )

        self.decoder6 = nn.Sequential(
            Deconv3DBlock(embed_dim * input_dim, 512),
            Deconv3DBlock(512, 256)
        )

        self.decoder9 = Deconv3DBlock(embed_dim * input_dim, 512)

        self.decoder12_upsampler = SingleDeconv3DBlock(embed_dim * input_dim, 512)

        self.decoder9_upsampler = nn.Sequential(
            Conv3DBlock(1024, 512),
            Conv3DBlock(512, 512),
            SingleDeconv3DBlock(512, 256)
        )

        self.decoder6_upsampler = nn.Sequential(
            Conv3DBlock(512, 256),
            Conv3DBlock(256, 256),
            SingleDeconv3DBlock(256, 128)
        )

        self.decoder3_upsampler = nn.Sequential(
            Conv3DBlock(256, 128),
            Conv3DBlock(128, 128),
            SingleDeconv3DBlock(128, 64),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        )

        self.decoder0_header = nn.Sequential(
            Conv3DBlock(128, 64),
            Conv3DBlock(64, 64),
            SingleConv3DBlock(64, output_dim, 1)
        )
        
        """
        self.decoder3_header = nn.Sequential(
            Conv3DBlock(768, 64),
            Conv3DBlock(64, 64),
            SingleConv3DBlock(64, output_dim, 1)
        )
        """
        

    def forward(self, x, features):
        z0, z3, z6, z9, z12 = x, *features
        B, D, C, H, W = x.shape    
        z0 = z0.permute(0,2,1,3,4) #B, C, D, H, W
        z3 = z3.permute(0,4,1,2,3) #B, 768, D, h, w
        z6 = z6.permute(0,4,1,2,3) #B, 768, D, h, w
        z9 = z9.permute(0,4,1,2,3) #B, 768, D, h, w
        z12 = z12.permute(0,4,1,2,3) #B, 768, D, h, w

        # Decoder operations
        z12 = self.decoder12_upsampler(z12)  # 512

        z9 = self.decoder9(z9)  # 512

        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))

        z6 = self.decoder6(z6)  # 256

        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1)) 

        z3 = self.decoder3(z3)  # 128

        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))

        z0 = self.decoder0(z0)  # 64

        logits = self.decoder0_header(torch.cat([z0, z3], dim=1))

    
        return logits
    


# === 3D 통합 모델 === #
class DummyEncoder(nn.Module):
    def __init__(self, batch_size=1, num_tokens=257, embed_dim=768):
        super(DummyEncoder, self).__init__()
        self.batch_size = batch_size
        self.num_tokens = num_tokens  
        self.embed_dim = embed_dim  #768
        self.d = 4
    def forward(self, x):
        """
        입력 x는 Decoder에서 첫 번째 입력(z0)으로 사용됩니다.
        """
        # 더미 feature map 생성
        z3 = torch.randn(self.batch_size,self.d, self.num_tokens, self.embed_dim)
        z6 = torch.randn(self.batch_size,self.d, self.num_tokens, self.embed_dim)
        z9 = torch.randn(self.batch_size,self.d, self.num_tokens, self.embed_dim)
        z12 = torch.randn(self.batch_size, self.d,self.num_tokens, self.embed_dim)
        
        return [z3, z6, z9, z12]
    
class UNETR(nn.Module):
    def __init__(self, encoder, embed_dim=768, patch_size = 14, input_dim=3, output_dim=3):
        super().__init__()
        self.encoder = encoder
        self.decoder = UNETRDecoder3D(embed_dim,patch_size, input_dim, output_dim)

    def forward(self, x):
        features = self.encoder(x)
        print(f'encoder_features : {len(features)}, {features[0].shape}')
        x = x.permute(0,2,3,4,1) 
        output = self.decoder(x, features)
        return output
   
#Text code
if __name__ == '__main__':
    model = UNETR(encoder=DummyEncoder())
    x = torch.randn(1, 4, 3, 224, 224) # B, D, C, H, W
    print(f"inpt shape : {x.shape}")
    features = model(x)
    print(f"output shape : {features.shape}")

