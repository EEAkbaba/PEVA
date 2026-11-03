import torch.nn as nn
import torch
import os
import torch.nn.functional as F

from nnunetv2.custom_model.encoder import Dinov2Encoder
from nnunetv2.custom_model.decoder_3D import UNETRDecoder3D_14, UNETRDecoder3D
from nnunetv2.custom_model.dinov3_encoder import Dinov3Encoder, Dinov3Encoder_multilayer
from nnunetv2.custom_model.decoder_2D import Decoder2D_16, Decoder2D_16_multilayer
from nnunetv2.custom_module.ldf import LDF
from nnunetv2.custom_module.cross_view_fusion import CrossViewFusion
from nnunetv2.custom_model.MEDDINOv3_encoder import MedDINOv3Encoder


class deepSupervisionDummy():
    def __init__(self):
        self.deep_supervision = False
    
class DINOv3_multilayer(nn.Module):
    def __init__(self, embed_dim=768, patch_size=16, input_dim=3, output_dim=3, encoder_type="dinov3"):
        super(DINOv3_multilayer, self).__init__()
        self.decoder = deepSupervisionDummy()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.mode = os.environ.get("MODE", "2D")   # "2D" or "2_5D"
        print(f"------- MODE ---------: {self.mode}")
        use_ldf_env = os.environ.get("USE_LDF", "True")   # "True" or "False"
        print(f"------- USE_LDF ---------: {use_ldf_env}")
        use_cross_view_fusion_env = os.environ.get("USE_CROSS_VIEW_FUSION", "True")
        print(f"------- USE_CROSS_VIEW_FUSION ---------: {use_cross_view_fusion_env}")
        full_ft_env = os.environ.get("FULL_FT", "False")
        print(f"------- FULL_FT ---------: {full_ft_env}")

        if self.mode not in ["2D", "2_5D"]:
            raise ValueError(f"Unknown MODE: {self.mode}. Expected '2D' or '2_5D'")
        if use_ldf_env not in ["True", "False"]:
            raise ValueError(f"Unknown USE_LDF: {use_ldf_env}. Expected 'True' or 'False'")
        if use_cross_view_fusion_env not in ["True", "False"]:
            raise ValueError(f"Unknown USE_CROSS_VIEW_FUSION: {use_cross_view_fusion_env}. Expected 'True' or 'False'")
        if full_ft_env not in ["True", "False"]:
            raise ValueError(f"Unknown FULL_FT: {full_ft_env}. Expected 'True' or 'False'")
        
        self.use_ldf = (use_ldf_env == "True")
        self.use_cross_view_fusion = (use_cross_view_fusion_env == "True")
        self.full_ft = (full_ft_env == "True")

        self.encoder_type = encoder_type
        print(f"------- ENCODER TYPE ---------: {self.encoder_type}")
        if self.encoder_type == "dinov3":
            self.encoder = Dinov3Encoder_multilayer(freeze=not self.full_ft)
        elif self.encoder_type == "meddinov3":
            self.encoder = MedDINOv3Encoder(freeze=not self.full_ft)
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}. Expected 'dinov3' or 'meddinov3'.")

        self.decoder2d = Decoder2D_16_multilayer(embed_dim=self.embed_dim, patch_size=self.patch_size, output_dim=self.output_dim, input_dim=self.input_dim)
        
        if self.use_cross_view_fusion:
            self.cross_view_fusion = CrossViewFusion(dim=self.embed_dim, bottleneck=192)

        if self.use_ldf:
            self.ldf_adapters_9 = nn.ModuleList([LDF(embed_dim=self.embed_dim, rank=16) for _ in range(self.input_dim)]) 
            self.ldf_adapters_12 = nn.ModuleList([LDF(embed_dim=self.embed_dim, rank=16) for _ in range(self.input_dim)]) 

            if self.use_cross_view_fusion:
                self.ldf_adapters_12_sag = nn.ModuleList([LDF(embed_dim=self.embed_dim, rank=16) for _ in range(self.input_dim)])
                self.ldf_adapters_12_cor = nn.ModuleList([LDF(embed_dim=self.embed_dim, rank=16) for _ in range(self.input_dim)])

        for param in self.encoder.parameters():
            param.requires_grad = self.full_ft

    def forward(self, x):
        # input: (B, C, D, H, W)
        B, C, D, H, W = x.shape

        need_cross_views = self.use_cross_view_fusion
        if need_cross_views:
            x_raw = x
        
        x = x.permute(0, 2, 1, 3, 4).contiguous() 

        if need_cross_views:
            sag = x_raw.permute(0, 4, 1, 2, 3).contiguous()
            cor = x_raw.permute(0, 3, 1, 2, 4).contiguous()

        if self.mode == "2D":
            x_2d = x.view(B * D, C, H, W).contiguous()
            if need_cross_views:
                sag_2d = sag.contiguous().view(B * W, C, D, H)
                cor_2d = cor.contiguous().view(B * H, C, D, W)
        elif self.mode == "2_5D":
            x_ext = torch.cat([x[:, 0:1, ...], x, x[:, -1:, ...]], dim=1)
            prev   = x_ext[:, 0:D,   ...]
            center = x_ext[:, 1:D+1, ...]
            nxt    = x_ext[:, 2:D+2, ...]
            x_2d = torch.cat([prev, center, nxt], dim=2).contiguous().view(B * D, 3 * C, H, W)
            if need_cross_views:
                sag_ext = torch.cat([sag[:, 0:1, ...], sag, sag[:, -1:, ...]], dim=1)
                prev_sag   = sag_ext[:, 0:W,   ...]   
                center_sag = sag_ext[:, 1:W+1, ...]  
                nxt_sag    = sag_ext[:, 2:W+2, ...]
                sag_2d = torch.cat([prev_sag, center_sag, nxt_sag], dim=2).contiguous().view(B * W, 3 * C, D, H)
                cor_ext = torch.cat([cor[:, 0:1, ...], cor, cor[:, -1:, ...]], dim=1)
                prev_cor   = cor_ext[:, 0:H,   ...]   
                center_cor = cor_ext[:, 1:H+1, ...]  
                nxt_cor    = cor_ext[:, 2:H+2, ...]
                cor_2d = torch.cat([prev_cor, center_cor, nxt_cor], dim=2).contiguous().view(B * H, 3 * C, D, W)

        # modality inputs
        if C == 1:
            if self.mode == "2D":
                t = x_2d[:, 0:1, ...].repeat(1, 3, 1, 1)
                if need_cross_views:
                    t_sag = sag_2d[:, 0:1, ...].repeat(1, 3, 1, 1)
                    t_cor = cor_2d[:, 0:1, ...].repeat(1, 3, 1, 1)
            elif self.mode == "2_5D":
                t = x_2d
                if need_cross_views:
                    t_sag = sag_2d
                    t_cor = cor_2d

            modalities = [t]
            if need_cross_views:
                modalities_sag = [t_sag]
                modalities_cor = [t_cor]

        elif C == 4:
            if self.mode == "2D":
                t1    = x_2d[:, 0:1, ...].repeat(1, 3, 1, 1)
                t2    = x_2d[:, 1:2, ...].repeat(1, 3, 1, 1)
                t1ce  = x_2d[:, 2:3, ...].repeat(1, 3, 1, 1)
                flair = x_2d[:, 3:4, ...].repeat(1, 3, 1, 1)

                if need_cross_views:
                    t1_sag    = sag_2d[:, 0:1, ...].repeat(1, 3, 1, 1)
                    t2_sag    = sag_2d[:, 1:2, ...].repeat(1, 3, 1, 1)
                    t1ce_sag  = sag_2d[:, 2:3, ...].repeat(1, 3, 1, 1)
                    flair_sag = sag_2d[:, 3:4, ...].repeat(1, 3, 1, 1)
                    t1_cor    = cor_2d[:, 0:1, ...].repeat(1, 3, 1, 1)
                    t2_cor    = cor_2d[:, 1:2, ...].repeat(1, 3, 1, 1)
                    t1ce_cor  = cor_2d[:, 2:3, ...].repeat(1, 3, 1, 1)
                    flair_cor = cor_2d[:, 3:4, ...].repeat(1, 3, 1, 1)

            elif self.mode == "2_5D":
                t1    = x_2d[:, 0:3,  ...]
                t2    = x_2d[:, 3:6,  ...]
                t1ce  = x_2d[:, 6:9,  ...]
                flair = x_2d[:, 9:12, ...]

                if need_cross_views:
                    t1_sag    = sag_2d[:, 0:3,  ...]
                    t2_sag    = sag_2d[:, 3:6,  ...]
                    t1ce_sag  = sag_2d[:, 6:9,  ...]
                    flair_sag = sag_2d[:, 9:12, ...]
                    t1_cor    = cor_2d[:, 0:3,  ...]
                    t2_cor    = cor_2d[:, 3:6,  ...]
                    t1ce_cor  = cor_2d[:, 6:9,  ...]
                    flair_cor = cor_2d[:, 9:12, ...]

            modalities = [t1, t2, t1ce, flair]
            if need_cross_views:
                modalities_sag = [t1_sag, t2_sag, t1ce_sag, flair_sag]
                modalities_cor = [t1_cor, t2_cor, t1ce_cor, flair_cor]

        else:
            if self.mode == "2D":
                modalities = [x_2d[:, i:i+1, ...].repeat(1, 3, 1, 1) for i in range(C)]
                if need_cross_views:
                    modalities_sag = [sag_2d[:, i:i+1, ...].repeat(1, 3, 1, 1) for i in range(C)]
                    modalities_cor = [cor_2d[:, i:i+1, ...].repeat(1, 3, 1, 1) for i in range(C)]
            elif self.mode == "2_5D":
                modalities = [x_2d[:, 3*i:3*(i+1), ...] for i in range(C)]
                if need_cross_views:
                    modalities_sag = [sag_2d[:, 3*i:3*(i+1), ...] for i in range(C)]
                    modalities_cor = [cor_2d[:, 3*i:3*(i+1), ...] for i in range(C)]

        slot = []
        if self.use_cross_view_fusion:
            proj_outputs = []  # Her modalitenin projected volume tuple'ını tutacak

        for i in range(C):
            one_modality = modalities[i]   # (B*D, 3, H, W)
            features = self.encoder(one_modality)   # [f3, f6, f9, f12]

            reshaped_features = [f.view(B, D, f.shape[1], f.shape[2]).contiguous() for f in features]   # [z3, z6, z9, z12], each (B, D, N, dim)

            if self.use_ldf:
                reshaped_features[2] = self.ldf_adapters_9[i](reshaped_features[2])   # z9
                reshaped_features[3] = self.ldf_adapters_12[i](reshaped_features[3])  # z12

            if need_cross_views:
                with torch.no_grad():
                    feat_sag_last = self.encoder(modalities_sag[i])[-1]   # [B*W, N_sag, C]
                    feat_cor_last = self.encoder(modalities_cor[i])[-1]   # [B*H, N_cor, C]

                feat_sag_last = feat_sag_last.view(B, W, feat_sag_last.shape[1], feat_sag_last.shape[2]).contiguous()
                feat_cor_last = feat_cor_last.view(B, H, feat_cor_last.shape[1], feat_cor_last.shape[2]).contiguous()

                if self.use_ldf:
                    feat_sag_last = self.ldf_adapters_12_sag[i](feat_sag_last)
                    feat_cor_last = self.ldf_adapters_12_cor[i](feat_cor_last)
            if self.use_cross_view_fusion:
                fusion_dict = self.cross_view_fusion(
                    reshaped_features[3], # axial z12: [B,32,196,C]
                    feat_sag_last,        # sagittal z12: [B,224,28,C]
                    feat_cor_last         # coronal z12: [B,224,28,C]
                )
                reshaped_features[3] = fusion_dict["fused"]
                proj_outputs.append((
                    fusion_dict["axial_vol"],
                    fusion_dict["sag_proj_vol"],
                    fusion_dict["cor_proj_vol"]
                ))

            slot.append(reshaped_features)

        if len(slot) > 1:
            encoder_features = [torch.cat([slot[m][i] for m in range(len(slot))], dim=3) for i in range(len(slot[0]))]
        else:
            encoder_features = slot[0]

        output = self.decoder2d(x, encoder_features)

        if self.training:
            if self.use_cross_view_fusion:
                return output, None, proj_outputs
        return output


class MedDINOv3_PEVA_multilayer(DINOv3_multilayer):
    def __init__(self, embed_dim=768, patch_size=16, input_dim=3, output_dim=3):
        super(MedDINOv3_PEVA_multilayer, self).__init__(
            embed_dim=embed_dim,
            patch_size=patch_size,
            input_dim=input_dim,
            output_dim=output_dim,
            encoder_type="meddinov3"
        )
    
class DINOUNETR(nn.Module):
    def __init__(self, embed_dim = 768, patch_size = 14, input_dim = 3, output_dim=3):
        super(DINOUNETR, self).__init__()
        self.decoder = deepSupervisionDummy()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.encoder = Dinov2Encoder(freeze=True)
        self.decoder1 = UNETRDecoder3D_14(embed_dim=self.embed_dim, patch_size=self.patch_size, input_dim=self.input_dim, output_dim=self.output_dim, ds = self.decoder.deep_supervision)
        
        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.permute(0,2,1,3,4).contiguous() # (B, C, D, H, W) -> (B, D, C, H, W)
        x_2d= x.view(B * D, C, H, W).contiguous()

        if C == 1: 
            # CT / Synapse case (single channel)
            # replicate to 3 channels for DINO (RGB expected)
            t = x_2d[:, 0:1, ...].repeat(1, 3, 1, 1)
            modalities = [t]

        elif C == 4:
            t1    = x_2d[:, 0:1, ...].repeat(1, 3, 1, 1)
            t2    = x_2d[:, 1:2, ...].repeat(1, 3, 1, 1)
            t1ce  = x_2d[:, 2:3, ...].repeat(1, 3, 1, 1)
            flair = x_2d[:, 3:4, ...].repeat(1, 3, 1, 1)
            modalities = [t1, t2, t1ce, flair]

        else:
            modalities = [x_2d[:, i:i+1, ...].repeat(1, 3, 1, 1) for i in range(C)]

        slot = []
        for i in range(C):
            one_modality = modalities[i] # (B*D, 3, H, W)
            features = self.encoder(one_modality) # (B*D, N, dim)
            # feature : 32, 256, 768
            reshaped_features = [f.view(B, D, f.shape[1], f.shape[2]).contiguous() for f in features]
            # feature : 2, 16, 256, 768
            slot.append(reshaped_features)
        if len(slot) > 1: 
            encoder_features = [torch.cat([slot[m][i] for m in range(len(slot))], dim=3)for i in range(len(slot[0]))]
        else:
            encoder_features = slot[0]
            
        output = self.decoder1(x.view(B, D, C, H, W).contiguous(), encoder_features)
        return output
    
class DINOUNETR_ENC(nn.Module):
    def __init__(self, embed_dim = 768, patch_size = 14, input_dim = 3, output_dim=3):
        super(DINOUNETR_ENC, self).__init__()
        
        self.decoder = deepSupervisionDummy()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.encoder = Dinov2Encoder(freeze=False)
        self.decoder1 = UNETRDecoder3D_14(embed_dim=self.embed_dim, patch_size=self.patch_size, input_dim=self.input_dim, output_dim=self.output_dim, ds = self.decoder.deep_supervision)
            
    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.permute(0,2,1,3,4).contiguous()
        x= x.view(B * D, C, H, W).contiguous()
        if C == 1:
            t = x[:, 0:1, ...].repeat(1, 3, 1, 1)
            modalities = [t]

        elif C == 4:
            t1    = x[:, 0:1, ...].repeat(1, 3, 1, 1)
            t2    = x[:, 1:2, ...].repeat(1, 3, 1, 1)
            t1ce  = x[:, 2:3, ...].repeat(1, 3, 1, 1)
            flair = x[:, 3:4, ...].repeat(1, 3, 1, 1)
            modalities = [t1, t2, t1ce, flair]

        else:
            modalities = [
                x[:, i:i+1, ...].repeat(1, 3, 1, 1) for i in range(C)
            ]

        slot = []
        for i in range(C):
            one_modality = modalities[i] # (B*D, 3, H, W)
            features = self.encoder(one_modality)
            # feature : 32, 256, 768
            reshaped_features = [
                f.view(B, D, f.shape[1], f.shape[2]).contiguous() for f in features
            ]

            slot.append(reshaped_features)
        if len(slot) > 1:
            encoder_features = [torch.cat([slot[m][i] for m in range(len(slot))], dim=3)for i in range(len(slot[0]))]
        else:
            encoder_features = slot[0]
            
        output = self.decoder1(x.view(B, D, C, H, W).contiguous(), encoder_features)
        return output

if __name__ == "__main__":
    import torch
    print("Creating model...")
    model = MedDINOv3_PEVA_multilayer(
        embed_dim=768,
        patch_size=16,
        input_dim=1,
        output_dim=14
    )
    print("Model created successfully")
    x = torch.randn(1, 1, 32, 224, 224)
    print("Running forward pass...")
    with torch.no_grad():
        y = model(x)
    if isinstance(y, tuple):
        print("Output is tuple")
        print(type(y[0]), y[0].shape)
    else:
        print("Output shape:", y.shape)
    print("Sanity check passed")
