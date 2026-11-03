import torch
import torch.nn as nn

class LDF(nn.Module):
    """
    Local Depth Fusion
    """
    def __init__(self, embed_dim=768, rank=16):
        super(LDF, self).__init__()
        self.v_t = nn.Linear(3 * embed_dim, rank, bias=False)
        self.u = nn.Linear(rank, embed_dim, bias=False)
        
    def forward(self, x):
        """
        Input: x(B, D, N, C) -> Batch, Depth, Num_Patches, Channels
        Output: h(B, D, N, C) -> Refined features
        """
        B, D, N, C = x.shape

        x_pad = torch.cat([x[:, 0:1, ...], x, x[:, -1:, ...]], dim=1)
        
        f_prev = x_pad[:, 0:D, :, :]
        f_curr = x_pad[:, 1:D+1, :, :]
        f_next = x_pad[:, 2:D+2, :, :]
        
        g_i = torch.cat([f_prev, f_curr, f_next], dim=-1)
        h_i = self.u(self.v_t(g_i))
        return x + h_i

    