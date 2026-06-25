import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveMerge(nn.Module):
    # This replaces the stride-2 1x1 conv at the finest feature level.
    # That conv keeps only 1 cell out of every 2x2 block and throws away the other 3.
    # Instead we take all 4 cells, add one learnable "blank" token, let them all
    # attend to each other, and use the blank token's output as the merged cell.
    # Shape stays the same as the conv:  (B, C_in, H, W) -> (B, C_out, H/2, W/2)

    def __init__(self, in_channels, out_channels, nheads=8):
        super().__init__()

        self.out_channels = out_channels

        # project every cell from C_in to C_out
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # the learnable blank token (the aggregator)
        self.blank = nn.Parameter(torch.zeros(1, 1, out_channels))

        # attention over the 4 cells + the blank token
        self.attn = nn.MultiheadAttention(out_channels, nheads, batch_first=True)

        self.norm = nn.LayerNorm(out_channels)

        nn.init.trunc_normal_(self.blank, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        H = x.shape[2]
        W = x.shape[3]
        C = self.out_channels

        # project each cell
        x = self.proj(x)

        # if H or W is odd, pad by 1 so the 2x2 blocks come out clean
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, W % 2, 0, H % 2))
            H = x.shape[2]
            W = x.shape[3]

        nh = H // 2
        nw = W // 2

        # break the map into 2x2 blocks, 4 cells in each block
        x = x.view(B, C, nh, 2, nw, 2)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.contiguous()
        cells = x.view(B * nh * nw, 4, C)

        # one blank token for each block
        blank = self.blank.expand(cells.shape[0], -1, -1)

        # put the blank token in front of the 4 cells -> 5 tokens
        tokens = torch.cat([blank, cells], dim=1)

        # let all 5 tokens attend to each other
        attended, _ = self.attn(tokens, tokens, tokens)

        # the blank token's output is our merged cell
        merged = attended[:, 0]

        # add the mean of the 4 cells (a residual) and then normalise
        merged = merged + cells.mean(dim=1)
        merged = self.norm(merged)

        # put the merged cells back into a feature map
        merged = merged.view(B, nh, nw, C)
        merged = merged.permute(0, 3, 1, 2)
        merged = merged.contiguous()

        return merged
