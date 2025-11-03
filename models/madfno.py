# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

class MADFNO_Filter(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    """

    def __init__(
        self,
        hidden_size,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=5,
        hidden_size_factor=1,
    ):
        super().__init__()
        assert hidden_size % num_blocks == 0, (
            f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"
        )

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale
            * torch.randn(
                2,
                self.num_blocks,
                self.block_size,
                self.block_size * self.hidden_size_factor,
            )
        )

        self.b1 = nn.Parameter(
            self.scale
            * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor)
        )

        self.w2 = nn.Parameter(
            self.scale
            * torch.randn(
                2,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
                self.block_size,
            )
        )

        self.b2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size)
        )

    def forward(self, x):
        bias = x  # [:,:,0,:]
        dtype = x.dtype
        x = x.float()
        B, N, N_P, C = x.shape
        x = torch.fft.rfft(x, dim=2, norm="ortho")
        N_reduced = N * (N_P // 2 + 1)
        x = x.reshape(B, N_reduced, self.num_blocks, self.block_size)
        o1_real = torch.zeros(
            [B, N_reduced, self.num_blocks, self.block_size * self.hidden_size_factor],
            device=x.device,
        )
        o1_imag = torch.zeros(
            [B, N_reduced, self.num_blocks, self.block_size * self.hidden_size_factor],
            device=x.device,
        )
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = N_P // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)


        o1_real[:, :kept_modes] = F.relu(
            torch.einsum("...bi,bio->...bo", x[:, :kept_modes].real, self.w1[0])
            - torch.einsum("...bi,bio->...bo", x[:, :kept_modes].imag, self.w1[1])
            + self.b1[0]
        )

        o1_imag[:, :kept_modes] = F.relu(
            torch.einsum("...bi,bio->...bo", x[:, :kept_modes].imag, self.w1[0])
            + torch.einsum("...bi,bio->...bo", x[:, :kept_modes].real, self.w1[1])
            + self.b1[1]
        )

        o2_real[:, :kept_modes] = (
            torch.einsum("...bi,bio->...bo", o1_real[:, :kept_modes], self.w2[0])
            - torch.einsum("...bi,bio->...bo", o1_imag[:, :kept_modes], self.w2[1])
            + self.b2[0]
        )

        o2_imag[:, :kept_modes] = (
            torch.einsum("...bi,bio->...bo", o1_imag[:, :kept_modes], self.w2[0])
            + torch.einsum("...bi,bio->...bo", o1_real[:, :kept_modes], self.w2[1])
            + self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, N, N_P // 2 + 1, C)

        x = torch.fft.irfft(x, n=N_P, dim=2, norm="ortho")
        x = x.type(dtype)
        x = x.view(B, N, N_P, C)
        return x + bias

class MADFNO(nn.Module):
    def __init__(self, d_model, n_slices, n_levels, n_points):
        super().__init__()

        n_slices = n_slices
        self.d_model = d_model
        self.filter = MADFNO_Filter(d_model, num_blocks=8)
        self.sapling_loc_layer = nn.Linear(d_model, n_levels * n_points * n_slices * 2)
        self.n_levels = n_levels
        self.n_points = n_points
        self.n_slices = n_slices

    # Vectorized version for better efficiency
    def sample_from_feature_map_vectorized(self, sampling_locations, content):
        """
        Vectorized implementation to sample content based on provided sampling locations.
        Minimizes loops for better efficiency on GPU.
        Args:
            sampling_locations: Tensor of shape (bs, n_slices, n_queries, n_levels, n_points, 2)
                            Contains normalized coordinates in range [0, 1]
            content: Tensor of shape (bs, n_slices, n_levels, H, W, hidden_dim)
                    Contains content features to sample from
        Returns:
            output tensor of shape: bs, n_queries, n_levels, n_points, hidden_dim * n_slices
        """
        bs, n_slices, n_queries, n_levels, n_points, _ = sampling_locations.shape
        _, _, _, H, W, hidden_dim = content.shape

        # Normalize from [0,1] to [-1,1] as required by grid_sample
        sampling_locations = sampling_locations.clone() * 2.0 - 1.0

        # Initialize output tensor
        output = torch.zeros(
            bs,
            n_queries,
            n_levels,
            n_points,
            hidden_dim * n_slices,
            device=sampling_locations.device,
            dtype=content.dtype,
        )

        # Process each level (still need this loop as grid_sample works on 4D tensors)
        for l in range(n_levels):
            # Reshape content for this level: (bs*n_slices, hidden_dim, H, W)
            curr_content = content[:, :, l].reshape(bs * n_slices, H, W, hidden_dim)
            curr_content = curr_content.permute(
                0, 3, 1, 2
            )  # ! [bs * n_slices, hidden_dim, H, W]

            # Reshape sampling locations for this level: # ! (bs*n_slices, n_queries, n_points, 2)
            curr_locs = sampling_locations[:, :, :, l].reshape(bs * n_slices, n_queries, n_points, 2)
            sampled = F.grid_sample(
                curr_content,
                curr_locs,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            # ! (bs*n_slices, hidden_dim, n_queries, n_points)

            # Reshape to proper output dimensions
            sampled = (
                sampled.reshape(bs, n_slices, hidden_dim, n_queries, n_points)
                .flatten(1, 2)
                .permute(0, 2, 3, 1)
            )  # ! ( bs, n_queries, n_points, num_heads * hidden_dim )

            output[:, :, l] = sampled
        return output

    def forward(self, query, key, ref_points, level_start_index, spatial_shapes):
        """
        Args:
            query: Tensor of shape (bs, n_queries, hidden_dim)
            key: Tensor of shape (bs, n_levels, H, W, hidden_dim)
            ref_points: (bs, n_queries, n_levels, 4)
            level_start_index: (n_levels,)
            spatial_shapes: (n_levels, 2)
        Returns:
           #! output tensor of shape: bs, n_queries, n_levels, n_points, hidden_dim * n_slices
        """
        bs, n_q, hdim = query.shape
        attn_keys = key #!bs, nL, H, W, hdim 
        sampled_locs = self.sapling_loc_layer(query).reshape(
            bs, n_q, self.n_slices, self.n_levels, self.n_points, 2
        )
        sampled_locs = (
            ref_points[:, :, None, :, None, :2]
            + sampled_locs / self.n_points * ref_points[:, :, None, :, None, 2:]
        )
        sampled_locs = sampled_locs.clamp_(0, 1).transpose(
            1, 2
        )  # bs, n_slices, n_q, n_levels, n_points, 2

        batch_size, n_levels, height, width, d_model = attn_keys.shape
        attn_keys = attn_keys.reshape(
            batch_size, n_levels, height, width, self.n_slices, d_model // self.n_slices
        ).permute(0, 4, 1, 2, 3, 5)
        attn_keys = self.sample_from_feature_map_vectorized(content=attn_keys, sampling_locations=sampled_locs).flatten(2, 3)  

        query = query.unsqueeze(2)
        inp = torch.cat([query, attn_keys], dim=2)
        out = self.filter(inp)
        out = out.mean(dim=2)
        return out.transpose(0, 1)
