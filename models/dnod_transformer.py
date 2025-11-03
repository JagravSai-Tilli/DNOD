import torch.nn as nn
import copy
import torch
import torch.nn.functional as F

from .model_utils import (
    _get_activation_fn,
    MLP,
    gen_encoder_output_proposals,
    gen_sineembed_for_position,
)
from utils.misc import inverse_sigmoid
from .madfno import MADFNO 


class MADFNODecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        d_model=256,
        query_dim=4,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers, layer_share=False)

        self.norm = norm
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)

        self.bbox_embed = None
        self.class_embed = None

        self.d_model = d_model
        self.rm_detach = None

    def forward(
        self,
        tgt,
        memory,
        refpoints_unsigmoid, 
        level_start_index,  
        spatial_shapes, 
        valid_ratios,
    ):
        output = tgt
        intermediate = []  
        intermediate_o2m = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]  

        for layer_id, layer in enumerate(self.layers):

            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
                )  
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = (
                    reference_points[:, :, None] * valid_ratios[None, :]
                )

            # A sinusoidal positional encoding tensor
            query_sine_embed = gen_sineembed_for_position(
                reference_points_input[:, :, 0, :]
            )  

            # conditional query
            raw_query_pos = self.ref_point_head(
                query_sine_embed
            )  

            query_pos = raw_query_pos

            output, output_o2m = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_reference_points=reference_points_input,
                memory=memory,
                memory_spatial_shapes=spatial_shapes,
                memory_level_start_index = level_start_index
            )

            # ref point update
            if self.bbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                reference_points = new_reference_points.detach()


                ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))
            
            if output_o2m is not None:
                intermediate_o2m.append(self.norm(output_o2m))
        
        if len(intermediate_o2m)>0:
            return [
                [itm_out.transpose(0, 1) for itm_out in intermediate],
                [itm_out_o2m.transpose(0, 1) for itm_out_o2m in intermediate_o2m],
                [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
            ]
        else:
            return [
                [itm_out.transpose(0, 1) for itm_out in intermediate],
                None,
                [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
            ]
        


class MADFNODecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_slices=8,
        n_points=4,
        key_aware_type=None,
        module_seq=["sa", "ca", "ffn"],
        use_aux_ffn=None,
    ):
        super().__init__()
        self.module_seq = module_seq
        self.use_aux_ffn = use_aux_ffn

        self.mad_fno = MADFNO(
            d_model=d_model, n_slices=n_slices, n_levels=n_levels, n_points=n_points
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.afno_1d = AFNO1D(
            hidden_size=d_model,
            num_blocks=8,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1,
            hidden_size_factor=1,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        if self.use_aux_ffn:
            self.linear3 = nn.Linear(d_model, d_ffn)
            self.dropout5 = nn.Dropout(dropout)
            self.linear4 = nn.Linear(d_ffn, d_model)
            self.dropout6 = nn.Dropout(dropout)
            self.norm4 = nn.LayerNorm(d_model)

        self.key_aware_type = key_aware_type
        self.key_aware_proj = None


    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_ca(
        self,
        tgt, 
        tgt_query_pos, 
        tgt_reference_points,  
        memory, 
        memory_level_start_index,  
        memory_spatial_shapes, 
    ):
        if self.key_aware_type is not None:
            if self.key_aware_type == "mean":
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == "proj_mean":
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError(
                    "Unknown key_aware_type: {}".format(self.key_aware_type)
                )

        tgt2 = self.mad_fno(
            query=self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
            key=memory.transpose(0, 1),
            ref_points=tgt_reference_points.transpose(0, 1).contiguous(),
            level_start_index=memory_level_start_index,
            spatial_shapes=memory_spatial_shapes,
        )

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if self.use_aux_ffn:
            tgt_o2m = self.forward_aux_ffn(tgt)
        else:
            tgt_o2m = None
            
            
        return tgt, tgt_o2m

    def forward_sa(self,tgt,tgt_query_pos,):
        q = k = self.with_pos_embed(tgt, tgt_query_pos)
        tgt2 = self.afno_1d(q.permute(1, 0, 2)).permute(1, 0, 2)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

    def forward_aux_ffn(self, tgt):
        tgt2 = self.linear4(self.dropout5(self.activation(self.linear3(tgt))))
        tgt = tgt + self.dropout6(tgt2)
        tgt = self.norm4(tgt)
        return tgt
    
    def forward(
        self,
        tgt, 
        tgt_query_pos, 
        tgt_reference_points,  
        memory, 
        memory_spatial_shapes,
        memory_level_start_index  
    ):

        for funcname in self.module_seq:
            if funcname == "ffn":
                tgt = self.forward_ffn(tgt)
            elif funcname == "ca":
                tgt, tgt_o2m = self.forward_ca(
                    tgt,
                    tgt_query_pos,
                    tgt_reference_points,
                    memory,
                    memory_level_start_index=memory_level_start_index,
                    memory_spatial_shapes = memory_spatial_shapes,
                )
            elif funcname == "sa":
                tgt = self.forward_sa(tgt,tgt_query_pos,)
            else:
                raise ValueError("unknown funcname {}".format(funcname))
        return tgt, tgt_o2m


class DNOD_transformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.num_feature_levels = args.num_feature_levels
        self.num_queries = args.num_queries
        self.d_model = args.hidden_dim
        self.nhead = args.nheads

        msfm_layer = MSFMLayer(
            d_model=self.d_model,
            d_ffn=args.dim_feedforward // 2,
            dropout=args.dropout,
            activation=args.transformer_activation,
            n_blocks=1,
        )

        encoder_norm = nn.LayerNorm(self.d_model) if args.pre_norm else None

        self.encoder = MSFM(
            layer=msfm_layer,
            num_layers=args.enc_layers,
            norm=encoder_norm,
        )


        decoder_layer = MADFNODecoderLayer(
            d_model=self.d_model,
            d_ffn=args.dim_feedforward,
            dropout=args.dropout,
            activation=args.transformer_activation,
            n_levels=self.num_feature_levels,
            n_slices=self.nhead,
            n_points=args.dec_n_points,
            key_aware_type=None,
            module_seq=args.decoder_module_seq,
            use_aux_ffn = args.use_aux_ffn
        )

        decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = MADFNODecoder(
            decoder_layer,
            args.dec_layers,
            decoder_norm,
            d_model=self.d_model,
            query_dim=args.query_dim,
        )

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.d_model)
        )

        self.tgt_embed = nn.Embedding(self.num_queries, self.d_model)
        nn.init.normal_(self.tgt_embed.weight.data)

        self.enc_output = nn.Linear(self.d_model, self.d_model)
        self.enc_output_norm = nn.LayerNorm(self.d_model)

        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None
        self._reset_parameters()
        self.args = args

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds):
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        src_inp = torch.stack(srcs, dim=1).permute(0, 1, 3, 4, 2)


        if self.num_feature_levels > 1:
            level_embed_reshaped = self.level_embed.view(
                self.num_feature_levels, 1, 1, -1
            )
            for lvl, pos_embed in enumerate(pos_embeds):
                bs, h, w, c = pos_embed.shape

                spatial_shapes.append((h, w))

                lvl_pos_embed = pos_embed + level_embed_reshaped[lvl]

                lvl_pos_embed_flatten.append(lvl_pos_embed)

        lvl_pos_embed_flatten = torch.stack(
            lvl_pos_embed_flatten, 1
        ) 
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_inp.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        memory = self.encoder(
            src_inp,
            pos=lvl_pos_embed_flatten,
        )

        mask_flatten = torch.stack(masks, 1)
        mask_flatten = mask_flatten.flatten(
            start_dim=1,
        )


        output_memory, output_proposals = gen_encoder_output_proposals(
            memory.flatten(start_dim=1, end_dim=3), mask_flatten, spatial_shapes
        )

        output_memory = self.enc_output_norm(
            self.enc_output(output_memory)
        ) 

        enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
        enc_outputs_coord_unselected = (
            self.enc_out_bbox_embed(output_memory) + output_proposals
        )  

        topk = self.num_queries
        topk_proposals = torch.topk(
            enc_outputs_class_unselected.max(-1)[0], topk, dim=1
        )[1] 

        # gather boxes
        refpoint_embed_undetach = torch.gather(
            enc_outputs_coord_unselected,
            1,
            topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
        )  # unsigmoid
        refpoint_embed_ = refpoint_embed_undetach.detach()
        init_box_proposal = torch.gather(
            output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        ).sigmoid()  # sigmoid

        # gather tgt
        tgt_undetach = torch.gather(
            output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model)
        )

        tgt_ = (
            self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
        ) 

        refpoint_embed, tgt = refpoint_embed_, tgt_


        # Decoder
        hs, hs_o2m, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=memory.transpose(0, 1),
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
        )
        # Getting the encoder queries ready to output

        hs_enc = tgt_undetach.unsqueeze(0)
        ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)

        return hs, hs_o2m, references, hs_enc, ref_enc, init_box_proposal


def _get_clones(module, N, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MSFM(nn.Module):
    def __init__(
        self,
        layer,
        num_layers,
        norm=None,
    ):
        super().__init__()
        self.layers = _get_clones(layer, num_layers, layer_share=False)
        self.norm = norm

    def forward(self, src, pos):
        output = src

        for layer_id, layer in enumerate(self.layers):
            output = layer(src=output, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class MSFMLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_blocks=1,
    ):
        super().__init__()
        
        self.self_attn = AFNO3D(
            hidden_size=d_model,
            num_blocks=n_blocks,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1,
            hidden_size_factor=1,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn) 
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos):
        inp = src + pos
        src2 = self.self_attn(inp)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # ffn
        src = self.forward_ffn(src)
        return src


class AFNO3D(nn.Module):
    """
    3D Adaptive Fourier Neural Operator

    Args:
        hidden_size: channel dimension size
        num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
        sparsity_threshold: lambda for softshrink
        hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^3 less FLOPs)
        hidden_size_factor: multiplicative factor for the hidden layer size
    """

    def __init__(
        self,
        hidden_size,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1,
        hidden_size_factor=1,
    ):
        super().__init__()
        assert hidden_size % num_blocks == 0, (
            f"hidden_size {hidden_size} should be divisible by num_blocks {num_blocks}"
        )
        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        # Parameters for the first transformation
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

        # Parameters for the second transformation
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
        """
        Input shape  : B, D, H, W, C
        Output shape : B, D, H, W, C
        """
        bias = x
        dtype = x.dtype
        x = x.float()

        B, D, H, W, C = x.shape
        # Apply 3D FFT
        x = torch.fft.rfftn(x, dim=(1, 2, 3), norm="ortho")
        x = x.reshape(B, x.shape[1], x.shape[2], x.shape[3], self.num_blocks, self.block_size)
        # Initialize output tensors
        o1_real = torch.zeros(
            [
                B,
                x.shape[1],
                x.shape[2],
                x.shape[3],
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
            ],
            device=x.device,
        )
        o1_imag = torch.zeros_like(o1_real)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros_like(o2_real)

        # Calculate modes to keep based on hard thresholding
        total_modes_d = D
        total_modes_h = H
        total_modes_w = W // 2 + 1  # Using rfftn, so fewer modes in last dimension

        kept_modes_d = int(total_modes_d * self.hard_thresholding_fraction)
        kept_modes_h = int(total_modes_h * self.hard_thresholding_fraction)
        kept_modes_w = int(total_modes_w * self.hard_thresholding_fraction)

        # First transformation
        o1_real[:, :kept_modes_d, :kept_modes_h, :kept_modes_w] = F.relu(
            torch.einsum(
                "...bi,bio->...bo",
                x[:, :kept_modes_d, :kept_modes_h, :kept_modes_w].real,
                self.w1[0],
            )
            - torch.einsum(
                "...bi,bio->...bo",
                x[:, :kept_modes_d, :kept_modes_h, :kept_modes_w].imag,
                self.w1[1],
            )
            + self.b1[0]
        )

        o1_imag[:, :kept_modes_d, :kept_modes_h, :kept_modes_w] = F.relu(
            torch.einsum(
                "...bi,bio->...bo",
                x[:, :kept_modes_d, :kept_modes_h, :kept_modes_w].imag,
                self.w1[0],
            )
            + torch.einsum(
                "...bi,bio->...bo",
                x[:, :kept_modes_d, :kept_modes_h, :kept_modes_w].real,
                self.w1[1],
            )
            + self.b1[1]
        )

        # Second transformation
        o2_real[:, :kept_modes_d, :kept_modes_h, :kept_modes_w] = (
            torch.einsum(
                "...bi,bio->...bo",
                o1_real[:, :kept_modes_d, :kept_modes_h, :kept_modes_w],
                self.w2[0],
            )
            - torch.einsum(
                "...bi,bio->...bo",
                o1_imag[:, :kept_modes_d, :kept_modes_h, :kept_modes_w],
                self.w2[1],
            )
            + self.b2[0]
        )

        o2_imag[:, :kept_modes_d, :kept_modes_h, :kept_modes_w] = (
            torch.einsum(
                "...bi,bio->...bo",
                o1_imag[:, :kept_modes_d, :kept_modes_h, :kept_modes_w],
                self.w2[0],
            )
            + torch.einsum(
                "...bi,bio->...bo",
                o1_real[:, :kept_modes_d, :kept_modes_h, :kept_modes_w],
                self.w2[1],
            )
            + self.b2[1]
        )

        # Combine real and imaginary parts
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        # Reshape back to original dimensions
        x = x.reshape(B, x.shape[1], x.shape[2], x.shape[3], C)
        # Inverse FFT
        x = torch.fft.irfftn(x, s=(D, H, W), dim=(1, 2, 3), norm="ortho")
        x = x.type(dtype)

        return x + bias


class AFNO1D(nn.Module):
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
        hard_thresholding_fraction=1,
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
        bias = x

        dtype = x.dtype
        x = x.float()
        B, N, C = x.shape
        x = torch.fft.rfft(x, dim=1, norm="ortho")
        x = x.reshape(B, N // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros(
            [B, N // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor],
            device=x.device,
        )
        o1_imag = torch.zeros(
            [B, N // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor],
            device=x.device,
        )
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = N // 2 + 1
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
        x = x.reshape(B, N // 2 + 1, C)
        x = torch.fft.irfft(x, n=N, dim=1, norm="ortho")
        x = x.type(dtype)
        return x + bias
