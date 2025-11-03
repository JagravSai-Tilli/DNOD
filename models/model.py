from models.backbone import build_backbone
from .dnod_transformer import DNOD_transformer
import torch.nn as nn
import torch
import math
from .model_utils import MLP, sigmoid_focal_loss
from models.position_encoding import build_position_encoding
import copy
from typing import List
from utils.misc import (
    NestedTensor,
    inverse_sigmoid,
    accuracy,
    nested_tensor_from_tensor_list,
    interpolate,
    is_dist_avail_and_initialized,
    get_world_size,
    dice_loss,
)
import torch.nn.functional as F
from .matcher_o2m import Stage2Assigner
from .matcher import HungarianMatcher
from utils import box_ops
from torchvision.ops.boxes import nms

class DNOD(nn.Module):
    def __init__(
        self,
        args,
        backbone,
        transformer,
        aux_loss=False,
    ):
        super().__init__()

        self.num_queries = args.num_queries
        self.transformer = transformer
        self.num_classes = args.num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = args.num_feature_levels
        self.nheads = args.nheads

        # setting query dim
        self.query_dim = args.query_dim
        self.label_enc = nn.Embedding(args.num_classes + 1, hidden_dim)

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.args = args

        # prepare class & box embed
        _class_embed = nn.Linear(hidden_dim, self.num_classes)
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3) 

        # init the two embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        
        box_embed_layerlist = [_bbox_embed for i in range(args.dec_layers)]
        class_embed_layerlist = [_class_embed for i in range(args.dec_layers)]
        
        # These will be applied to bbox embed
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
        self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

        for layer in self.transformer.decoder.layers:
            layer.label_embedding = None
        self.label_embedding = None
            
        # backbone upsampling
        self.position_embedding = build_position_encoding(args)
        self.num_channels = backbone.num_channels
        if args.num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            hidden_dim = args.hidden_dim
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                if _ == 0:
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=2),
                        nn.GroupNorm(32, hidden_dim),
                    ))
                if _ == 1:
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
                if _ == 2:
                    input_proj_list.append(nn.Sequential(
                        nn.ConvTranspose2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=2, stride=2),
                        nn.GroupNorm(32, hidden_dim),
                    ))
            for _ in range(args.num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
            
    def forward(self, samples: NestedTensor, patches: torch.Tensor = None, targets: List = None):
            
        features = []
        pos = []
        mask = []
        
        xs = self.backbone(samples)
        for idx, (name, x) in enumerate(sorted(xs.items())):
            features.append(self.input_proj[idx](x.tensors))
            pos.append(self.position_embedding[idx](features[-1], xs['1'].mask).to(x.tensors.dtype))
            mask.append(xs['1'].mask)
        

        hs, hs_o2m, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            features, mask, pos,
        )
        
        hs[0] += self.label_enc.weight[0, 0] * 0.0

        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        
        outputs_coord_list = torch.stack(outputs_coord_list)
        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        )
        
        if hs_o2m is not None:  
            outputs_coord_list_o2m = []
            for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
                zip(reference[:-1], self.bbox_embed, hs_o2m)
            ):
                layer_delta_unsig = layer_bbox_embed(layer_hs)
                layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
                layer_outputs_unsig = layer_outputs_unsig.sigmoid()
                outputs_coord_list_o2m.append(layer_outputs_unsig)
                
            outputs_coord_list_o2m = torch.stack(outputs_coord_list_o2m)

            outputs_class_o2m = torch.stack(
                [
                    layer_cls_embed(layer_hs)
                    for layer_cls_embed, layer_hs in zip(self.class_embed, hs_o2m)
                ]
            )
        
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}
        
        if hs_o2m is not None:
            out['o2m_outputs'] = {'pred_logits': outputs_class_o2m[-1], 'pred_boxes': outputs_coord_list_o2m[-1]}
        
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord_list)
            
        if hs_o2m is not None:
            if self.aux_loss:
                out['o2m_outputs']['aux_outputs'] = self._set_aux_loss(outputs_class_o2m, outputs_coord_list_o2m)
                
        # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])
            out["interm_outputs"] = {
                "pred_logits": interm_class,
                "pred_boxes": interm_coord,
            }
            out["interm_outputs_for_matching_pre"] = {
                "pred_logits": interm_class,
                "pred_boxes": init_box_proposal,
            }

            # prepare enc outputs
            if hs_enc.shape[0] > 1:
                enc_outputs_coord = []
                enc_outputs_class = []
                for layer_id, (
                    layer_box_embed,
                    layer_class_embed,
                    layer_hs_enc,
                    layer_ref_enc,
                ) in enumerate(
                    zip(
                        self.enc_bbox_embed,
                        self.enc_class_embed,
                        hs_enc[:-1],
                        ref_enc[:-1],
                    )
                ):
                    layer_enc_delta_unsig = layer_box_embed(layer_hs_enc)
                    layer_enc_outputs_coord_unsig = (
                        layer_enc_delta_unsig + inverse_sigmoid(layer_ref_enc)
                    )
                    layer_enc_outputs_coord = layer_enc_outputs_coord_unsig.sigmoid()
                    # print()
                    layer_enc_outputs_class = layer_class_embed(layer_hs_enc)
                    enc_outputs_coord.append(layer_enc_outputs_coord)
                    enc_outputs_class.append(layer_enc_outputs_class)

                out["enc_outputs"] = [
                    {"pred_logits": a, "pred_boxes": b}
                    for a, b in zip(enc_outputs_class, enc_outputs_coord)
                ]


        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


class SetCriterion(nn.Module):
    """This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses,
                 o2m_matcher_threshold=0.4, o2m_matcher_k=6, use_indices_merge=False):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.use_indices_merge = use_indices_merge
        self.matcher_o2m = Stage2Assigner(k=o2m_matcher_k, threshold=o2m_matcher_threshold)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = (
            sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                num_boxes,
                alpha=self.focal_alpha,
                gamma=2,
            )
            * src_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        if num_boxes == 0:
            print("model.py line 349")
            losses["loss_bbox"] = loss_bbox.sum()
        else:
            losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
                None,
            )
        )
        if num_boxes == 0:
            print("model.py line 362")
            losses["loss_giou"] = loss_giou.sum()
        else:
            losses["loss_giou"] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss
        with torch.no_grad():
            if num_boxes == 0:
                print("model.py line 370")
                losses["loss_xy"] = loss_bbox[..., :2].sum()
                losses["loss_hw"] = loss_bbox[..., 2:].sum()
            else:
                losses["loss_xy"] = loss_bbox[..., :2].sum() / num_boxes
                losses["loss_hw"] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(
            src_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, return_indices=False):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        
        o2o_indices_list = []
        
        device = next(iter(outputs.values())).device
        indices_o2o = self.matcher(outputs_without_aux, targets)
        
        o2o_indices_list.append(indices_o2o)
        if return_indices:
            indices0_copy = indices_o2o
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # prepare for dn loss



        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices_o2o, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:

            for idx, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                o2o_indices_list.append(indices)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{idx}": v for k, v in l_dict.items()}
                    losses.update(l_dict)


        # one-to-many losses
        if 'o2m_outputs' in outputs:

            o2m_outputs = outputs['o2m_outputs']
            indices = self.matcher_o2m(o2m_outputs, targets)

            if self.use_indices_merge:
                o2o_indices = o2o_indices_list.pop(0)
                indices = self.indices_merge(self.num_queries, o2o_indices, indices)

            for loss in self.losses:
                kwargs = {}
                l_dict = self.get_loss(loss, o2m_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + '_o2m': v for k, v in l_dict.items()}
                losses.update(l_dict)

            if "aux_outputs" in o2m_outputs:
                for i, aux_outputs in enumerate(o2m_outputs['aux_outputs']):
                    indices = self.matcher_o2m(aux_outputs, targets)

                    if self.use_indices_merge:
                        o2o_indices = o2o_indices_list[i]
                        indices = self.indices_merge(self.num_queries, o2o_indices, indices)
                    
                    for loss in self.losses:
                        if loss == 'masks':
                            # Intermediate masks losses are too costly to compute, we ignore them.
                            continue
                        kwargs = {}
                        if loss == 'labels':
                            # Logging is enabled only for the last layer
                            kwargs['log'] = False
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                        l_dict = {k + f'_{i}_o2m': v for k, v in l_dict.items()}
                        losses.update(l_dict)
                        
        # interm_outputs loss
        if "interm_outputs" in outputs:
            interm_outputs = outputs["interm_outputs"]
            indices = self.matcher(interm_outputs, targets)
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == "masks":
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == "labels":
                    # Logging is enabled only for the last layer
                    kwargs = {"log": False}
                l_dict = self.get_loss(
                    loss, interm_outputs, targets, indices, num_boxes, **kwargs
                )
                l_dict = {k + f"_interm": v for k, v in l_dict.items()}
                losses.update(l_dict)

        # enc output loss
        if "enc_outputs" in outputs:
            for i, enc_outputs in enumerate(outputs["enc_outputs"]):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}
                    l_dict = self.get_loss(
                        loss, enc_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses


    @staticmethod
    def indices_merge(num_queries, o2o_indices, o2m_indices):
        bs = len(o2o_indices)
        temp_indices = torch.zeros(bs, num_queries, dtype=torch.int64).cuda() - 1
        new_one2many_indices = []

        for i in range(bs):
            one2many_fg_inds = o2m_indices[i][0].cuda()
            one2many_gt_inds = o2m_indices[i][1].cuda()
            one2one_fg_inds = o2o_indices[i][0].cuda()
            one2one_gt_inds = o2o_indices[i][1].cuda()
            temp_indices[i][one2one_fg_inds] = one2one_gt_inds
            temp_indices[i][one2many_fg_inds] = one2many_gt_inds
            fg_inds = torch.nonzero(temp_indices[i] >= 0).squeeze(1)
            # fg_inds = torch.argwhere(temp_indices[i] >= 0).squeeze(1)
            gt_inds = temp_indices[i][fg_inds]
            new_one2many_indices.append((fg_inds, gt_inds))

        return new_one2many_indices
    
    
class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    def __init__(self, num_select=100, nms_iou_threshold=-1) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), num_select, dim=1
        )
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        if test:
            assert not not_to_xyxy
            boxes[:, :, 2:] = boxes[:, :, 2:] - boxes[:, :, :2]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if self.nms_iou_threshold > 0:
            item_indices = [
                nms(b, s, iou_threshold=self.nms_iou_threshold)
                for b, s in zip(boxes, scores)
            ]

            results = [
                {"scores": s[i], "labels": l[i], "boxes": b[i]}
                for s, l, b, i in zip(scores, labels, boxes, item_indices)
            ]
        else:
            results = [
                {"scores": s, "labels": l, "boxes": b}
                for s, l, b in zip(scores, labels, boxes)
            ]

        return results


def build_model(args):
    backbone = build_backbone(args)

    dnod_transformer = DNOD_transformer(args=args)

    model = DNOD(args=args, backbone=backbone, transformer=dnod_transformer, aux_loss=True)

    matcher = HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
        focal_alpha=args.focal_alpha,
    )

    # prepare weight dict
    weight_dict = {"loss_ce": args.cls_loss_coef, "loss_bbox": args.bbox_loss_coef}
    weight_dict["loss_giou"] = args.giou_loss_coef
    clean_weight_dict = copy.deepcopy(weight_dict)
    
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update(
                {k + f"_{i}": v for k, v in clean_weight_dict.items()}
            )
        weight_dict.update(aux_weight_dict)

    interm_weight_dict = {}
    try:
        no_interm_box_loss = args.no_interm_box_loss
    except:
        no_interm_box_loss = False
    _coeff_weight_dict = {
        "loss_ce": 1.0,
        "loss_bbox": 1.0 if not no_interm_box_loss else 0.0,
        "loss_giou": 1.0 if not no_interm_box_loss else 0.0,
    }
    try:
        interm_loss_coef = args.interm_loss_coef
    except:
        interm_loss_coef = 1.0
        interm_weight_dict.update(
        {
            k + f"_interm": v * interm_loss_coef * _coeff_weight_dict[k]
            for k, v in clean_weight_dict.items()
        }
    )
    weight_dict.update(interm_weight_dict)
    
    
    o2m_weight_dict = {'loss_ce': args.o2m_cls_loss_coef, 'loss_bbox': args.o2m_bbox_loss_coef, 'loss_giou': args.o2m_giou_loss_coef}
    
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in o2m_weight_dict.items()})
        o2m_weight_dict.update(aux_weight_dict)
    o2m_weight_dict = {k + '_o2m': v for k, v in o2m_weight_dict.items()}
    weight_dict.update(o2m_weight_dict)
    
    
    losses = ["labels", "boxes", "cardinality"]
    
    if args.masks:
        losses += ["masks"]
        
    criterion = SetCriterion(
        args.num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=args.focal_alpha,
        losses=losses,
        o2m_matcher_threshold = args.o2m_matcher_threshold,
        o2m_matcher_k = args.o2m_matcher_k,
        use_indices_merge = args.use_indices_merge
    )
    criterion.to(args.device)
    postprocessors = {
        "bbox": PostProcess(
            num_select=args.num_select, nms_iou_threshold=args.nms_iou_threshold
        )
    }
    return model, criterion, postprocessors
