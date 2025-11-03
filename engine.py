import math
import os
import sys
from typing import Iterable

from utils.utils import slprint, to_device
import numpy as np
import torch

import utils.misc as utils
from datasets.coco_eval import CocoEvaluator


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    wo_class_error=False,
    lr_scheduler=None,
    args=None,
    logger=None,
    ema_m=None,
):
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)


    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    metric_logger.add_meter(
        "grad_norm", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )

    header = "Epoch: [{}]".format(epoch)
    print_freq = 500

    _cnt = 0

    for idx, (samples, patches, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, logger=logger)
    ):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast(enabled=args.amp, device_type="cuda"):

            outputs = model(samples, patches)

            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            losses = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            )

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # original backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                grad_total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm
                )
            optimizer.step()



        metric_logger.update(
            loss=loss_value,
        )  # **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if "class_error" in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        
        _cnt += 1

    if getattr(criterion, "loss_weight_decay", False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, "tuning_matching", False):
        criterion.tuning_matching(epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    print("Averaged stats:", metric_logger)
    resstat = {
        k: meter.global_avg
        for k, meter in metric_logger.meters.items()
        if meter.count > 0
    }
    if getattr(criterion, "loss_weight_decay", False):
        resstat.update({f"weight_{k}": v for k, v in criterion.weight_dict.items()})
        
    return resstat


@torch.no_grad()
def evaluate(
    model,
    criterion,
    postprocessors,
    data_loader,
    base_ds,
    device,
    output_dir,
    wo_class_error=False,
    args=None,
    logger=None,
    epoch=None,
):

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter(
            "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
        )
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    _cnt = 0
    output_state_dict = {}  # for debug only
    for samples, patches, targets in metric_logger.log_every(
        data_loader, 1000, header, logger=logger
    ):
        samples = samples.to(device)

        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast(enabled=args.amp, device_type="cuda"):

            outputs = model(samples)

            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()))
        #  ,**loss_dict_reduced_scaled,
        #  **loss_dict_reduced_unscaled)
        if "class_error" in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B

        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results)
        }

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        _cnt += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        class_wise_ap = {}
        if "bbox" in coco_evaluator.coco_eval:
            coco_eval = coco_evaluator.coco_eval["bbox"]
            if coco_eval.eval is not None:
                precisions = coco_eval.eval["precision"]  # shape: [T, R, K, A, M]
                cat_ids = coco_eval.params.catIds
                for idx, catId in enumerate(cat_ids):
                    # precision[:, :, k, 0, 2] is for IoU thresholds, all areas, maxDets=100
                    # Mean across IoU thresholds and recall thresholds
                    precision_per_class = precisions[:, :, idx, 0, -1]
                    valid = precision_per_class > -1
                    ap = np.mean(precision_per_class[valid]) if np.any(valid) else float("nan")
                    class_name = coco_evaluator.coco_gt.loadCats([catId])[0]["name"]
                    class_wise_ap[class_name] = ap

        coco_evaluator.summarize()
        print("\nPer-class Average Precision (AP) @[ IoU=0.50:0.95 | area=all | maxDets=100 ]:")
        for class_name, ap in class_wise_ap.items():
            print(f"  {class_name:<30}: {ap:.4f}\n")         

    stats = {
        k: meter.global_avg
        for k, meter in metric_logger.meters.items()
        if meter.count > 0
    }
    if coco_evaluator is not None:
        if "bbox" in postprocessors.keys():
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()

    return stats, coco_evaluator
