"""
COCO dataset which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import json
from pathlib import Path
import random
import os
import numpy as np
import torch
from tqdm import tqdm
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
from torchvision.transforms import transforms
from datasets.data_utils import preparing_dataset
import datasets.transforms as T
from utils.box_ops import box_cxcywh_to_xyxy, box_iou
from PIL import ImageFilter
__all__ = ["build"]

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        return_masks,
        args=None,
        image_set=None,
    ):
        super(CocoDetection, self).__init__(img_folder, ann_file)

        # self.ids = self.ids[0:100]
        print(f"found {len(self.ids)} Total images in {image_set} set")
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.args = args
        self.image_set = image_set

    def __getitem__(self, idx):
        """
        Output:
            - target: dict of multiple items
                - boxes: Tensor[num_box, 4]. \
                    Init type: x0,y0,h,w. unnormalized data.
                    Final type: cx,cy,w,h. normalized data. 
        """
        try:
            img, target = super(CocoDetection, self).__getitem__(idx)
        except:  
            print("Error idx: {}".format(idx))
            idx += 1
            img, target = super(CocoDetection, self).__getitem__(idx)

        image_id = self.ids[idx]

        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target) 
        patches = image_id
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, patches, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno]
        )
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, args=None):
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    if image_set == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomResize([512], max_size=512),
                normalize,
            ]
        )

    if image_set in ["val", "eval_debug", "train_reg", "test"]:
        return T.Compose(
            [
                T.RandomResize([512], max_size=512),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def build_SAR_DET100K(image_set, args):
    root = Path(args.data_path)
    PATHS = {
        "train": (
            root / "images/train",
            root / "Annotations" / f"train.json",
        ),
        "val": (
            root / "images/val",
            root / "Annotations" / f"val.json",
        ),
        "test": (
            root / "images/test",
            root / "Annotations" / f"test.json",
        ),
    }
    img_folder, ann_file = PATHS[image_set]
    # copy to local path
    if os.environ.get("DATA_COPY_SHILONG") == "INFO":
        preparing_dataset(
            dict(img_folder=img_folder, ann_file=ann_file), image_set, args
        )

    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False
    dataset = CocoDetection(
        img_folder,
        ann_file,
        transforms=make_coco_transforms(image_set, args=args),
        return_masks=args.masks,
        args=args,
        image_set=image_set,
    )

    return dataset


