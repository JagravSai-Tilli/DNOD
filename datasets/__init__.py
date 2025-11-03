import torch.utils.data
import torchvision
from .coco import build_SAR_DET100K
def build_dataset(image_set, args):
    print(args.dataset_file)
    if args.dataset_file == 'SAR_DET100K':
        return build_SAR_DET100K(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
