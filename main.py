# ---------------------------------------------------------
# Importing librarires
import argparse
from pathlib import Path
import torch
import json
import os
import datetime
import numpy as np
import random
from torch.utils.data import DataLoader, DistributedSampler
import time
import gc

# Importing functions and classes
from utils.slconfig import SLConfig
from models.model import build_model
from datasets import build_dataset
from utils import misc
from engine import evaluate, train_one_epoch
from utils.utils import BestMetricHolder, ModelEma, count_parameters_detailed

# -----------------------------------------------------------


def get_args_parser():
    parser = argparse.ArgumentParser("Object Detection for SAR data", add_help=False)

    # Training related
    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        default="config/cfg_dnod.py",
        help="give the path of required config file",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--run_name", default="check", help="Name of the current run")
    parser.add_argument(
        "--train_backbone",
        default=True,
        help="Set True to train final layers of backbone (Finetune)",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--val_batch_size", default=8, type=int)
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )

    # Eval
    parser.add_argument(
        "--eval", default=False, type=bool, help="To do only evaluation"
    )
    parser.add_argument(
        "--save_results",
        default=False,
        type=bool,
        help="To save results during evaluation",
    )

    # Resume training
    parser.add_argument("--resume", default=False, type=bool, help="To resume Training")
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="To resume Training from this checkpoint",
    )


    # dataset parameters

    # Saving and logging related
    parser.add_argument(
        "--output_dir", default="results", help="Path to save trained models"
    )

    # distributed training parameters
    parser.add_argument("--find_unused_params", action="store_true")
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        help="local rank for DistributedDataParallel",
    )
    parser.add_argument("--amp", action="store_true", help="Train with mixed precision")

    return parser


def main(args):
    misc.init_distributed_mode(args)
    # Modifying output_dir acording to run name
    args.output_dir = args.output_dir + "/" + args.run_name
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    time.sleep(args.rank * 0.02)

    # Reading args from config file
    cfg = SLConfig.fromfile(args.config_file)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)  # converts the Namespace object args into a regular Python dictionary
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # Logging args and config file args
    save_json_path = os.path.join(args.output_dir, "config_args_all.json")
    with open(save_json_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print(vars(args))
    print("Done Loading Args and config file from -> ''{}''".format(args.config_file))

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False

    # Build dataset
    dataset_train = build_dataset(image_set="train", args=args)
    dataset_val = build_dataset(image_set="val", args=args)
    

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.train_batch_size, drop_last=True)

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=misc.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    data_loader_val = DataLoader(
        dataset_val,
        args.val_batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=misc.collate_fn,
        num_workers=args.num_workers,
    )

    # Build model
    model, criterion, postprocessors = build_model(args)
    model.to(args.device)

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], 
                                                          find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(" ||| ")
    print(
        "Backbone params:",
        sum(
            p.numel()
            for p in model_without_ddp.backbone.parameters()
            if p.requires_grad
        )
        / 1e6,
        "M",
    )
    print(
        "Total params:",
        sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad) / 1e6,
        "M",
    )
    print(
        "detector params:",
        sum(
            p.numel()
            for p in model_without_ddp.transformer.parameters()
            if p.requires_grad
        )
        / 1e6,
        "M",
    )
    param_dicts = misc.get_param_dict(args, model_without_ddp)
    # Build Optimizer
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    base_ds = dataset_val.coco
    output_dir = Path(args.output_dir)

    if args.resume or args.eval:
        if os.path.exists(os.path.join(args.output_dir, "checkpoint.pth")):
            if args.checkpoint_path:
                path = args.checkpoint_path
            else:
                path = os.path.join(args.output_dir, "checkpoint.pth")

            checkpoint = torch.load(path, map_location=args.device, weights_only=False)
            model_without_ddp.load_state_dict(checkpoint["model"])

            if (
                not args.eval
                and "optimizer" in checkpoint
                and "lr_scheduler" in checkpoint
                and "epoch" in checkpoint
            ):
                optimizer.load_state_dict(checkpoint["optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                args.start_epoch = checkpoint["epoch"] + 1
            print("\n\n Checkpoint loaded")

        else:
            ValueError("No check point found to resume Training, Started training without checkpoint")

    # TODO
    # args.eval -> for only evaluation, if not train + evaluation
    if args.eval:
        test_stats, coco_evaluator = evaluate(
            model,
            criterion,
            postprocessors,
            data_loader_val,
            base_ds,
            args.device,
            args.output_dir,
            wo_class_error=False,
            args=args,
        )
        if args.output_dir:
            misc.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth"
            )
        log_stats = {**{f"test_{k}": v for k, v in test_stats.items()}}
        if args.output_dir and misc.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        if args.distributed:
            torch.distributed.destroy_process_group()
        return 0

    print("Start training")
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=False)
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            args.device,
            epoch,
            args.clip_max_norm,
            lr_scheduler=lr_scheduler,
            args=args,
        )

        lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]  # Recent
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")

            for (checkpoint_path) in checkpoint_paths:  # [checkpoint, checkpoint{epoch:04}]
                weights = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                }
                misc.save_on_master(weights, checkpoint_path)

        # eval
        test_stats, coco_evaluator = evaluate(
            model,
            criterion,
            postprocessors,
            data_loader_val,
            base_ds,
            args.device,
            args.output_dir,
            wo_class_error=False,
            args=args,
            logger=None,
            epoch=epoch,
        )
        print("Done Evaluating images \n ")
        map_regular = test_stats["coco_eval_bbox"][0]
        _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
        if _isbest:
            checkpoint_path = output_dir / "checkpoint_best_regular.pth"
            misc.save_on_master(
                {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                },
                checkpoint_path,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val{k}": v for k, v in test_stats.items()},
        }

        log_stats.update(best_map_holder.summary())

        ep_paras = {"epoch": epoch, "n_parameters": n_parameters}
        log_stats.update(ep_paras)
        try:
            log_stats.update({"now_time": str(datetime.datetime.now())})
        except:
            pass

        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats["epoch_time"] = epoch_time_str

        if args.output_dir and misc.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / "eval").mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ["latest.pth"]
                    if epoch % 50 == 0:
                        filenames.append(f"{epoch:03}.pth")
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,output_dir / "eval" / name)
       
        gc.collect()
        torch.cuda.empty_cache()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    if args.distributed:
        torch.distributed.destroy_process_group()
    # remove the copied files.
    copyfilelist = vars(args).get("copyfilelist")
    if copyfilelist and args.local_rank == 0:
        from datasets.data_utils import remove

        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
