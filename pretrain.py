import argparse
import os
import torch
import torch.distributed as dist

from src.common import PATHS
from src.tasks.pretrain_task import Pretraining
from src.utils.config import Config
from src.utils.tools import control_randomness, make_dir_if_not_exists, parse_config

def main_worker():
    # --------- Get Environment Variables ----------
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    # --------- Load Config ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml")
    parser.add_argument("--patch_len", type=int, default=None)
    parser.add_argument("--patch_stride_len", type=int, default=None)
    parser.add_argument("--seq_len_channel", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--pos_embed_type", type=str, default=None)
    parser.add_argument("--encoder_type", type=str, default=None)
    args_cmd = parser.parse_args()

    config = Config(
        config_file_path=args_cmd.config,
        default_config_file_path="configs/default.yaml"
    ).parse()

    control_randomness(config["random_seed"])

    config["device"] = local_rank
    config["rank"] = global_rank
    config["world_size"] = world_size
    config["distributed"] = True
    config["checkpoint_path"] = PATHS.CHECKPOINTS_DIR
    ### Override config with command line arguments
    if args_cmd.patch_len is not None:
        config["patch_len"] = args_cmd.patch_len
    if args_cmd.patch_stride_len is not None:
        config["patch_stride_len"] = args_cmd.patch_stride_len
    if args_cmd.seq_len_channel is not None:
        config["seq_len_channel"] = args_cmd.seq_len_channel
    if args_cmd.train_batch_size is not None:
        config["train_batch_size"] = args_cmd.train_batch_size
    if args_cmd.val_batch_size is not None:
        config["val_batch_size"] = args_cmd.val_batch_size
    if args_cmd.pos_embed_type is not None:
        config["pos_embed_type"] = args_cmd.pos_embed_type
    if args_cmd.max_epochs is not None:
        config["max_epochs"] = args_cmd.max_epochs
    if args_cmd.encoder_type is not None:
        config["encoder_type"] = args_cmd.encoder_type
    make_dir_if_not_exists("./checkpoints/")

    args = parse_config(config)

    print(f"[Rank {global_rank}] Running with config:\n{args}\n")
    task_obj = Pretraining(args=args)

    if global_rank == 0:
        task_obj.setup_logger(notes="Pre-training runs")

    task_obj.train()

    if global_rank == 0:
        task_obj.end_logger()

    dist.destroy_process_group()


if __name__ == "__main__":
    main_worker()