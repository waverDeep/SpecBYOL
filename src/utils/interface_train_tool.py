import argparse
import json
import random
import numpy as np
import os
import torch.cuda
import src.optimizers.optimizer as optimizers
import src.data.dataset as dataset
import src.models.model as model_pack
import src.utils.interface_tensorboard as tensorboard
from apex.parallel import DistributedDataParallel as DDP
from datetime import datetime
import src.utils.interface_plot as plots
import src.utils.interface_file_io as file_io


def setup_seed(random_seed=777):
    torch.manual_seed(random_seed)
    # torch.backends.cudnn.deterministic = True # 연산 속도가 느려질 수 있음
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def setup_timestamp():
    now = datetime.now()
    return "{}_{}_{}_{}_{}_{}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)


def setup_config(configuration):
    return file_io.load_json_config(configuration)


def make_target(speaker_id, speaker_dict):
    targets = torch.zeros(len(speaker_id)).long()
    for idx in range(len(speaker_id)):
        targets[idx] = speaker_dict[speaker_id[idx]]
    return targets


def save_checkpoint(config, model, optimizer, loss, epoch, mode="best", date=""):
    if not os.path.exists(os.path.join(config['checkpoint_save_directory_path'], config['checkpoint_file_name'])):
        file_io.make_directory(os.path.join(config['checkpoint_save_directory_path'], config['checkpoint_file_name']))
    base_directory = os.path.join(config['checkpoint_save_directory_path'], config['checkpoint_file_name'])
    if mode == "best":
        file_path = os.path.join(base_directory,
                                 config['checkpoint_file_name'] + "-model-best-{}-epoch-{}.pt".format(date, epoch))
    elif mode == 'step':
        file_path = os.path.join(base_directory,
                                 config['checkpoint_file_name'] + "-model-{}-epoch-{}.pt".format(date, epoch))

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch, "loss": loss}, file_path)