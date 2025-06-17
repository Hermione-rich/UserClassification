'''
file description: 执行模型训练
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import torch
import torch.nn as nn
from loguru import logger
import argparse
from torch.utils.tensorboard import SummaryWriter

from src.train import train, test
from src.model import MyModel
from src.utils import *
from src.predict import predict

CONFIG_PATH = "config/config.json"
tensorboard_log_dir = "log/tensorboard"

parser = argparse.ArgumentParser()
parser.add_argument("--save_model", default=False, action="store_true", help="是否保存模型参数，默认为False")
parser.add_argument("--hitk", default="1,5", help="hit@k中k的取值，取值用,号连接")
runtime_args = parser.parse_args()

logger.add("log/log_run_{time}.txt", format="{time} {level} {message}")

writer = SummaryWriter(log_dir=tensorboard_log_dir)
logger.info(f"Tensorboard logging to {tensorboard_log_dir}")

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Current device is {device}")
    set_random_seed()
    with open(CONFIG_PATH, "r") as jsonfile:
        # loading model's configuration
        config = json.load(jsonfile)
    label_list = set()
    with open(config["label_path"], "r") as f:
        for line in f.readlines():
            label = line.strip()
            label_list.add(label)
    label_num = len(label_list)

    # create model and train & test
    model = MyModel(device, label_num, config)
    model = model.to(device=device)
    train(model, device, config, writer)
    test(model, device, config, hitk=runtime_args.hitk)

    # using trained model to predict unlabeled account at once
    # predict(model, device, config)

    # save trained model using pickle
    if runtime_args.save_model:
        save_model(model, config)

    # post-process
    writer.close()