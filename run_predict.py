'''
file description: 单次执行预测
'''

import os
import json
import torch
import torch.nn as nn
from loguru import logger
import argparse

from src.utils import *
from src.model import MyModel
from src.predict import predict

parser = argparse.ArgumentParser()
parser.add_argument("--pred_file", default=None, type=str, help="待预测文件的路径")
runtime_args = parser.parse_args()

logger.add("log/log_pred_{time}.txt", format="{time} {level} {message}")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CONFIG_PATH = "config/config.json"

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Current device is {device}")
    with open(CONFIG_PATH, "r") as jsonfile:
        # loading model's configuration
        config = json.load(jsonfile)
    label_list = set()
    with open(config["label_path"], "r") as f:
        for line in f.readlines():
            label = line.strip()
            label_list.add(label)
    label_num = len(label_list)
    
    if runtime_args.pred_file:
        config["pred_path"] = runtime_args.pred_file
    logger.info(f"The path of prediction file is {config['pred_path']}")
    
    # load model architecture and its state dict
    logger.info(f"Loading saved model in {config['save_path']}")
    model = MyModel(device, label_num, config).to(device)
    model.load_state_dict(torch.load(config["save_path"]))
    predict(model, device, config)