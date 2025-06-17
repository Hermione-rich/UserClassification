'''
file description: 调度式执行预测
'''

import os
import sys
import json
import torch
import torch.nn as nn
import pandas as pd
from loguru import logger
import networkx as nx
from tqdm import tqdm
import schedule
import time
import shutil # pip install pytest-shutil

from src.utils import *
from src.model import MyModel
from src.predict import predict
from run_batch import create_graph, get_neighbors, get_one_hop_neighbors, process_text, filter_label, file_preprocess

logger.add("log/log_pred_{time}.txt", format="{time} {level} {message}")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CONFIG_PATH = "config/config.json"

def my_job():
    '''
        与工程师联调的文件夹:
        1. dir_name: 待预测文件的目录
        2. dir_name_history: 已处理文件的目录
        3. threshold_pred_base_path: 存预测结果的目录
        其他参数：
        1. threshold: 预设阈值，用于预测结果的阈值过滤
        2. process_num: 处理数量，如果需要处理所有文件，则设置为-1
        3. pred_file_name_template: 预测文件的文件名模版
    '''
    threshold = 0.85
    process_num = -1
    pred_file_name_template = "pred_{0}.txt"
    dir_name = "/home/gpu4/173/javadir/samples"
    dir_name_history = "/home/gpu4/173/javadir/history/samples"
    threshold_pred_base_path = "/home/gpu4/173/javadir/generate"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Current device is {device}")

    # loading model's configuration
    with open(CONFIG_PATH, "r") as jsonfile:
        config = json.load(jsonfile)
    
    # loading label list
    label_list = set()
    with open(config["label_path"], "r") as f:
        for line in f.readlines():
            label = line.strip()
            label_list.add(label)
    label_num = len(label_list)

    # loading file information
    file_list = os.listdir(dir_name)
    file_ids = list()
    for file in file_list:
        if file[-4:] == ".xls":
            file_id = str(file[4:-4])
            '''需要同时判断三个文件存在才可以执行预测'''
            if f"friend_{file_id}.txt" in file_list and f"sample_{file_id}.txt" in file_list:
                file_ids.append(file_id)
    file_ids = sorted(file_ids)
    file_ids = file_ids[:] if process_num == -1 else file_ids[0:process_num]
    # file_ids = file_ids[:30] # 此命令用于选择前xx个文件进行处理，而不处理所有文件
    logger.info(f"File ids are: {file_ids}")
    
    # load model architecture and its state dict
    logger.info(f"Loading saved model in {config['save_path']}")
    model = MyModel(device, label_num, config).to(device)
    model.load_state_dict(torch.load(config["save_path"]))

    processed_file_number = 0
    # 遍历所有待预测文件id，并进行数据预处理和模型打标
    for file_id in file_ids:
        logger.info(f"The current file id is {file_id}")
        try:
            '''step1: 数据预处理'''
            processed_file_path, IsNoData = file_preprocess(dir_name, file_id, label_list)
            if IsNoData: # 如果出现The size of pred dataset is: 0则跳过
                logger.info("No data to predict")
                continue
            config["pred_path"] = processed_file_path
            threshold_pred_path = os.path.join(threshold_pred_base_path, pred_file_name_template.format(file_id))
            '''step2: 模型预测'''
            predict(model, device, config, file_id, threshold=threshold, threshold_pred_path=threshold_pred_path)
            '''step3: 文件后处理'''
            shutil.move(os.path.join(dir_name, f"sample_{file_id}.txt"), dir_name_history)
            shutil.move(os.path.join(dir_name, f"friend_{file_id}.txt"), dir_name_history)
            shutil.move(os.path.join(dir_name, f"abc_{file_id}.xls"), dir_name_history)
            shutil.move(os.path.join(dir_name, f"{file_id}_processed.csv"), dir_name_history)
            processed_file_number += 1
        except Exception as e:
            logger.info(e)
    
    logger.info("End prediction")
    logger.info(f"{processed_file_number} files were processed successfully")

if __name__ == "__main__":
    hour_schedule = 24
    schedule.every(hour_schedule).hours.do(my_job)
    schedule.run_all()
    while True:
        schedule.run_pending()
        time.sleep(1)
    # my_job() # jobs -l