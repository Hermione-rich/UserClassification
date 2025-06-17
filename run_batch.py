'''
file description: 单次批量执行预测
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

from src.utils import *
from src.model import MyModel
from src.predict import predict

logger.add("log/log_pred_{time}.txt", format="{time} {level} {message}")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CONFIG_PATH = "config/config.json"

def create_graph(path):
    G = nx.Graph()
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines(), desc='', mininterval=30): # ncols=10使进度条在单行内滚动 ，较为简洁
            line = line.strip()
            line = line.split("|")
            assert len(line) == 2
            G.add_edge(line[0], line[1])
    return G

def get_neighbors(G, node, depth=1):
    output = {}
    layers = dict(nx.bfs_successors(G, source=node, depth_limit=depth))
    nodes = [node]
    for i in range(1,depth+1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x,[]))
        nodes = output[i]
    return output

def get_one_hop_neighbors(G, node):
    try:
        res = G.neighbors(node)
        return list(res)
    except:
        # print(f"unk node: {node}")
        return []

def process_text(text):
    text = text.replace("\"", "''")
    text = text.replace("\'", "''")
    text = text.replace("&&&", ",")
    text = text.strip(",")
    return text

def filter_label(friend_list, user_label_map):
    friend_list_res = []
    label_list = []
    for f_id in friend_list:
        if user_label_map.get(f_id):
            friend_list_res.append(f_id)
            label_list.append(user_label_map[f_id])
    return friend_list_res, label_list

def file_preprocess(dir_name, file_id, label_list):

    ''' load social link via networkx '''
    logger.info("Loading user social link information")
    friend_path = os.path.join(dir_name, "friend_"+file_id+".txt")
    G = create_graph(friend_path)

    ''' load account profile information '''
    logger.info("Loading user profile information")
    raw_data_path = os.path.join(dir_name, "abc_"+file_id+".xls")
    excel_file = pd.ExcelFile(raw_data_path)
    account_profile_df_raw = pd.DataFrame()
    for sn in excel_file.sheet_names:
        df = pd.read_excel(raw_data_path, sheet_name=sn).fillna("")
        account_profile_df_raw = pd.concat([account_profile_df_raw, df])

    ''' record user to warship id mappings '''
    user_label_map = dict()
    user_id_list = []
    for _, row in tqdm(account_profile_df_raw.iterrows(), desc='', mininterval=30):
        warship_id = str(int(row["warship_id"])) if row["warship_id"] != "" else ""
        if warship_id == "":
            user_label_map[row["id"]] = ""
        elif warship_id not in label_list:
            user_label_map[row["id"]] = ""
        else:
            user_label_map[row["id"]] = warship_id
        user_id_list.append(row["id"])

    ''' load predictable target account '''
    target_account_list = list()
    with open(os.path.join(dir_name, "sample_"+file_id+".txt"), "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            target_account_list.append(line)
    
    ''' save user id to identical id mapping '''
    index = 1
    with open(os.path.join(dir_name, 'user_to_id.txt'), "w") as f:
        for user_id in user_id_list:
            f.write(f"{user_id}\t{index}\n")
            index += 1

    ''' begin process '''
    logger.info("Begin process the file")
    target_account_dict = {key:[] for key in ["id", "name", "education", "work", "living", "basic-info", "year-overviews", "friend_label", "friend_id"]}
    for _, row in tqdm(account_profile_df_raw.iterrows(), desc='', mininterval=30):
        if row["id"] not in target_account_list or row["warship_id"] != "":
            # 因为工程师提供的账户属性信息包含了采样的目标账户以及其朋友账户，因此需要将数据中非目标账户给剔除
            continue;

        friend_list = get_one_hop_neighbors(G, row["id"])
        friend_list, label_list = filter_label(friend_list, user_label_map)
        if len(friend_list) <= 2: #如果当前账户的已确定朋友数量少于等于2个，就暂时不预测该账户
            continue

        # id/name features
        target_account_dict["id"].append(row["id"])
        target_account_dict["name"].append(row["name"])

        # texutal features
        target_account_dict["education"].append(process_text(row["education"]))
        target_account_dict["work"].append(process_text(row["work"]))
        target_account_dict["living"].append(process_text(row["living"]))
        target_account_dict["basic-info"].append(process_text(row["basic-info"]))
        target_account_dict["year-overviews"].append(process_text(row["year-overviews"]))

        # social link features
        target_account_dict["friend_label"].append(",".join(label_list))
        target_account_dict["friend_id"].append("|*|".join(friend_list))
    
    IsNoData = True if len(target_account_dict["id"]) == 0 else False

    processed_file_path = os.path.join(dir_name, file_id+"_processed.csv")
    target_account_df = pd.DataFrame(target_account_dict)
    target_account_df.to_csv(processed_file_path, header=target_account_dict.keys(), index=False, encoding="utf-8")
    logger.info(f"Processed file is saved in {processed_file_path}")

    return processed_file_path, IsNoData

##### main function #####
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Current device is {device}")
    with open(CONFIG_PATH, "r") as jsonfile:
        config = json.load(jsonfile) # loading model's configuration
    label_list = set()
    with open(config["label_path"], "r") as f:
        for line in f.readlines():
            label = line.strip()
            label_list.add(label)
    label_num = len(label_list)

    # 获取待处理文件的id列表
    dir_name = "/home/gpu4/173"
    files = os.listdir(dir_name)
    file_ids = list()
    for file in files:
        if file[-4:] == ".xls":
            file_ids.append(int(file[4:-4]))
    file_ids = sorted(file_ids)
    file_ids = file_ids[10:]
    logger.info(f"File ids are: {file_ids}")
    file_ids = [str(file_id) for file_id in file_ids]
    
    # load model architecture and its state dict
    logger.info(f"Loading saved model in {config['save_path']}")
    model = MyModel(device, label_num, config).to(device)
    model.load_state_dict(torch.load(config["save_path"]))

    # 遍历所有文件id，并进行数据预处理和模型打标
    for file_id in file_ids:
        logger.info(f"The current file id is {file_id}")
        '''数据预处理'''
        processed_file_path, _ = file_preprocess(dir_name, file_id, label_list)
        config["pred_path"] = processed_file_path
        '''模型预测'''
        predict(model, device, config, file_id, threshold=0.8)
    
    logger.info("End prediction")