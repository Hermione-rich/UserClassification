import os
import pandas as pd
import numpy as np

'''用于合并不同阶段下的训练数据和测试数据'''
def merge_train_test(data_path, merge_dir_name, target_dir_name):
    train_df_list = list()
    test_df_list = list()
    for dir_name in merge_dir_name:
        train_df_list.append(pd.read_csv(os.path.join(data_path, dir_name, "train_data.csv")))
        test_df_list.append(pd.read_csv(os.path.join(data_path, dir_name, "test_data.csv")))
    train_df = pd.concat(train_df_list)
    test_df = pd.concat(test_df_list)
    train_df.to_csv(os.path.join(data_path, target_dir_name, "merged_train_data.csv"), index=False)
    test_df.to_csv(os.path.join(data_path, target_dir_name, "merged_test_data.csv"), index=False)

'''用于合并不同阶段下的user_to_id.txt'''
def merge_user2id(data_path, merge_dir_name, target_dir_name):
    idx = 1
    user2id = dict()
    for dir_name in merge_dir_name:
        with open(os.path.join(data_path, dir_name, "user_to_id.txt"), "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.split("\t")
                if not user2id.get(line[0]):
                    user2id[line[0]] = idx
                    idx += 1
    with open(os.path.join(data_path, target_dir_name, "merged_user_to_id.txt"), "w", encoding="utf-8") as f:
        for k,v in user2id.items():
            f.write(f"{k}\t{v}\n")

'''用于合并不同阶段下的army_labels.txt'''
def merge_army_labels(data_path, merge_dir_name, target_dir_name):
    army_label_set = set()
    for dir_name in merge_dir_name:
        with open(os.path.join(data_path, dir_name, "army_labels.txt"), "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                army_label_set.add(str(line))
    with open(os.path.join(data_path, target_dir_name, "army_labels.txt"), "w", encoding="utf-8") as f:
        for label in army_label_set:
            f.write(f"{label}\n")


'''
数据合并函数：
主要用于将多个数据目录中的数据进行合并
'''
if __name__=="__main__":
    data_path = "../data/" # 数据目录
    merge_dir_name = ["phase1_data", "phase2_data", "phase3_data"] # 具体需要合并的数据目录
    target_dir_name = "phase3_data" # 目标输出目录

    merge_train_test(data_path, merge_dir_name, target_dir_name)
    merge_user2id(data_path, merge_dir_name, target_dir_name)
    merge_army_labels(data_path, merge_dir_name, target_dir_name)