import os
import sys
import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

'''用于合并社交信息至friend.txt文件中，合并带标签的账户为target_account.xlsx'''
def merge_data(data_path, dir_name):
    file_list = os.listdir(os.path.join(data_path, dir_name))
    friend_file_list = list()
    sample_file_list = list()
    profile_file_list = list()
    for file_name in file_list:
        if "friend" in file_name:
            friend_file_list.append(file_name)
        elif "sample" in file_name:
            sample_file_list.append(file_name)
        elif "abc" in file_name:
            profile_file_list.append(file_name)
    
    # 生成friend.txt文件
    friend_merge_file = open(os.path.join(data_path, dir_name, "friend.txt"), 'w', encoding='utf8')
    for file_name in friend_file_list:
        for line in open(os.path.join(data_path, dir_name, file_name), encoding='utf8'):  
            friend_merge_file.writelines(line)  
        # friend_merge_file.write('\n')
    friend_merge_file.close()
    
    # 生成target_account.xlsx文件
    xls_list = list()
    for file_name in profile_file_list:
        xls_list.append(pd.read_excel(os.path.join(data_path, dir_name, file_name)).fillna(""))
    merge_df = pd.concat(xls_list)
    merge_df = merge_df.loc[merge_df['warship_id'] != ""]
    merge_df.to_excel(os.path.join(data_path, dir_name, "target_account.xlsx"), 'Sheet1', index=False)
    return

'''基于friend.txt文件生成社交网络图'''
def create_graph(path):
    G = nx.Graph()
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            line = line.split("|")
            assert len(line) == 2
            G.add_edge(line[0], line[1])
    return G

'''在社交网络图中获得节点的一阶（根据depth来确定）邻居'''
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

def filter_label(friend_list, target_label_map):
    friend_list_res = []
    label_list = []
    for f_id in friend_list:
        if target_label_map.get(f_id):
            friend_list_res.append(str(f_id))
            label_list.append(target_label_map[f_id])
    return friend_list_res, label_list

'''用于生成user_to_id.txt文件'''
def generate_user2id():
    index = 1
    with open(os.path.join(data_path, dir_name, 'user_to_id.txt'), "w") as f:
        for _, row in tqdm(account_profile_df_raw.iterrows()):
            f.write(f"{row['id']}\t{index}\n")
            index += 1

'''用于生成army_labels.txt文件'''
def generate_army_labels(target_label_map):
    army_label_set = set()
    for label in target_label_map.values():
        army_label_set.add(label)
    with open(os.path.join(data_path, dir_name, "army_labels.txt"), "w", encoding="utf-8") as f:
        for label in army_label_set:
            f.write(f"{label}\n")


'''
数据预处理函数：
运行脚本示例：python preprocess.py phase2_data    # 注释：phase2_data为待处理的目标数据目录
在运行完preprocess.py文件后，目标数据目录下会生成如下文件：
1. friend.txt
2. target_account.xlsx
3. target_account_processed.csv
4. army_labels.txt
'''
if __name__=="__main__":

    data_path = "../data/"
    dir_name = sys.argv[1]

    try:
        second_arg = sys.argv[2]
        if second_arg == "pred":
            is_pred = True
        else:
            is_pred = False
    except:
        is_pred = False

    # 进行数据的预处理，合并生成friend.txt和target_account.xlsx
    merge_data(data_path, dir_name)

    # 基于friend.txt，通过networkx加载社交网络
    friend_path = os.path.join(data_path, dir_name, "friend.txt")
    G = create_graph(friend_path)

    # 基于target_account.xlsx，加载目标账户信息
    if not is_pred:
        raw_data_name = "target_account.xlsx"
    else:
        raw_data_name = "account_profile.xlsx"
    raw_data_path = os.path.join(data_path, dir_name, raw_data_name)
    account_profile_df_raw = pd.read_excel(raw_data_path, sheet_name="Sheet1", dtype=str).fillna("")

    # 记录用户和warship id之间的映射关系，保存在target_label_map中
    target_label_map = dict()
    for _, row in tqdm(account_profile_df_raw.iterrows()):
        target_label_map[row["id"]] = row["warship_id"]
    target_account_dict = {key:[] for key in ["id", "name", "education", "work", "living", "basic-info", "year-overviews", "friend_label", "friend_id", "label"]}

    # if is_pred=True, load target account
    if is_pred:
        target_account_list = list()
        with open(os.path.join(data_path, dir_name, "target_account_list.txt"), "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                target_account_list.append(line)
        # save user id to identical id mapping
        index = 1
        with open(os.path.join(data_path, dir_name, 'user_to_id.txt'), "w") as f:
            for user_id in target_account_list:
                f.write(f"{user_id}\t{index}\n")
                index += 1
        target_account_dict = {key:[] for key in ["id", "name", "education", "work", "living", "basic-info", "year-overviews", "friend_label", "friend_id"]}
    else:
        generate_user2id()

    generate_army_labels(target_label_map)

    for _, row in tqdm(account_profile_df_raw.iterrows()):
        if is_pred and (row["id"] not in target_account_list or row["warship_id"] != ""):
            # 因为工程师提供的账户属性信息包含了采样的目标账户以及其朋友账户，因此需要将数据中非目标账户给剔除
            print("hello")
            continue;

        friend_list = get_one_hop_neighbors(G, row["id"])
        friend_list, label_list = filter_label(friend_list, target_label_map)

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

        # label id (only for labeled data)
        if not is_pred:
            target_account_dict["label"].append(row["warship_id"])
    
    target_account_df = pd.DataFrame(target_account_dict)
    target_account_df.to_csv(os.path.join(data_path, dir_name, raw_data_name.replace(".xlsx", "_processed.csv")), header=target_account_dict.keys(), index=False, encoding="utf-8")