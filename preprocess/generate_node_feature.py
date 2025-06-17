import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from transformers import BertTokenizer, BertModel

class MyDataset(Dataset):
    def __init__(self, profile_list):

        self.length = len(profile_list)
        self.tokenizer = BertTokenizer.from_pretrained("../mbert")
        profile_dict = self.tokenizer(profile_list, max_length=20, truncation=True, padding="max_length")

        self.profile_id = torch.tensor(profile_dict["input_ids"], dtype=torch.long)
        self.profile_mask = torch.tensor(profile_dict["attention_mask"], dtype=torch.long)
        self.profile_type = torch.tensor(profile_dict["token_type_ids"], dtype=torch.long)
    
    def __getitem__(self, index):
        item = {
            "profile_id": self.profile_id[index],
            "profile_mask": self.profile_mask[index],
            "profile_type": self.profile_type[index],
        }
        return item

    def __len__(self):
        return self.length
    
def MyDataLoader(profile_list):

    shuffle = False
    dataset = MyDataset(profile_list)
    loader = DataLoader(
        dataset=dataset,
        batch_size=500,
        shuffle=shuffle
    )
    return loader

if __name__=="__main__":

    data_path = "../data/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load all users' texual profiles
    userid_with_profile = {"id": [], "profile": []}
    '''
    profile_path_list = os.listdir(os.path.join(data_path, "phase1_data"))
    print("file in dir: ", profile_path_list)
    for profile_path in profile_path_list:
        if profile_path.split(".")[-1] != "xls" and profile_path.split(".")[-1] != "xlsx":
            continue
        print(f"loading {profile_path}...")
        profile_df = pd.read_excel(f"{os.path.join(data_path, 'phase1_data/', profile_path)}", dtype=str).fillna("")
        for _, row in profile_df.iterrows():
            userid_with_profile["id"].append(row["id"])
            userid_with_profile["profile"].append(row["work"])
    print(f"length of total users' profile: {len(userid_with_profile['id'])}")
    '''
    profile_df = pd.read_csv(os.path.join(data_path, 'phase1_data', 'target_account_processed.csv'), dtype=str).fillna("")
    for _, row in profile_df.iterrows():
        userid_with_profile["id"].append(row["id"])
        userid_with_profile["profile"].append(row["work"])
    print(f"length of total users' profile: {len(userid_with_profile['id'])}")

    # generate users' features using bert
    feature_list = []
    bert_model = BertModel.from_pretrained("../mbert").to(device=device)
    dataloader = MyDataLoader(userid_with_profile["profile"])
    for _, batch in tqdm(enumerate(dataloader)):
        profile_id = batch["profile_id"].to(device)
        profile_mask = batch["profile_mask"].to(device)
        profile_type = batch["profile_type"].to(device)

        profile_output = bert_model(
            input_ids=profile_id,
            attention_mask=profile_mask,
            token_type_ids=profile_type,
            return_dict=True
        )
        profile_pooler_output = profile_output.pooler_output
        feature_list.extend(profile_pooler_output.detach().to('cpu').numpy().tolist())
    
    # 将userid: feature的字典保存为.pkl
    userid2feature = dict()
    userid2feature[0] = [0] * 768
    user_id_list = userid_with_profile["id"]
    for idx in range(len(user_id_list)):
        userid2feature[user_id_list[idx]] = feature_list[idx]
    # f_save = open(os.path.join(data_path, "phase1_data", 'account_feature.pkl'), 'wb')
    # pickle.dump(userid2feature, f_save)
    # f_save.close()
    
    # 将用户feature的列表保存为.pkl
    feature_list.insert(0, [0] * 768) # append [pad] token
    f_save = open(os.path.join(data_path, "phase1_data", 'account_feature.pkl'), 'wb')
    pickle.dump(np.array(feature_list), f_save)
    f_save.close()
    
    # 将userid: feature的字典保存为txt
    with open(os.path.join(data_path, "phase1_data", 'account_feature.txt'), "w") as f:
        for userid, feature in userid2feature.items():
            f.write(f"{userid}\t{feature}\n")

    # 保存用户名字id和数值型id的映射关系
    index = 1
    with open(os.path.join(data_path, "phase1_data", 'user_to_id.txt'), "w") as f:
        for user_id in user_id_list:
            f.write(f"{user_id}\t{index}\n")
            index += 1