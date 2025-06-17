import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers import logging
logging.set_verbosity_error()
from loguru import logger

class MyDataset(Dataset):
    def __init__(self, phase, data_path, label_path, **kwargs):
        
        self.phase = phase

        data = pd.read_csv(data_path, dtype=str).fillna("")
        logger.info(f"The size of {self.phase} dataset is: {data.shape[0]}")
        self.tokenizer = BertTokenizer.from_pretrained("./mbert")
        user_mapping_path = data_path.split("/")
        user_mapping_path[-1] = "user_to_id.txt"
        self.user2id_dict = self.load_user_mapping(user_mapping_path = "/".join(user_mapping_path))
        
        # 处理用户名到id的映射
        user_name = data["id"].tolist()
        user_id = self.map_user_to_id(user_name)
        self.user_id = torch.tensor(user_id, dtype=torch.float)
        
        # 处理账户的标签label
        self.label_dict = self.load_label_mapping(label_path)
        if self.phase != "pred":
            label = data["label"].map(self.label_dict).tolist()
            self.label = torch.tensor(label, dtype=torch.long)

        # 处理目标账户的profile信息
        profile_list = data["work"].tolist()
        profile_dict = self.tokenizer(profile_list, max_length=kwargs["text_max_length"], truncation=True, padding="max_length")

        self.profile_id = torch.tensor(profile_dict["input_ids"], dtype=torch.long)
        self.profile_mask = torch.tensor(profile_dict["attention_mask"], dtype=torch.long)
        self.profile_type = torch.tensor(profile_dict["token_type_ids"], dtype=torch.long)

        # 处理目标账户的关联账户特征
        friend_label_list = data["friend_label"].tolist()
        friend_name_list = data["friend_id"].tolist()
        friend_name_list = self.map_friend_to_id(friend_name_list)
        friend_label_id, friend_name, friend_label_mask = self.process_friend_token(friend_label_list, friend_name_list, kwargs["friend_max_length"])
        
        self.friend_label_id = torch.tensor(friend_label_id, dtype=torch.long)
        self.friend_label_mask = torch.tensor(friend_label_mask, dtype=torch.long)
        self.friend_name = torch.tensor(friend_name, dtype=torch.long)
        
        self.label_num = len(self.label_dict.keys())
        # self.save_intermediate_result()
    
    def map_user_to_id(self, user_name):
        mapped_result = list()
        for user in user_name:
            mapped_result.append(self.user2id_dict[user])
        return mapped_result
    
    def map_friend_to_id(self, friend_name):
        mapped_result = list()
        for friend in friend_name:
            tmp = []
            friend = friend.split("|*|")
            for name in friend:
                if name != "":
                    tmp.append(self.user2id_dict[name])
            mapped_result.append(tmp)
        return mapped_result

    def load_user_mapping(self, user_mapping_path):
        user2id_dict = dict()
        with open(user_mapping_path, "r") as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                user2id_dict[line[0]] = int(line[1])
        return user2id_dict

    def load_label_mapping(self, label_path):
        label_dict = dict()
        curr_id = 0
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                label_dict[line] = curr_id
                curr_id += 1
        return label_dict

    def process_friend_token(self, friend_label_list, friend_name_list, max_length):
        friend_label_id = list()
        friend_name = list()
        friend_label_mask = list()
        for f_l, f_n in zip(friend_label_list, friend_name_list):
            # 处理每个用户的朋友信息
            f_l_list = f_l.split(",")
            f_n_list = f_n
            if f_l_list[0] == "":
                friend_label_id.append([0] * max_length)
                friend_name.append([0] * max_length)
                friend_label_mask.append([0] * max_length)
                continue
            f_l_list = [self.label_dict[i]+1 for i in f_l_list]
            if len(f_l_list) >= max_length:
                friend_label_id.append(f_l_list[:max_length])
                friend_name.append(f_n_list[:max_length])
                friend_label_mask.append([1] * max_length)
            elif len(f_l_list) < max_length:
                friend_label_mask.append([1] * len(f_l_list) + [0] * (max_length - len(f_l_list)))
                f_l_list += [0] * (max_length - len(f_l_list))
                friend_label_id.append(f_l_list)
                f_n_list += [0] * (max_length - len(f_n_list))
                friend_name.append(f_n_list)
        return friend_label_id, friend_name, friend_label_mask

    def save_intermediate_result(self):
        with open("result/user2id.txt", "w", encoding="utf-8") as f:
            for k, v in self.user2id_dict.items():
                f.write(f"{k}, {v}\n")
        with open("result/label2id.txt", "w", encoding="utf-8") as f:
            for k, v in self.label_dict.items():
                f.write(f"{k}, {v}\n")

    def __getitem__(self, index):
        item = {
            "user_id": self.user_id[index],
            "profile_id": self.profile_id[index],
            "profile_mask": self.profile_mask[index],
            "profile_type": self.profile_type[index],
            "friend_label_id": self.friend_label_id[index],
            "friend_label_mask": self.friend_label_mask[index],
            "friend_name": self.friend_name[index]
        }
        if self.phase != "pred":
            item.update({
                "label": self.label[index]
            })
        return item

    def __len__(self):
        return self.user_id.shape[0]

def MyDataLoader(config, phase, shuffle=True):

    shuffle = True if phase == 'train' else False

    try:
        assert phase in ['train', 'test', 'pred']
    except:
        raise Exception("please choose phase word in ['train', 'test', 'pred']")
    
    if phase == 'train': data_path = config["train_path"]
    elif phase == 'test': data_path = config["test_path"]
    elif phase == "pred": data_path = config["pred_path"]

    dataset = MyDataset(phase, data_path, config["label_path"], text_max_length=config["text_max_length"], friend_max_length=config["friend_max_length"])
    loader = DataLoader(
        dataset=dataset,
        batch_size=config["bsz"],
        shuffle=shuffle
    )
    return loader, dataset.label_num, dataset.label_dict

if __name__=="__main__":
    dataloader = MyDataLoader(
        data_path="../data/data_test.csv",
        phase="train",
        bsz=5,
        max_length=5
    )
    print(next(iter(dataloader))) # DataLoader不支持下标括号访问