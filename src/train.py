import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from loguru import logger

from .data import MyDataLoader
from .utils import *

@torch.no_grad()
def test(model, device, config, hitk):
    model.eval()
    testloader, label_num, label_dict = MyDataLoader(config, phase="test")
    id2label = {v: k for k, v in label_dict.items()}
    total_real = list()
    total_pred = list()
    total_logit = list()
    total_logit_torch = list()
    node_feature_np = pickle.load(open(config["feature_path"],'rb'))
    node_features = nn.Embedding.from_pretrained(torch.from_numpy(node_feature_np).float())
    for _, batch in enumerate(testloader):
        # batch["friend_name"] = node_features(batch["friend_name"]) # 根据friend的nameid得到其profile特征向量
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        logit = model(batch)
        logit = F.softmax(logit, dim=1)
        pred = logit.argmax(dim=1)
        pred_logit = torch.max(logit, dim=1)[0]
        real = batch["label"]
        total_real.extend(real.detach().to('cpu').numpy().tolist())
        total_pred.extend(pred.detach().to('cpu').numpy().tolist())
        total_logit.extend(pred_logit.detach().to('cpu').numpy().tolist())
        total_logit_torch.append(logit)
    total_real = [id2label[i] for i in total_real]
    total_pred = [id2label[i] for i in total_pred]
    total_logit_torch = torch.cat(total_logit_torch, dim=0)

    calculate_metric(real=total_real, pred=total_pred) # accuracy
    calculate_hit(total_real, total_logit_torch, id2label, ks=hitk.split(",")) # hit@k
    calculate_hit_threshold(total_real, total_logit, total_logit_torch, id2label, test_threshold=0.6, ks=hitk.split(",")) # hit@k after threshold
    save_test_case(config["test_path"], total_pred, total_logit, total_real, phase="test")


def train(model, device, config, writer):
    dataloader, label_num, label_dict = MyDataLoader(config, phase="train")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    node_feature_np = pickle.load(open(config["feature_path"],'rb'))
    node_features = nn.Embedding.from_pretrained(torch.from_numpy(node_feature_np).float())
    has_write_graph = False
    for e in range(config["epoch"]):
        model.train()
        total_loss = 0
        for _, batch in enumerate(dataloader):
            optimizer.zero_grad()
            # batch["friend_name"] = node_features(batch["friend_name"]) # 根据friend的nameid得到其profile特征向量
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            logit = model(batch)
            loss = criterion(logit, batch["label"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"In epoch {e+1}, the total loss is {round(total_loss, 5)}")
        writer.add_scalar(tag="loss", scalar_value=total_loss, global_step=e)
        if not has_write_graph:
            writer.add_graph(model, batch)
            has_write_graph = True

if __name__=="__main__":
    print("this is the model training and testing")