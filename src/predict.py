import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .model import MyModel
from .data import MyDataLoader
from .utils import *

@logger.catch
@torch.no_grad()
def predict(model, device, config, file_id="", threshold=None, threshold_pred_path="result/pred_res.txt"):
    model.eval()
    testloader, label_num, label_dict = MyDataLoader(config, phase="pred")
    id2label = {v: k for k, v in label_dict.items()}
    total_pred = list()
    total_logit = list()
    
    for _, batch in enumerate(testloader):
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        logit = model(batch)
        logit = F.softmax(logit, dim=1)
        pred = logit.argmax(dim=1)
        pred_logit = torch.max(logit, dim=1)[0]
        total_pred.extend(pred.detach().to('cpu').numpy().tolist())
        total_logit.extend(pred_logit.detach().to('cpu').numpy().tolist())
    total_pred = [id2label[i] for i in total_pred]
    save_test_case(config["pred_path"], total_pred, total_logit, phase="pred", file_id=file_id, threshold=threshold, threshold_pred_path=threshold_pred_path)
    