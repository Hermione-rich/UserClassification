import os
import torch
import numpy as np
import pandas as pd
import random
import openpyxl
from loguru import logger
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def set_random_seed(seed_value=42):
    logger.info(f"Current seed is set to {seed_value}")
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)

def calculate_metric(real, pred):
    # 标准的分类评价指标为准确率Accuracy
    # measure_result = classification_report(y_true=real, y_pred=pred, digits=3, zero_division=0)
    # print(measure_result)
    acc = accuracy_score(y_true=real, y_pred=pred)
    logger.info(f"The accuracy is {round(acc*100, 2)}%")

def calculate_hit(real, logit, id2label, ks=[1, 5, 10]):
    hit_scores = list()
    for k in ks:
        hit_score = 0
        top_k_logits, top_k_indices = torch.topk(logit, k=k)
        top_k_indices = top_k_indices.detach().to('cpu').numpy().tolist()
        for idx in range(len(top_k_indices)):
            line = top_k_indices[idx]
            line = [id2label[i] for i in line]
            if real[idx] in line:
                hit_score += 1
        hit_scores.append(hit_score)
    test_num = len(top_k_indices)
    hit_scores = [hit_score / test_num for hit_score in hit_scores]
    for idx in range(len(ks)):
        logger.info(f"The hit@{ks[idx]} is {round(hit_scores[idx]*100, 2)}%")

def calculate_hit_threshold(real, logit, logit_torch, id2label, test_threshold, ks=[1, 5, 10]):
    hit_scores = list()
    for k in ks:
        test_num = 0
        hit_score = 0
        top_k_logits, top_k_indices = torch.topk(logit_torch, k=k)
        top_k_indices = top_k_indices.detach().to('cpu').numpy().tolist()
        for idx in range(len(top_k_indices)):
            line = top_k_indices[idx]
            line = [id2label[i] for i in line]
            if float(logit[idx]) > test_threshold: # 相比于calculate_hit，额外判断pred_logit需要大于阈值
                test_num += 1
                if real[idx] in line:
                    hit_score += 1
        hit_scores.append(hit_score)
    if test_num == 0:
        logger.info("The test_num is 0, please check your test_threshold")
    else:
        hit_scores = [hit_score / test_num for hit_score in hit_scores]
        for idx in range(len(ks)):
            if int(ks[idx]) == 1:
                logger.info(f"The hit@1 (准确率) is {round(hit_scores[idx]*100, 2)}%")
            logger.info(f"The hit@{ks[idx]} is {round(hit_scores[idx]*100, 2)}%")

def save_model(model, config):
    save_path = config["save_path"]
    torch.save(model.state_dict(), save_path)
    # torch.save(model, save_path) # 保存整个模型
    logger.info(f"The model has been saved in {save_path} successfully")

def save_test_case(test_path, pred, logit, real=None, phase="test", file_id="", threshold=None, threshold_pred_path="result/pred_res.txt"):
    '''
        phase = "test" or "pred"
    '''
    try:
        assert phase in ['test', 'pred']
    except:
        raise Exception("please choose phase word in ['test', 'pred']")
    
    df = pd.read_csv(test_path)
    txt_res = [] # 保存置信度高于threshold的结果进入txt
    if phase == "test":
        save_keys = ["id", "name", "education", "work", "living", "basic-info", "year-overviews", "friend_label", "friend_id", "label", "real", "pred", "logit"]
        save_res = {key:[] for key in save_keys}
        for idx, row in df.iterrows():
            save_res["id"].append(row["id"])
            save_res["name"].append(row["name"])
            save_res["education"].append(row["education"])
            save_res["work"].append(row["work"])
            save_res["living"].append(row["living"])
            save_res["basic-info"].append(row["basic-info"])
            save_res["year-overviews"].append(row["year-overviews"])
            save_res["friend_label"].append(row["friend_label"])
            save_res["friend_id"].append(row["friend_id"])
            save_res["label"].append(str(row["label"]))
            save_res["real"].append(str(real[idx]))
            save_res["pred"].append(str(pred[idx]))
            save_res["logit"].append(logit[idx])
    elif phase == "pred":
        save_keys = ["id", "name", "education", "work", "living", "basic-info", "year-overviews", "friend_label", "friend_id", "pred", "logit"]
        save_res = {key:[] for key in save_keys}
        for idx, row in df.iterrows():
            save_res["id"].append(row["id"])
            save_res["name"].append(row["name"])
            save_res["education"].append(row["education"])
            save_res["work"].append(row["work"])
            save_res["living"].append(row["living"])
            save_res["basic-info"].append(row["basic-info"])
            save_res["year-overviews"].append(row["year-overviews"])
            save_res["friend_label"].append(row["friend_label"])
            save_res["friend_id"].append(row["friend_id"])
            save_res["pred"].append(str(pred[idx]))
            save_res["logit"].append(logit[idx])
            if threshold and logit[idx] >= threshold:
                txt_res.append([row["id"], str(pred[idx]), logit[idx]])
    
    '''保存excel文件'''
    save_res_df = pd.DataFrame(save_res)
    if not os.path.exists("result"):
        os.mkdir("result")
    save_res_df.to_excel(f"result/test_case_{phase+file_id}.xlsx", index=False, header=save_res.keys())
    logger.info(f"The result file has been saved in result/test_case_{phase+file_id}.xlsx successfully")

    '''保存txt文件'''
    if threshold:
        logger.info(f"There are total {len(txt_res)} results bigger than threshold {threshold}")
        with open(threshold_pred_path, "a", encoding="utf-8") as f:
            for res in txt_res:
                f.write(f"{res[0]}|{res[1]}|{res[2]}\n")
        logger.info(f"The result file (txt) has been saved in {threshold_pred_path} successfully")