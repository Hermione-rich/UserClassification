import os
import pandas as pd

def load_warship_to_base(file_path):
    warship_to_base = dict()
    df = pd.read_excel(file_path).fillna("")
    for _, row in df.iterrows():
        # 保存舰船所在地的简写地点
        if row["naval_base_abbr"] != "":
            warship_to_base[str(row["id"])] = row["naval_base_abbr"]
    return warship_to_base


if __name__=="__main__":

    dir_name = "round_4/"
    file_list = os.listdir(dir_name)
    # 加载warship id到naval base的映射字典
    warship_to_base = load_warship_to_base(file_path="data/fb_warship_info.xlsx")

    # 逻辑规则的后处理步骤
    save_res = {k: [] for k in ["id", "name", "education", "work", "living", "basic-info", "year-overviews", "friend_label", "friend_id", "pred", "logit"]}
    for file in file_list:
        df = pd.read_excel(os.path.join(dir_name, file)).fillna("")
        for _, row in df.iterrows():
            if float(row["logit"]) < 0.8: continue;
            if not warship_to_base.get(str(row["pred"])): continue;
            for place in warship_to_base[str(row["pred"])]:
                if (place in row["work"] or place in row["living"]) and row["id"] not in save_res["id"]:
                    save_res["id"].append(row["id"])
                    save_res["name"].append(row["name"])
                    save_res["education"].append(row["education"])
                    save_res["work"].append(row["work"])
                    save_res["living"].append(row["living"])
                    save_res["basic-info"].append(row["basic-info"])
                    save_res["year-overviews"].append(row["year-overviews"])
                    save_res["friend_label"].append(row["friend_label"])
                    save_res["friend_id"].append(row["friend_id"])
                    save_res["pred"].append(row["pred"])
                    save_res["logit"].append(row["logit"])
                    break;
    save_res_df = pd.DataFrame(save_res)
    save_res_df.to_excel(os.path.join(dir_name, "final_result.xlsx"), index=False, header=save_res.keys())