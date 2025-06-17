import os
import sys
import math
import numpy as np
import pandas as pd

def split_train_test(data, test_ratio):

    #设置随机数种子，保证每次生成的结果都是一样的
    np.random.seed(42)

    #permutation随机生成0-len(data)随机序列
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = math.floor(int(len(data)) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    return data.iloc[train_indices],data.iloc[test_indices]


'''
训练测试数据划分函数：
运行脚本示例：python split_data.py 0.3    # 注释：0.3为测试数据划分比例
在运行完split_data.py文件后，目标数据目录下会生成如下文件：
1. train_data.csv
2. test_data.csv
'''
if __name__=="__main__":
    data_path = "../data/"
    training_dir = "phase3_data"
    try:
        test_ratio = float(sys.argv[1])
        assert test_ratio < 1 and test_ratio >= 0
    except:
        raise Exception("传递的test_ratio参数必须大于等于0且小于1！")

    path = os.path.join(data_path, training_dir, "target_account_processed.csv")
    data = pd.read_csv(path, dtype=str, encoding="utf-8")
    train_data, test_data = split_train_test(data, test_ratio=test_ratio)
    train_df, test_df = pd.DataFrame(train_data), pd.DataFrame(test_data)
    train_df.to_csv(os.path.join(data_path, training_dir, "train_data.csv"), index=False, header=test_data.keys())
    test_df.to_csv(os.path.join(data_path, training_dir, "test_data.csv"), index=False, header=test_data.keys())