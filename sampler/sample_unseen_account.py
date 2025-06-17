import sys
import random
import numpy as np

def my_softmax(input_list):
    output = np.exp(input_list) / np.sum(np.exp(input_list))
    return output

if __name__=="__main__":
    N = int(sys.argv[1])

    # 保存已经采样/拓展的用户
    exist_user = list()
    with open("sampled_new_user_20231108.txt", "r", encoding="utf-8") as f:
        for line in f.readlines():
            exist_user.append(line.strip())

    link_account_dict = dict()
    with open("../data/phase1_data/friend.txt", "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            line = line.split("|")

            # 如果已经拓展过，则跳过
            if line[1] in exist_user:
                continue;

            if link_account_dict.get(line[1]):
                link_account_dict[line[1]] += 1
            else:
                link_account_dict[line[1]] = 1
    link_account_set = list(link_account_dict.keys())
    link_account_prob = my_softmax(list(link_account_dict.values()))
    output = np.random.choice(link_account_set, size=N, p=link_account_prob, replace=False)
    output = output.tolist()
    with open("sampled_new_user.txt", "w", encoding="utf-8") as f:
        for u in output:
            f.write(f"{u}\n")