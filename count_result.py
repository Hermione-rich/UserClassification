import os


if __name__=="__main__":

    count_dir = {}

    dir_name = r"..\generate"
    file_list = os.listdir(dir_name)
    for file in file_list:
        with open(os.path.join(dir_name, file), "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.split("|")
                if count_dir.get(line[1]):
                    if line[1] == "12":
                        print(file)
                    count_dir[line[1]] += 1
                else:
                    count_dir[line[1]] = 1
    print(count_dir)
    print(count_dir['12'])