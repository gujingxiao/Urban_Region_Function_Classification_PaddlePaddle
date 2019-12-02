import random
import os
import sys

root_dir = '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/CityFunction/'

def train_val():
    path = root_dir + "train"
    dirs = sorted(os.listdir(path))
    files = {}
    for index, dir in enumerate(dirs):
        path_ = path + "/" + dir + "/"
        files[int(dir)] = []
        for file in os.listdir(path_):
            files[int(dir)].append(path_+file)
        sys.stdout.write('\r>> Loading data %d/%d'%(index+1, 9))
        sys.stdout.flush()
    sys.stdout.write("\n")

    lines = []
    for i in range(1, 10):
        file = files[i]
        for item in file:
            lines.append(item)

    random.shuffle(lines)
    random.shuffle(lines)
    print(len(lines))

    train_num = 360000
    val_num = 40000

    train_lines = lines[0:train_num]
    val_lines = lines[train_num : train_num + val_num]

    print(len(train_lines))
    print(len(val_lines))

    f = open(root_dir + 'train.txt', 'w+')
    for line in train_lines:
        f.writelines(line + '\n')
    f.close()

    f = open(root_dir + 'val.txt', 'w+')
    for line in val_lines:
        f.writelines(line + '\n')
    f.close()

def test():
    # 生成测试 list
    path = root_dir + "test"
    f = open(root_dir + "test.txt", "w+")
    for file in os.listdir(path):
        f.write(path + '/' + file + '\n')
    f.close()

if __name__ == '__main__':
    train_val()
    test()