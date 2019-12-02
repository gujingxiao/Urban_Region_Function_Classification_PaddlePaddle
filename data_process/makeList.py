from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
from data_process.configData import config

# 生成训练 list
import random

# 读取训练集图片
path = config.main_data_path + "train"
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

# 划分训练/验证集
f = open(config.val_table_path, "w+")
valid_data = {}
train_data = {}
for i in range(1, 10):
    valid_data[i] = random.sample(files[i], 1000)
    train_data[i] = list(set(files[i]) - set(valid_data[i]))
    for item in valid_data[i]:
        f.write(item+"\n")
f.close()

f = open(config.train_table_path, "w+")
for i in range(1, 10):
    for item in train_data[i]:
        f.write(item+"\n")
f.close()

# 使训练集不同类别样本数相同
file_num_ = []
for i in range(1, 10):
    file_num_.append(len(train_data[i]))
max_amount = max(file_num_)
for i in range(1, 10):
    for j in range(max_amount-len(train_data[i])):
        train_data[i].append(train_data[i][random.randint(0, file_num_[i-1]-1)])

# 生成测试 list
path = config.main_data_path + "test"
f = open(config.test_table_path, "w+")
for file in os.listdir(path):
    f.write(path+'/'+file+'\n')
f.close()
