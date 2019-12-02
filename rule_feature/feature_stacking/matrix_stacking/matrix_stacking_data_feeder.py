import numpy as np
import paddle
import os
import random
from multiprocessing import cpu_count
# 数据读取代码

def train_mapper(sample):
    path, feature, label = sample

    return feature, label


def val_mapper(sample):
    path, feature, label = sample

    return feature, label


def infer_mapper(sample):
    path, feature, label = sample

    areaID = path.split('.')[0]
    return feature, label, areaID

def infer_data_reader(file_list, mapper, path_visit):
    def reader():
        file_list = os.listdir(path_visit)
        for filename in file_list:
            path = filename
            feature = np.load(path_visit + filename)
            feature = np.expand_dims(feature, axis=0)
            label = int(filename.split('.')[0].split("_")[-1]) - 1
            yield path, feature, label

    return paddle.reader.xmap_readers(mapper, reader, cpu_count(), 1024)

def data_reader(file_list, mapper, path_visit):
    def reader():
        random.shuffle(file_list)
        for filename in file_list:
            path = filename
            feature = np.load(path_visit + filename)
            feature = np.expand_dims(feature, axis=0)
            label = int(filename.split('.')[0].split("_")[-1]) - 1
            yield path, feature, label

    return paddle.reader.xmap_readers(mapper, reader, cpu_count(), 1024)