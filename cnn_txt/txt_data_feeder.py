import numpy as np
import paddle
from multiprocessing import cpu_count
# 数据读取代码

train_visit_max = 3160.
test_visit_max = 4779.

def txt_train_mapper(sample):
    img_path, visit, label = sample
    visit = visit / train_visit_max
    return visit, label


def txt_val_mapper(sample):
    img_path, visit, label = sample
    visit = visit / train_visit_max
    return visit, label

def txt_infer_mapper(sample):
    img_path, visit, label = sample
    areaID = img_path.split('/')[-1].split('.')[0].split('_')[0]
    visit = visit / test_visit_max
    return visit, label, areaID

def txt_data_reader(file_list, mapper, path_visit):
    def reader():
        with open(file_list, 'r') as f:
            lines = f.read().splitlines()
            indexes = np.arange(0, len(lines))
            np.random.shuffle(indexes)
            for i in indexes:
                line = lines[i]
                image = line
                visit = np.load(path_visit + line.split('/')[-1].split('.')[0] + ".npy")
                label = int(line.split("/")[-1].split("_")[-1].split(".")[0]) - 1
                yield image, visit, label

    return paddle.reader.xmap_readers(mapper, reader, cpu_count(), 1024)