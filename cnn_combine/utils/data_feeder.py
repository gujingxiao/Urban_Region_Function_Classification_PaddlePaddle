import numpy as np
from numpy import random
import cv2
from imgaug import augmenters as iaa
import paddle
from multiprocessing import cpu_count
# 数据读取代码

train_visit_max = 3160.
test_visit_max = 4779.


def augumentor(image):
    augment_img = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.SomeOf((0,4),[
            iaa.Affine(rotate=90),
            iaa.Affine(rotate=180),
            iaa.Affine(rotate=270),
            iaa.Affine(shear=(-16, 16)),
        ]),
        iaa.OneOf([
                iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
            ]),
        #iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
        ], random_order=True)

    image_aug = augment_img.augment_image(image)
    return image_aug

def train_mapper(sample):
    img_path, visit, label = sample

    img = cv2.imread(img_path)
    img = augumentor(img)

    # 把图片转换成numpy值
    img = np.array(img).astype(np.float32)
    # 转换成CHW
    img = img.transpose((2, 0, 1))
    # 转换成BGR
    img = img / 255.0

    visit /= train_visit_max
    return img, visit, label


def val_mapper(sample):
    img_path, visit, label = sample

    img = cv2.imread(img_path)
    img = np.array(img).astype(np.float32)
    img = img.transpose((2, 0, 1))
    img = img / 255.0

    visit /= train_visit_max
    return img, visit, label


def infer_mapper(sample):
    img_path, visit, label = sample

    img = cv2.imread(img_path)
    img = np.array(img).astype(np.float32)
    img = img.transpose((2, 0, 1))
    img = img / 255.0

    areaID = img_path.split('/')[-1].split('.')[0].split('_')[0]
    visit /= test_visit_max
    return img, visit, label, areaID

def infer_data_reader(file_list, mapper, path_visit):
    def reader():
        with open(file_list, 'r') as f:
            lines = f.read().splitlines()
            indexes = np.arange(0, len(lines))
            for i in indexes:
                line = lines[i]
                image = line
                visit = np.load(path_visit + line.split('/')[-1].split('.')[0] + ".npy")
                label = int(line.split("/")[-1].split("_")[-1].split(".")[0]) - 1
                yield image, visit, label

    return paddle.reader.xmap_readers(mapper, reader, cpu_count(), 1024)

def data_reader(file_list, mapper, path_visit):
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