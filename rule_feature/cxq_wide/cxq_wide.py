import pandas as pd
import numpy as np
import os
from time import time

root_dir = '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/CityFunction/'
base_folder = 'cxq_wide'

def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files

def getfiles(mode):
    def get_name_from_txt(file_name):
        name_id = []
        f = open(file_name)
        lines = f.readlines()
        for line in lines:
            s = line.split(".jpg")[0]
            s = s.split("/")[-1]
            name_id.append(s)
        f.close()
        return name_id

    return_ans = []
    if mode == "test":
        path = root_dir + "test_visit/"
        files = os.listdir(path)
        for file in files:
            return_ans.append(path + file)
        return_ans.sort()
        return return_ans
    elif mode == "train":
        file_name = root_dir + "train.txt"
    elif mode == "val":
        file_name = root_dir + "val.txt"
    name_id = get_name_from_txt(file_name)
    name2file = {}
    path = root_dir + "train_visit/"
    files = os.listdir(path)
    for file in files:
        s = file.split(".txt")[0]
        name2file[s] = path + file
    return_ans = []
    for name in name_id:
        return_ans.append(name2file[name])
    return return_ans


def find_key(keys, obj):
    len_keys = len(keys)
    l = 0
    r = len_keys - 1
    while True:
        if r - l < 3:
            break
        mid = (l + r) // 2
        if keys[mid] == obj:
            return True
        if keys[mid] < obj:
            l = mid + 1
        else:
            r = mid + 1
    for k in range(l, r + 1):
        if keys[k] == obj:
            return True
    return False

# Read all train files
# 350000 files - need 2750s
def read_train():
    #    读取训练集
    tr_files = getfiles("train")
    print('Rule: {}, Processing Train Files.'.format(base_folder))
    time1 = time()
    out = open(root_dir + 'ruleFeature/{}/train_userid_label.csv'.format(base_folder), 'w')
    for i, file in enumerate(tr_files):
        if i % 1000 == 0:
            time2 = time()
            print('{} / {}, Total Time: {}'.format(i, len(tr_files), time2 - time1))
        area_label = int(file.split("/")[-1].split('_')[1].split('.txt')[0])
        df = pd.read_csv(file, sep='\t', names=['user_id', 'info'])
        df['day_count'] = df['info'].apply(lambda x: len(x.split(',')))

        user_id = list(df['user_id'])
        day_count = list(df['day_count'])

        for index in range(len(user_id)):
            out.write(str(user_id[index]) + ',' + str(area_label) + ',' + str(day_count[index]) + '\n')
    out.close()

# Deal with all train files
# Need 2303s
# 50.0GB Memory Use
def deal_train():
    print('Rule: {}, Dealing with Train Files.'.format(base_folder))
    time1 = time()
    df = pd.read_csv(root_dir + 'ruleFeature/{}/train_userid_label.csv'.format(base_folder), names=['user_id', 'label', 'day_count'])
    # 只保留在某个类别呆的天数超过一半的用户
    print(len(df))
    df = df.groupby(['user_id', 'label'])['day_count'].sum().reset_index()
    print(len(df))
    sum_df = df.groupby('user_id')['day_count'].sum().reset_index()
    max_df = df.groupby('user_id')['day_count'].max().reset_index()
    merge_df = pd.merge(max_df, sum_df, on='user_id', how='left')
    user_df = merge_df[merge_df['day_count' + '_x'] > merge_df['day_count' + '_y'] * 0.5][['user_id', 'day_count' + '_x']]
    user_df = user_df.rename(columns={'day_count' + '_x': 'day_count'})
    tr_df = pd.merge(user_df, df, on=['user_id', 'day_count'], how='left')
    tr_df = tr_df[['user_id', 'label']]
    print(len(tr_df))
    tr_df.to_csv(root_dir + 'ruleFeature/{}/tr_df.csv'.format(base_folder), index=False, header=False)
    time2 = time()
    print('Total Time: {}'.format(time2 - time1))

def deal_test():
    tr_df = pd.read_csv(root_dir + 'ruleFeature/{}/tr_df.csv'.format(base_folder), names=['user_id', 'label'])
    #    读取测试集
    user_label = {'user_id': [], 'label': []}
    ts_files = getfiles("test")
    print(len(ts_files))
    time1 = time()
    for i, file in enumerate(ts_files):
        if i % 1000 == 0:
            time2 = time()
            print(i, time2 - time1)
            time1 = time2
        df = pd.read_csv(file, sep='\t', names=['user_id', 'info'])
        user_label['user_id'].extend(list(df['user_id']))
    ts_df = pd.DataFrame(user_label, columns=['user_id'])

    # 规则
    ts_user_label = pd.merge(ts_df, tr_df, on='user_id', how='left')
    ts_user_label = ts_user_label[~ts_user_label['label'].isnull()].groupby('user_id')['label'].mean().reset_index()
    ts_user_label.to_csv(root_dir + 'ruleFeature/{}/test_user_label.csv'.format(base_folder), index=False, header=False)

def all_test():
    # 测试集submission生成
    ts_user_label = pd.read_csv(root_dir + 'ruleFeature/{}/test_user_label.csv'.format(base_folder), names=['user_id', 'label'])
    user_label = ts_user_label
    set2 = set(user_label['user_id'])
    ul = user_label['user_id']
    test_files = getfiles("test")
    test_files.sort()
    print(len(test_files))
    start = time()
    if not os.path.exists(root_dir + "ruleFeature/{}/test_npy/".format(base_folder)):
        os.makedirs(root_dir + "ruleFeature/{}/test_npy/".format(base_folder))
    for i, file in enumerate(test_files):
        if i % 100 == 0:
            end = time()
            print('num: ' + str(i) + ' / ' + str(len(test_files)) + '  time:' + str(end - start))
        df = pd.read_csv(file, sep='\t', names=['user_id', 'info'])
        inter = set(df['user_id']).intersection(set2)

        temp = user_label[ul.isin(inter)].groupby('label')['user_id'].count().reset_index()
        label_cnt = np.zeros([9])
        for j in range(9):
            if len(temp[temp['label'] == j + 1]) == 0:
                continue
            label_cnt[j] += int(temp[temp['label'] == j + 1]['user_id'])  # 此处的user_id是类别为j+1的个数

        npy_name = file.split('/')[-1].split(".")[0] + ".npy"
        np.save(root_dir + "ruleFeature/{}/test_npy/".format(base_folder)+npy_name, label_cnt)

def deal_val():
    tr_df = pd.read_csv(root_dir + 'ruleFeature/{}/tr_df.csv'.format(base_folder), names=['user_id', 'label'])
    # 读取验证集
    user_label = {'user_id': [], 'label': []}
    val_files = getfiles("val")
    val_files.sort()
    print(len(val_files))
    time1 = time()
    for i, file in enumerate(val_files):
        if i % 1000 == 0:
            time2 = time()
            print(i, time2 - time1)
        df = pd.read_csv(file, sep='\t', names=['user_id', 'info'])
        user_label['user_id'].extend(list(df['user_id']))
    val_df = pd.DataFrame(user_label, columns=['user_id'])

    # 规则
    val_user_label = pd.merge(val_df, tr_df, on='user_id', how='left')
    val_user_label = val_user_label[~val_user_label['label'].isnull()].groupby('user_id')['label'].mean().reset_index()
    val_user_label.to_csv(root_dir + 'ruleFeature/{}/val_user_label.csv'.format(base_folder), index=False, header=False)

def all_val():
    val_user_label = pd.read_csv(root_dir + 'ruleFeature/{}/val_user_label.csv'.format(base_folder), names=['user_id', 'label'])
    # threshold = 100
    user_label = val_user_label
    cnt = 0
    set2 = set(user_label['user_id'])
    ul = user_label['user_id']
    val_files = getfiles("val")
    val_files.sort()
    print(len(val_files))
    start = time()
    if not os.path.exists(root_dir + "ruleFeature/{}/val_npy/".format(base_folder)):
        os.makedirs(root_dir + "ruleFeature/{}/val_npy/".format(base_folder))
    for i, file in enumerate(val_files):
        if i % 100 == 0:
            end = time()
            print('num:' + str(i) + '  time:' + str(end - start), "val acc ", cnt / (i + 1))
        true_label = int(file.split('.txt')[0].split("/")[-1].split("_")[1])
        df = pd.read_csv(file, sep='\t', names=['user_id', 'info'])
        inter = set(df['user_id']).intersection(set2)

        temp = user_label[ul.isin(inter)].groupby('label')['user_id'].count().reset_index()
        label_cnt = np.zeros([9])
        for j in range(9):
            if len(temp[temp['label'] == j + 1]) == 0:
                continue
            label_cnt[j] += int(temp[temp['label'] == j + 1]['user_id'])  # 此处的user_id是类别为j+1的个数
        label = np.argmax(label_cnt) + 1
        if true_label == label:
            cnt += 1

        npy_name = file.split('/')[-1].split(".")[0] + ".npy"
        np.save(root_dir + "ruleFeature/{}/val_npy/".format(base_folder) + npy_name, label_cnt)

def submission():
    npy_dir = root_dir + "ruleFeature/{}/test_npy/".format(base_folder)
    f = open('{}Submit.txt'.format(base_folder), 'w+')
    for item in os.listdir(npy_dir):
        npy_file = os.path.join(npy_dir, item)
        npy = np.load(npy_file)
        label = np.argmax(npy) + 1
        f.write('%s\t%03d\n' % (item.split('.')[0], label))

if __name__ == '__main__':
    # read_train()  # 读取训练集，获得老用户及其label
    # deal_train()  # 处理老用户，进行一些过滤
    #deal_val()
    #deal_test()
    #all_val()
    # all_test()
    submission()