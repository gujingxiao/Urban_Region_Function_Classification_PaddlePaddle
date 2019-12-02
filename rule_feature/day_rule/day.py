import numpy as np
import pandas as pd
import os
import math
from time import time

root_dir = '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/CityFunction/'
base_folder = 'day_rule'

#定义几个函数
# def find_key(keys,obj):  二分查找user id为obj的用户是否在训练集用户列表keys中出现
# def softmax(vec): 9类别的时长转softmax概率，时长太大，指数溢出暂时不用
# def cnt2pro(vec): 9类别的时长转概率，分别除以9类别时长总和
# def close_txt(id_cnt_files): 一次性关闭多个txt文件，id_cnt_files为文件列表
# def getfiles(mode): 返回文件名列表，mode："train", "val", "test"

def find_key(keys,obj):
    len_keys=len(keys)
    l=0
    r=len_keys-1
    while True:
        if r-l<3:
            break
        mid=(l+r)//2
        if keys[mid]==obj:
            return True
        if keys[mid]<obj:
            l=mid+1
        else:
            r=mid+1
    for k in range(l,r+1):
        if keys[k]==obj:
            return True
    return False
def softmax(vec):
    return_ans=np.zeros([vec.shape[0]])
    for k in range(vec.shape[0]):
        SUM=0.0
        for n in range(vec.shape[0]):
            SUM+=math.exp(vec[n]-vec[k])
        return_ans=1.0/SUM
    return return_ans
def cnt2pro(vec):
    return_ans=np.zeros([vec.shape[0]])
    SUM=0.0
    for n in range(vec.shape[0]):
        SUM+=vec[n]
    for k in range(vec.shape[0]):
        return_ans[k]=vec[k]#/SUM
    return return_ans
def close_txt(id_cnt_files):
    for file in id_cnt_files:
        file.close()
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


# 对训练集进行处理的代码
# 遍历训练集的原始visit txt文件，分9个类别记录每个用户的总时长
# 输出9个txt，每个文件对应一个类别，每个文件中两列记录用户id和时长
# 分两步进行：第一步：遍历txt，第二步：对于每个类别，依据时长对用户排序，删除时长低的80% 或 90% 的用户，再合并同一类别中的相同用户，时长相加
def train_1():
    ###################################################第一步
    id_cnt_files=[]
    for i in range(1,10):
        id_cnt_file=open(root_dir + "ruleFeature/{}/id_cnt/id_cnt_".format(base_folder) + str(i)+".txt","a+")
        id_cnt_files.append(id_cnt_file)
    files=getfiles("train")
    all_cnt=len(files)
    print(all_cnt)
    cnt=0
    last_time=time()
    for file in files:
        label=int(file.split(".txt")[0].split("/")[-1].split("_")[1])
        table = pd.read_csv(file, header=None, sep='\t')
        tv=table.values
        user_num=tv.shape[0]
        temp=""
        for i in range(user_num):
            temp+=str(tv[i,0])+","+str(tv[i,1].count("&")) +"\n"
        id_cnt_files[label-1].write(temp)
        if cnt%1000==0:
            now_time=time()
            print("%.4f%%"%(cnt/all_cnt*100),now_time-last_time)
        cnt+=1
    close_txt(id_cnt_files)


###################################################第二步
def train_2():
    id_cnt_files = []
    for i in range(1, 10):
        id_cnt_file = open(root_dir + "ruleFeature/{}/id_cnt_del_same/id_cnt_".format(base_folder) + str(i) + ".txt", "a+")
        id_cnt_files.append(id_cnt_file)

    last_time = time()
    for i in range(1, 10):
        table = pd.read_table(root_dir + "ruleFeature/{}/id_cnt/id_cnt_".format(base_folder) + str(i) + ".txt", sep=",", header=None)
        temp = table.values[:, :]
        user_num = temp.shape[0]
        sorted_data = temp[np.lexsort(temp.T)]
        rate = 0.8
        first_user = int(user_num * rate)
        user2cnt = {}
        last_user = ""
        for j in range(first_user, user_num):
            if (j - first_user) % 10000 == 0:
                now_time = time()
                print(i, (j - first_user) / (user_num - first_user), now_time - last_time)
            now_user = sorted_data[j, 0]
            if last_user == now_user:
                user2cnt[now_user] += sorted_data[j, 1]
            else:
                user2cnt[now_user] = sorted_data[j, 1]
                last_user = now_user
        users = user2cnt.keys()
        for user in users:
            id_cnt_files[i - 1].write(str(user) + "," + str(user2cnt[user]) + "\n")
    close_txt(id_cnt_files)


def test():
    # 测试代码
    user2hours=[{} for i in range(9)]
    for i in range(1,10):
        label=i
        table = pd.read_table(root_dir + "ruleFeature/{}/id_cnt_del_same/id_cnt_".format(base_folder)+str(i)+".txt",sep=",", header=None)
        temp=table.values[:,:]
        user_num=temp.shape[0]
        sorted_data=temp[np.lexsort(temp.T)]
        for j in range(user_num):
            if (j)%50000==0:
                print(i,(j)/(user_num))
            now_user=sorted_data[j,0]
            user2hours[i-1][now_user]=int(sorted_data[j,1])
    ################################################### 根据user2hours记录老用户的时长，
    # 因为字典操作需要保证key值存在，所以需要二分搜索key (用户id)是否是在字典的keys中出现过

    id2label=np.zeros([100000])
    id2pro=np.zeros([100000,9])
    id2num=np.zeros([100000,9])
    all_keys=[]  #记录所有训练集用户id，用于后续二分查询
    for i in range(9):
        temp=list(user2hours[i].keys())
        temp.sort()
        all_keys.append(temp[:])
    files=getfiles("test")
    all_cnt=len(files)
    print(all_cnt)
    cnt=0
    last_time=time()
    if not os.path.exists(root_dir + "ruleFeature/{}/test_npy/".format(base_folder)):
        os.makedirs(root_dir + "ruleFeature/{}/test_npy/".format(base_folder))

    for file in files:
        reg_id=int(file.split(".txt")[0].split("/")[-1])
        table = pd.read_table(file, header=None)
        table2cnt=table.values[:,:]
        user_num=table2cnt.shape[0]
        for i in range(user_num):
            now_user_hours=table2cnt[i,1].count("&")
            table2cnt[i,1]=now_user_hours
        temp=table2cnt[:,:]
        sorted_data=temp[np.lexsort(temp.T)]
        rate=0.8
        first_user=int(user_num*rate)
        reg_per_label=np.zeros([9])
        for i in range(first_user,user_num):
            now_user=str(sorted_data[i,0])
            hours_per_label=np.zeros([9])
            for j in range(9):
                if find_key(all_keys[j],now_user):
                    hours_per_label[j]=user2hours[j][now_user]
            reg_per_label+=hours_per_label
        id2num[reg_id]=reg_per_label
        pro=cnt2pro(reg_per_label)
        pre_label=np.argmax(pro)+1
        if np.sum(reg_per_label)==0:  #该测试样本内老用户为0，用类别 -1 标记缺失
            pre_label=-1
        id2label[reg_id]=pre_label
        id2pro[reg_id]=pro[:]
        if cnt%1000==0:
            print(pro)
            now_time=time()
            print("%.4f%%"%(cnt/all_cnt*100),now_time-last_time)
        cnt+=1

        npy_name = file.split('/')[-1].split(".")[0] + ".npy"
        np.save(root_dir + "ruleFeature/{}/test_npy/".format(base_folder) + npy_name, pro)

def val():
    # 验证集代码
    ##################################################从id_cnt_del_same文件夹读取训练集的用户时长，存入9个字典中 user2hours
    user2hours = [{} for i in range(9)]
    for i in range(1, 10):
        label = i
        table = pd.read_table(
            root_dir + "ruleFeature/{}/id_cnt_del_same/id_cnt_".format(base_folder) + str(i) + ".txt",
            sep=",", header=None)
        temp = table.values[:, :]
        user_num = temp.shape[0]
        sorted_data = temp[np.lexsort(temp.T)]
        for j in range(user_num):
            if (j) % 50000 == 0:
                print(i, (j) / (user_num))
            now_user = sorted_data[j, 0]
            user2hours[i - 1][now_user] = int(sorted_data[j, 1])
    ################################################### 根据user2hours记录老用户的时长，
    # 因为字典操作需要保证key值存在，所以需要二分搜索key (用户id)是否是在字典的keys中出现过
    id2label = {}
    id2pro = {}
    id2num = {}
    all_keys = []  # 记录所有训练集用户id，用于后续二分查询
    for i in range(9):
        temp = list(user2hours[i].keys())
        temp.sort()
        all_keys.append(temp[:])
    files = getfiles("val")
    all_cnt = len(files)
    print(all_cnt)
    cnt = 0
    acc_cnt = 0
    last_time = time()
    if not os.path.exists(root_dir + "ruleFeature/{}/val_npy/".format(base_folder)):
        os.makedirs(root_dir + "ruleFeature/{}/val_npy/".format(base_folder))

    for file in files:
        s = file.split(".txt")[0].split("/")[-1].split("_")
        reg_id = s[0]
        true_label = int(s[1])
        table = pd.read_table(file, header=None)
        table2cnt = table.values[:, :]
        user_num = table2cnt.shape[0]
        for i in range(user_num):
            now_user_hours = table2cnt[i, 1].count("&")
            table2cnt[i, 1] = now_user_hours
        temp = table2cnt[:, :]
        sorted_data = temp[np.lexsort(temp.T)]
        rate = 0.8  # 剔除单个测试样本中时长较短的用户，剔除比例为rate
        first_user = int(user_num * rate)
        reg_per_label = np.zeros([9])
        for i in range(first_user, user_num):
            now_user = str(sorted_data[i, 0])
            hours_per_label = np.zeros([9])
            for j in range(9):
                if find_key(all_keys[j], now_user):
                    hours_per_label[j] = user2hours[j][now_user]
            reg_per_label += hours_per_label
        id2num[reg_id] = reg_per_label
        pro = cnt2pro(reg_per_label)
        pre_label = np.argmax(pro) + 1
        if np.sum(reg_per_label) == 0:  # 该测试样本内老用户为0，用类别 -1 标记缺失
            pre_label = -1
        id2label[reg_id] = pre_label
        id2pro[reg_id] = pro[:]
        if true_label == pre_label:
            acc_cnt += 1
        cnt += 1
        if cnt % 1000 == 0:
            now_time = time()
            print("%.4f%%" % (cnt / all_cnt * 100), "val acc ", acc_cnt / cnt, now_time - last_time)

        npy_name = file.split('/')[-1].split(".")[0] + ".npy"
        np.save(root_dir + "ruleFeature/{}/val_npy/".format(base_folder) + npy_name, pro)

def submission():
    npy_dir = root_dir + "ruleFeature/{}/test_npy/".format(base_folder)
    f = open('daySubmit.txt', 'w+')
    for item in os.listdir(npy_dir):
        npy_file = os.path.join(npy_dir, item)
        npy = np.load(npy_file)
        label = np.argmax(npy) + 1
        f.write('%s\t%03d\n' % (item.split('.')[0], label))

if __name__ == '__main__':
    # train_1()  # 读取训练集，获得老用户及其label
    # train_2()  # 处理老用户，进行一些过滤
    # val()
    test()  # 处理测试集，获得测试集中老用户的label
    # submission()
