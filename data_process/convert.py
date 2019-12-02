import time
import numpy as np
import sys
import pandas as pd
from data_process.configData import config
from data_process.feature import visit2array # config #visit2array
from multiprocessing import Process

main_data_path = config.main_data_path

train_table_path = config.train_table_path  # main_data_path + 'train.txt'
train_main_visit_path = config.train_main_visit_path  # main_data_path + "train_visit/train/"

val_table_path = config.val_table_path  # main_data_path + 'train.txt'
val_main_visit_path = config.train_main_visit_path  # main_data_path + "train_visit/train/"

test_table_path = config.test_table_path  # main_data_path + 'test.txt'
test_main_visit_path = config.test_main_visit_path  # main_data_path + "test_visit/test/"

train_num = 9000
test_num = 100000
file_num_each_job_train = 1500
file_num_each_job_test = 20000
workers_train = int(train_num/file_num_each_job_train)
workers_test = int(test_num/file_num_each_job_test)

def visit2array_train(num):
    table = pd.read_csv(train_table_path, header=None)
    filenames = [a[0].split("/")[-1].split('.')[0] for a in table.values]
    length = len(filenames)
    start_time = time.time()
    total_users = []
    cnt_users = 0
    for index, filename in enumerate(filenames[num*file_num_each_job_train: (num+1) * file_num_each_job_train]):
        table = pd.read_csv(train_main_visit_path + filename + ".txt", header=None, sep='\t')
        init_cishu = visit2array(table)
        users = table[0]

        np.save(config.train_npy_visit_path + filename + '.npy', init_cishu)

        cnt_users += len(users)
        total_users += list(users.values)
        sys.stdout.write('\r>> Processing train visit data %d/%d, Time %.2fs' % (index + 1, length, time.time() - start_time))
        sys.stdout.flush()

    sys.stdout.write('\n')
    print('train users:', cnt_users)
    print("using time:%.2fs" % (time.time() - start_time))
    return total_users

def visit2array_val(num):
    table = pd.read_csv(val_table_path, header=None)
    filenames = [a[0].split("/")[-1].split('.')[0] for a in table.values]
    length = len(filenames)
    start_time = time.time()
    total_users = []
    cnt_users = 0
    for index, filename in enumerate(filenames[num*file_num_each_job_train: (num+1) * file_num_each_job_train]):
        table = pd.read_csv(val_main_visit_path + filename + ".txt", header=None, sep='\t')
        init_cishu = visit2array(table)
        users = table[0]

        np.save(config.train_npy_visit_path + filename + '.npy', init_cishu)

        cnt_users += len(users)
        total_users += list(users.values)
        sys.stdout.write('\r>> Processing val visit data %d/%d, Time %.2fs' % (index + 1, length, time.time() - start_time))
        sys.stdout.flush()

    sys.stdout.write('\n')
    print('train users:', cnt_users)
    print("using time:%.2fs" % (time.time() - start_time))
    return total_users

def visit2array_test(num):
    table = pd.read_csv(test_table_path, header=None)
    filenames = [a[0].split("/")[-1].split('.')[0] for a in table.values]
    length = len(filenames)
    start_time = time.time()
    total_users = []
    cnt_user = 0
    for index, filename in enumerate(filenames[num*file_num_each_job_test: (num+1) * file_num_each_job_test]):
        table = pd.read_csv(test_main_visit_path + filename + ".txt", header=None, sep='\t')
        init_cishu = visit2array(table)
        users = table[0]

        np.save(config.test_npy_visit_path + filename + '.npy', init_cishu)

        cnt_user += len(users)
        total_users += list(users.values)

        sys.stdout.write('\r>> Processing test visit data %d/%d' % (index + 1, length))
        sys.stdout.flush()

    sys.stdout.write('\n')
    print('test users:', cnt_user)
    print("using time:%.2fs" % (time.time() - start_time))
    return total_users


def run_train_feature(num):
    train_users = visit2array_train(num)

def run_val_feature(num):
    val_users = visit2array_val(num)

def run_test_feature(num):
    test_users = visit2array_test(num)

def run():

    threads = []
    # for i in range(workers_train):
    #     p = Process(target=run_train_feature, args=[i])
    #     threads.append(p)
    #     p.start()

    # for i in range(workers_train):
    #     p = Process(target=run_val_feature, args=[i])
    #     threads.append(p)
    #     p.start()

    for i in range(workers_test):
        p = Process(target=run_test_feature, args=[i])
        threads.append(p)
        p.start()

if __name__ == '__main__':
    run()
    print('run feature done!')