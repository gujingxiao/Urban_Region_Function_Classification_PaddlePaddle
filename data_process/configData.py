# coding=utf-8
import warnings

class DefaultConfigs(object):
    main_data_path = '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/CityFunction/'

    train_table_path = main_data_path + 'train.txt'
    train_main_visit_path = main_data_path + "train_visit/"
    train_npy_visit_path = '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/CityFunction/npy/train_visit/'

    val_table_path = main_data_path + 'val.txt'
    val_main_visit_path = main_data_path + "train_visit/"
    val_npy_visit_path = '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/CityFunction/npy/train_visit/'

    test_table_path = main_data_path + 'test.txt'
    test_main_visit_path = main_data_path + "test_visit/"
    test_npy_visit_path = '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/CityFunction/npy/test_visit/'

def parse(self, kwargs):
    """
    update config by kwargs
    """
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfigs.parse = parse
config = DefaultConfigs()
# print(config.main_data_path)