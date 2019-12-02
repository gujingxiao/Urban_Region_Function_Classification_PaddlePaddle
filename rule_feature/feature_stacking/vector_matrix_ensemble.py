import numpy as np
import os

base_path = '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/CityFunction/cnnStacking/stacking/'

file_list = os.listdir(base_path + 'vector/1/')
f = open('eSubmit.txt', 'w+')
for index, item in enumerate(file_list):
    matrix_npy1 = np.load(os.path.join(base_path, 'matrix/1', item))
    vector_npy1 = np.load(os.path.join(base_path, 'vector/1', item))
    matrix_npy2 = np.load(os.path.join(base_path, 'matrix/2', item))
    vector_npy2 = np.load(os.path.join(base_path, 'vector/2', item))
    matrix_npy3 = np.load(os.path.join(base_path, 'matrix/3', item))
    vector_npy3 = np.load(os.path.join(base_path, 'vector/3', item))
    matrix_npy4 = np.load(os.path.join(base_path, 'matrix/4', item))
    vector_npy4 = np.load(os.path.join(base_path, 'vector/4', item))

    file_id = item.split('.')[0]
    label = np.argmax(vector_npy1 + matrix_npy1 + vector_npy2 + matrix_npy2 + vector_npy3 + matrix_npy3 +
                      vector_npy4 + matrix_npy4) + 1
    if index % 1000 == 0:
        print(index, file_id, label)
    f.write('%s\t%03d\n' % (file_id, label))
f.close()