import os
import numpy as np

root_path = '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/CityFunction/'

mode = 'matrix'  # matrix vector
# train
stacking_paths = [['ruleFeature/cxq_limit/val_npy',
                   'ruleFeature/cxq_wide/val_npy',
                   'ruleFeature/day_rule/val_npy',
                   'cnnStacking/val/seresnext50_resnet20',
                   'cnnStacking/val/densenet121_resnext20',
                   'cnnStacking/val/resnet50_dpn26',
                   'cnnStacking/val/txt'],

                  ['ruleFeature/cxq_limit/test_npy',
                   'ruleFeature/cxq_wide/test_npy',
                   'ruleFeature/day_rule/test_npy',
                   'cnnStacking/test/seresnext50_resnet20',
                   'cnnStacking/test/densenet121_resnext20',
                   'cnnStacking/test/resnet50_dpn26',
                   'cnnStacking/test/txt']]

save_paths = ['ruleFeature/stacking/val_npy_{}'.format(mode),
              'ruleFeature/stacking/test_npy_{}'.format(mode)]

for index in range(2):
    file_list = os.listdir(os.path.join(root_path, stacking_paths[index][0]))
    if not os.path.exists(os.path.join(root_path, save_paths[index])):
        os.makedirs(os.path.join(root_path, save_paths[index]))
    i = 0
    for filename in file_list:
        i += 1
        stackings = []
        count = 0
        for sub_dir in stacking_paths[index]:
            count += 1

            if index < 1:
                if count > 3:
                    cnnFilename = filename.split('_')[0] + '.npy'
                    sub_path = os.path.join(root_path, sub_dir, cnnFilename)
                else:
                    sub_path = os.path.join(root_path, sub_dir, filename)
            else:
                sub_path = os.path.join(root_path, sub_dir, filename)

            npy_load = np.load(sub_path)

            if np.sum(npy_load) == 0.0:
                npy_load[:] = 0.111

            npy_load = npy_load / np.sum(npy_load)

            stackings.append(npy_load)

        npy_stacking = np.array(stackings)
        if mode == 'vector':
            npy_stacking = np.reshape(npy_stacking, (1, len(stacking_paths[0])* 9))
        np.save(os.path.join(root_path, save_paths[index], filename), npy_stacking)

        if i % 1000 == 0:
            print('Finished %d / %d ' % (i, len(file_list)))