import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import paddle
import paddle.fluid as fluid
from rule_feature.feature_stacking.matrix_stacking.train_matrix_stacking import stackingCnnModel
from rule_feature.feature_stacking.matrix_stacking.config_matrix_stacking import config
from rule_feature.feature_stacking.matrix_stacking.matrix_stacking_data_feeder import infer_mapper, infer_data_reader

fold = 4

# 模型固化代码
def freeze_model():
    """ 模型固化函数 """
    freeze_prog = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(freeze_prog, startup_prog):
        # 模型定义
        input_stacking = fluid.layers.data(shape=config['stacking_shape'], name='stacking')
        model = stackingCnnModel(input_stacking)

    exe = fluid.Executor(fluid.CPUPlace())

    model_path = 'best_model/best_model_fold{}'.format(fold)
    if os.path.isdir(model_path):
        fluid.io.load_params(exe, model_path, main_program=freeze_prog)

    # 固化模型
    fluid.io.save_inference_model(config['freeze_path'], ['stacking'], model, exe, freeze_prog)


def infer():
    batch_size = 1000
    """ 模型预测函数 """
    place = fluid.CUDAPlace(0) if config['use_gpu'] else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # 加载先前固化的模型
    [inference_program, feed_target_names, fetch_list] = fluid.io.load_inference_model(dirname=config['freeze_path'],
                                                                                       executor=exe)

    # 生成预测数据读取器
    infer_reader = paddle.batch(
        reader=infer_data_reader(config['root_path']+'test.txt', infer_mapper, config['root_path']+"ruleFeature/stacking/test_npy_matrix/"), batch_size=batch_size)

    f = open('mSubmit.txt', 'w+')
    for i, data in enumerate(infer_reader()):

        features = []

        for index in range(len(data)):
            features.append(data[index][0])

        features_feed = np.array(features).astype('float32')

        result = exe.run(inference_program, fetch_list=fetch_list, feed={
            feed_target_names[0]: features_feed})

        for index in range(len(result[0])):
            item = result[0][index]
            category = np.argmax(item) + 1
            f.write('%s\t%03d\n' % (data[index][2], category))
            np.save('/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/CityFunction/cnnStacking/stacking/matrix/{}/{}.npy'.format(fold, data[index][2]),
                    item)

        if i % 10 == 0:
            print('Step %d / Total %d ' % (i, int(100000 / batch_size)))
    f.close()

if __name__ == '__main__':
    freeze_model()
    infer()