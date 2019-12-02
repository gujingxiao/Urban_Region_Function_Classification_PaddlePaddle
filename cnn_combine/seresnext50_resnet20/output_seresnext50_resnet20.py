import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import paddle
import paddle.fluid as fluid
import numpy as np
from cnn_combine.seresnext50_resnet20.config_seresnext50_resnet20 import config
from cnn_combine.utils.data_feeder import infer_mapper, infer_data_reader
from cnn_combine.seresnext50_resnet20.train_seresnext50_resnet20 import MultiModel

# 模型固化代码
def freeze_model():
    """ 模型固化函数 """
    freeze_prog = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(freeze_prog, startup_prog):
        # 模型定义
        input_image = fluid.layers.data(shape=config['image_shape'], name='image')
        input_visit = fluid.layers.data(shape=config['visit_shape'], name='visit')
        model = MultiModel(input_image, input_visit)

    exe = fluid.Executor(fluid.CPUPlace())

    model_path = 'best_model'
    if os.path.isdir(model_path):
        fluid.io.load_params(exe, model_path, main_program=freeze_prog)

    # 固化模型
    fluid.io.save_inference_model(config['freeze_path'], ['image', 'visit'], model, exe, freeze_prog)


# 模型预测代码
def infer():
    mode = 'val' # val test
    batch_size = 64

    if mode == 'val':
        length = 40000
        txt_file = 'val.txt'
        folder = 'train'
    else:
        length = 100000
        txt_file = 'test.txt'
        folder = 'test'

    """ 模型预测函数 """
    place = fluid.CUDAPlace(0) if config['use_gpu'] else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # 加载先前固化的模型
    [inference_program, feed_target_names, fetch_list] = fluid.io.load_inference_model(dirname=config['freeze_path'],
                                                                                       executor=exe)

    # 生成预测数据读取器
    infer_reader = paddle.batch(
        reader=infer_data_reader(config['root_path']+txt_file, infer_mapper, config['root_path']+"npy/{}_visit/".format(folder)), batch_size=batch_size)

    for i, data in enumerate(infer_reader()):
        images = []
        visits = []


        for index in range(len(data)):
            images.append(data[index][0])
            visits.append(data[index][1])

        images_feed = np.array(images).astype('float32')
        visits_feed = np.array(visits).astype('float32')

        result = exe.run(inference_program, fetch_list=fetch_list, feed={
            feed_target_names[0]: images_feed,
            feed_target_names[1]: visits_feed})

        for index in range(len(result[0])):
            item = result[0][index]
            np.save(config['root_path'] + 'cnnStacking/{}/seresnext50_resnet20/'.format(mode) + data[index][3] + '.npy', item)

        if i % 100 == 0:
            print('Step %d / Total %d ' % (i, int(length / batch_size)))

if __name__ == '__main__':
    freeze_model()
    infer()