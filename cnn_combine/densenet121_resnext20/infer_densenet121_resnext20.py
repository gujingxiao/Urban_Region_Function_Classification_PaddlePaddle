import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import paddle
import paddle.fluid as fluid
import numpy as np
from cnn_combine.densenet121_resnext20.config_densenet121_resnext20 import config
from cnn_combine.utils.data_feeder import infer_mapper, infer_data_reader
from cnn_combine.densenet121_resnext20.train_densenet121_resnext20 import MultiModel
from imgaug import augmenters as iaa

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

    model_path = 'model/7'
    if os.path.isdir(model_path):
        fluid.io.load_params(exe, model_path, main_program=freeze_prog)

    # 固化模型
    fluid.io.save_inference_model(config['freeze_path'], ['image', 'visit'], model, exe, freeze_prog)


# 模型预测代码
def infer():
    batch_size = 64
    tta = False
    """ 模型预测函数 """
    place = fluid.CUDAPlace(0) if config['use_gpu'] else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # 加载先前固化的模型
    [inference_program, feed_target_names, fetch_list] = fluid.io.load_inference_model(dirname=config['freeze_path'],
                                                                                       executor=exe)

    # 生成预测数据读取器
    infer_reader = paddle.batch(
        reader=infer_data_reader(config['root_path']+'val.txt', infer_mapper, config['root_path']+"npy/train_visit/"), batch_size=batch_size)

    f = open('sSubmit.txt', 'w+')
    for i, data in enumerate(infer_reader()):
        images = []
        visits = []

        if tta == True:
            filpImages = []

            augment_img = iaa.Sequential([
                iaa.Fliplr(1.0),
            ], random_order=False)


        for index in range(len(data)):
            images.append(data[index][0])
            visits.append(data[index][1])

            if tta == True:
                image_flip = augment_img.augment_image(data[index][0])
                filpImages.append(image_flip)

        images_feed = np.array(images).astype('float32')
        visits_feed = np.array(visits).astype('float32')
        if tta == True:
            flip_feed = np.array(filpImages).astype('float32')

        result = exe.run(inference_program, fetch_list=fetch_list, feed={
            feed_target_names[0]: images_feed,
            feed_target_names[1]: visits_feed})

        if tta == True:
            flip_result = exe.run(inference_program, fetch_list=fetch_list, feed={
                feed_target_names[0]: flip_feed,
                feed_target_names[1]: visits_feed})

        if tta == True:
            for index in range(len(result[0])):
                item = result[0][index]
                flip_item = flip_result[0][index]
                category = np.argmax((item + flip_item) / 2.0) + 1
                f.write('%s\t%03d\n' % (data[index][3], category))
        else:
            for index in range(len(result[0])):
                item = result[0][index]
                category = np.argmax(item) + 1
                f.write('%s\t%03d\n' % (data[index][3], category))
                np.save(
                    '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/CityFunction/cnnStacking/val/densenet121_resnext20/{}.npy'.format(
                        data[index][3]),
                    item)

        if i % 100 == 0:
            print('Step %d / Total %d ' % (i, int(100000 / batch_size)))
    f.close()

if __name__ == '__main__':
    freeze_model()
    infer()