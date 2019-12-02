import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from cnn_combine.seresnext50_resnet20.config_seresnext50_resnet20 import config
from cnn_combine.utils.data_feeder import train_mapper, val_mapper, data_reader
from modelzoo.SE_ResNeXt import SE_ResNeXt
from modelzoo.ResNet_vd import ResNet20_vd
# 网络结构定义

def MultiModel(image, visit):
    se_resnext = SE_ResNeXt()
    resnet20_vd = ResNet20_vd()
    output_image = se_resnext.net(image, class_dim=1024)
    output_visit = resnet20_vd.net(visit, class_dim=1024)
    output = fluid.layers.concat([output_image, output_visit], axis=1)
    prediction = fluid.layers.fc(input=output, size=9, act='softmax',
        param_attr=ParamAttr(name='multi_fc_weights'),
        bias_attr=ParamAttr(name='multi_fc_bias'))
    return prediction

# 模型训练代码
def train():
    batch_size = 48

    iters = 360000 // batch_size
    lr = config['lr']
    boundaries = [i * iters  for i in config["num_epochs"]]
    values = [ i * lr for i in config["lr_decay"]]

    train_prog = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        input_image = fluid.layers.data(shape=config['image_shape'], name='image')
        input_visit = fluid.layers.data(shape=config['visit_shape'], name='visit')
        label = fluid.layers.data(shape=[1], name='label', dtype='int64')

        with fluid.unique_name.guard():
            out = MultiModel(input_image, input_visit)
            # 获取损失函数和准确率函数
            cost = fluid.layers.cross_entropy(out, label=label)
            avg_cost = fluid.layers.mean(cost)
            acc = fluid.layers.accuracy(out, label=label)

            # 获取训练和测试程序
            test_program = train_prog.clone(for_test=True)

            # 定义优化方法
            optimizer = fluid.optimizer.AdamOptimizer(learning_rate=fluid.layers.piecewise_decay(boundaries, values),
                                                     regularization=fluid.regularizer.L2DecayRegularizer(1e-5))
            optimizer.minimize(avg_cost)

    # 定义一个使用GPU的执行器
    place = fluid.CUDAPlace(0) if config['use_gpu'] else fluid.CPUPlace()
    exe = fluid.Executor(place)
    # 进行参数初始化
    exe.run(startup_prog)

    # 继续训练
    if os.path.isdir('best_model'):
       fluid.io.load_params(exe, 'best_model', main_program=train_prog)

    # def if_exist_se_resnext(var):
    #    return os.path.exists(os.path.join('pretrained', var.name))
    #
    # fluid.io.load_vars(exe, 'pretrained', main_program=train_prog, predicate=if_exist_se_resnext)

    train_reader = paddle.batch(reader=data_reader(config['root_path']+ 'train.txt', train_mapper,
                                                   config['root_path']+ 'npy/train_visit/'), batch_size=batch_size)
    test_reader = paddle.batch(reader=data_reader(config['root_path']+ 'val.txt', val_mapper,
                                                  config['root_path']+ 'npy/train_visit/'), batch_size=batch_size)

    # 定义输入数据维度
    feeder = fluid.DataFeeder(place=place, feed_list=[input_image, input_visit, label])

    train_losses = []
    train_accs = []
    best_acc = 0
    for epoch in range(config['total_epochs']):
        for step, data in enumerate(train_reader()):
            train_loss, train_acc = exe.run(program=train_prog,
                                            feed=feeder.feed(data),
                                            fetch_list=[avg_cost, acc])
            train_losses.append(train_loss[0])
            train_accs.append(train_acc[0])

            # 每 100 个 batch 打印一次信息
            if step % 100 == 0:
                print('Epoch %d step %d: loss %0.5f accuracy %0.5f' %
                      (epoch, step, sum(train_losses) / len(train_losses), sum(train_accs) / len(train_accs)))

            if step % 500 == 0 and step != 0:
                # 保存模型参数
                if not os.path.isdir(os.path.join(config['model_path'], str(epoch))):
                    os.makedirs(os.path.join(config['model_path'], str(epoch)))
                fluid.io.save_params(exe, os.path.join(config['model_path'], str(epoch)), main_program=train_prog)


                # 进行测试
                test_accs = []
                test_costs = []
                for batch_id, data in enumerate(test_reader()):
                    test_cost, test_acc = exe.run(program=test_program,
                                                  feed=feeder.feed(data),
                                                  fetch_list=[avg_cost, acc])
                    test_accs.append(test_acc[0])
                    test_costs.append(test_cost[0])
                # 求测试结果的平均值
                test_cost = (sum(test_costs) / len(test_costs))
                test_acc = (sum(test_accs) / len(test_accs))
                print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (step, test_cost, test_acc))

        # 进行测试
        test_accs = []
        test_costs = []
        for batch_id, data in enumerate(test_reader()):
            test_cost, test_acc = exe.run(program=test_program,
                                          feed=feeder.feed(data),
                                          fetch_list=[avg_cost, acc])
            test_accs.append(test_acc[0])
            test_costs.append(test_cost[0])
        # 求测试结果的平均值
        test_cost = (sum(test_costs) / len(test_costs))
        test_acc = (sum(test_accs) / len(test_accs))
        print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (step, test_cost, test_acc))

        # 保存模型参数
        if not os.path.isdir(os.path.join(config['model_path'], str(epoch))):
            os.makedirs(os.path.join(config['model_path'], str(epoch)))
        #if test_acc > best_acc:
        fluid.io.save_params(exe, os.path.join(config['model_path'], str(epoch)), main_program=train_prog)
        best_acc = test_acc

if __name__ == '__main__':
    train()