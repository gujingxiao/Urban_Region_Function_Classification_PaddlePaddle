# 全局参数
config = {
    'use_gpu': True,                # 是否使用 GPU
    'stacking_shape': (1, 63),
    'lr': 0.02,                     # 学习率
    'lr_decay': [1, 0.5, 0.25, 0.125, 0.0625, 0.03125],
    'num_epochs': [8, 16, 24, 32, 40],              # 训练轮数
    'total_epochs': 50,
    'model_path': 'model',          # 模型缓存路径
    'freeze_path': 'freeze_model',  # 模型固化路径
    'root_path': '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/CityFunction/',
}