# 全局参数
config = {
    'use_gpu': True,                # 是否使用 GPU
    'image_shape': (3, 100, 100),   # Image Network 输入尺寸
    'visit_shape': (7, 26, 24),     # Visit Network 输入尺寸
    'lr': 0.0001,                     # 学习率
    'lr_decay': [1, 0.5, 0.25, 0.1, 0.05],
    'num_epochs': [3, 6, 9, 12],              # 训练轮数
    'total_epochs': 15,
    'model_path': 'model',          # 模型缓存路径
    'freeze_path': 'freeze_model',  # 模型固化路径
    'root_path': '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/CityFunction/',
}