"""
配置文件 - K-Planes BRDF 模型
"""

config = {
    # 实验名称
    'expname': 'kplanes_brdf',
    'logdir': './logs',
    'device': 'cuda:0',

    # 数据设置
    'data_downsample': 1.0,
    'data_dirs': ['data'],
    'contract': False,
    'ndc': False,

    # 优化设置
    'num_steps': 30001,
    'batch_size': 4096,
    'optim_type': 'adam',
    'scheduler_type': 'warmup_cosine',
    'lr': 0.01,

    # 正则化
    'plane_tv_weight': 0.0001,
    'plane_tv_weight_proposal_net': 0.0001,
    'histogram_loss_weight': 1.0,
    'distortion_loss_weight': 0.001,

    # 训练设置
    'save_every': 30000,
    'valid_every': 30000,
    'save_outputs': True,
    'train_fp16': True,

    # 光线采样设置
    'single_jitter': False,
    'num_samples': 48,
    'num_proposal_samples': [256, 128],
    'num_proposal_iterations': 2,
    'use_same_proposal_network': False,
    'use_proposal_weight_anneal': True,
    'proposal_net_args_list': [
        {'num_input_coords': 3, 'num_output_coords': 8, 'resolution': [90, 90, 180]},
    ],

    # 模型设置
    'multiscale_res': [1],
    'density_activation': 'trunc_exp',
    'concat_features_across_scales': True,
    'linear_decoder': False,
    'linear_decoder_layers': 4,
    
    # K-Planes 网格配置
    # resolution: [theta_h, theta_d, phi_d, time/wavelength]
    # - theta_h: 半程向量极角 [0, 90°]
    # - theta_d: 差分向量极角 [0, 90°]  
    # - phi_d: 差分向量方位角 [0, 180°]
    # - time: 时间/波长维度 (39 个采样点)
    'grid_config': [{
        'grid_dimensions': 2,           # 2D 平面
        'input_coordinate_dim': 4,      # 4D 输入 (theta_h, theta_d, phi_d, time)
        'output_coordinate_dim': 64,    # 64 维特征
        'resolution': [90, 90, 180, 39] # 分辨率
    }],
}
