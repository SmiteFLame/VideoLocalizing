"""
SRNet - Editing Text in the Wild
Some configurations.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License (see LICENSE for details)
Written by Yu Qian
"""

# device
gpu = 0

# pretrained vgg
vgg19_weights = 'models/vgg19.pb'

# model parameters
lt = 1.
lt_alpha = 1.
lb = 1.
lb_beta = 10.
lf = 1.
lf_theta_1 = 10.
lf_theta_2 = 1.
lf_theta_3 = 500.
epsilon = 1e-8

# train
learning_rate = 1e-4 # default 1e-3
decay_rate = 0.9
decay_steps = 10000
staircase = False
beta1 = 0.9 # default 0.9
beta2 = 0.999 # default 0.999
max_iter = 500000
show_loss_interval = 50
write_log_interval = 50
save_ckpt_interval = 10000
gen_example_interval = 1000
pretrained_ckpt_path = None
train_name = None # used for name examples and tensorboard logdirs, set None to use time

# data
batch_size = 8
data_shape = [64, None]

# predict
predict_ckpt_path = 'models/synthesizing_model'
predict_data_dir = None