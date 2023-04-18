#!/usr/bin/env python
# -*- coding:utf-8 -*-
from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    # voc_data_dir = '../../BCCD_Dataset/BCCD'
    voc_data_dir = '/home/goujiaxiang/Code/ILMIL/ILMIL/VOCdevkit/VOC2007'
    min_size = 600  # image resize
    max_size = 1000  # image resize
    num_workers = 8
    test_num_workers = 8
    predict_socre = 0.05
    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.33
    lr_decay_step = 5 # 降低学习率的间隔
    lr = 1e-4
    best_map = 0

    # visualization
    env = 'faster-rcnn-1'  # visdom env
    port = 8097
    plot_every = 40  # vis every N iter

    # preset
    data = 'voc'
    pretrained_model = 'vgg16'

    # training
    start_epoch = 0
    epoch = 20

    use_adam = False  # Use Adam optimizer
    use_chainer = False  # try match everything as chainer
    use_drop = False  # use dropout in RoIHead
    # debug
    debug_file = './tmp/debugf'
    threshold = 0.5
    test_num = 5000
    # model
    is_distillation = False
    only_use_cls_distillation = is_distillation and True
    use_hint = is_distillation and True
    testtxt = 'test'
    datatxt = 'trainval'
    load_path = "../fasterrcnn.pth"
    # load_path = None

    # 训练异常终止后恢复训练
    resume = ""

    caffe_pretrain = False  # use caffe pretrained model instead of torchvision
    caffe_pretrain_path = 'pretrained_model/fasterrcnn_12231419_16_0.7089819173484984'

    test_path = ""

    VOC_BBOX_LABEL_NAMES_all = (
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow'
    )

    VOC_BBOX_LABEL_NAMES_test = (
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow'
    )
    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        # pprint(self._state_dict())
        print(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items()
                if not k.startswith('_')}


opt = Config()
