#!/usr/bin/env python
# -*- coding:utf8 -*-
from __future__ import absolute_import
# though cupy is not used but without this line, it raise errors...
import cupy as cp
import os
import numpy as np
import ipdb
import matplotlib
from tqdm import tqdm
import torch as t

import resource

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize, Transform, TestDataset_all
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
from PIL import Image
from matplotlib import pyplot as plt
from data.util import read_image
from data import util
from utils.utils import *
import argparse
#from uitils import *
# 更改gpu使用的核心
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#使用作者的模型训练

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


#CUDA_VISIBLE_DEVICES=1 python3 test.py --test_path="./checkpoints/fasterrcnn_11091854_0.4595028584047549.pth"

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
  parser.add_argument('--test_path', dest='test_path',
                      help='directory to load models', default="checkpoints",
                      type=str)
  args = parser.parse_args()
  return args
def test(**kwargs):
    args = parse_args()
    opt._parse(kwargs)
    opt.test_path = args.test_path
    print('load data')

    dataset = Dataset(opt)
    dataloader = data_.DataLoader(dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False,
                                       pin_memory=True
                                       )
    testset_all = TestDataset_all(opt, 'test')
    test_all_dataloader = data_.DataLoader(testset_all,
                                           batch_size=1,
                                           num_workers=opt.test_num_workers,
                                           shuffle=False,
                                           pin_memory=True
                                           )

    faster_rcnn = FasterRCNNVGG16()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    print('model construct completed')

    # 加载训练过的模型，在config配置路径就可以了
    if opt.test_path:
        trainer.load(opt.test_path)
        print('load test model from %s' % opt.test_path)

    trainer.eval()
    eval_result = eval(test_all_dataloader, faster_rcnn, test_num=opt.test_num)
    trainer.vis.plot('test_map', eval_result['map'])
    log_info = 'ap:{}\nmap:{}'.format(str(eval_result['ap']),
                                      str(eval_result['map']))
    print(log_info)
    trainer.vis.log(log_info)

    result_path = 'eval_result/'
    results_file = '_'.join(opt.test_path.split('/')[-2:]) + '.txt'

    with open(result_path + results_file, "w") as f:
        f.write(log_info)

if __name__ == '__main__':
    test()
