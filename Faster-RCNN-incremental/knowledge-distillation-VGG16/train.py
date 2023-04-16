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
# import cv2
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
import datetime
# 更改gpu使用的核心
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')
# VOC_BBOX_LABEL_NAMES = opt.VOC_BBOX_LABEL_NAMES

def train(**kwargs):
    opt._parse(kwargs)
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

    # 用来保存log_info的文件
    results_file = "results{}_isDistillation{}_onlyUseClsDistillation{}_useHint{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), opt.is_distillation, opt.only_use_cls_distillation, opt.use_hint)

    tsf = Transform(opt.min_size, opt.max_size)
    faster_rcnn = FasterRCNNVGG16()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    print('model construct completed')

    # 加载训练过的模型，在config配置路径就可以了
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)

    if opt.resume: # 恢复训练
        state_dict = t.load(opt.resume)
        trainer.faster_rcnn.load_state_dict(state_dict['model'])
        results_file = state_dict['results_file_name']
        opt.lr = state_dict['lr']
        opt.best_map = state_dict['best_map']
        opt.start_epoch = state_dict['epoch'] + 1
        trainer.optimizer.load_state_dict(state_dict['optimizer'])
        trainer.vis.load_state_dict(state_dict['vis_info'])

    # 提取蒸馏知识所需要的软标签（恢复训练时不能执行这段代码，教师模型和数据集一致时可以注释掉该段代码以提高训练速度）
    if opt.is_distillation == True and not opt.resume:
        opt.predict_socre = 0.3
        for ii, (imgs, sizes, gt_bboxes_, gt_labels_, scale, id_) in tqdm(enumerate(dataloader)):
            if len(gt_bboxes_) == 0:
                continue
            sizes = [sizes[0][0].item(), sizes[1][0].item()]
            pred_bboxes_, pred_labels_, pred_scores_, features_ = trainer.faster_rcnn.predict(imgs, [sizes])

            img_file = os.path.join(opt.voc_data_dir, 'JPEGImages', id_[0] + '.jpg')
            ori_img = read_image(img_file, color=True)
            img, pred_bboxes_, pred_labels_, scale_ = tsf((ori_img, pred_bboxes_[0], pred_labels_[0]))

            #去除软标签和真值标签重叠过多的部分，去除错误的软标签
            pred_bboxes_, pred_labels_, pred_scores_ = py_cpu_nms(
                gt_bboxes_[0], gt_labels_[0], pred_bboxes_, pred_labels_, pred_scores_[0])

            #存储软标签，这样存储不会使得GPU占用过多
            np.save('s_label/label/' + str(id_[0]) + '.npy', pred_labels_.cpu())
            np.save('s_label/bbox/' + str(id_[0]) + '.npy', pred_bboxes_.cpu())
            np.save('s_label/feature/' + str(id_[0]) + '.npy', features_.cpu())
            np.save('s_label/score/' + str(id_[0]) + '.npy', pred_scores_.cpu())

        opt.predict_socre = 0.05
    t.cuda.empty_cache()

    # visdom 显示所有类别标签名
    # trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = opt.best_map
    lr_ = opt.lr

    for epoch in range(opt.start_epoch, opt.epoch):
        print('epoch=%d' % epoch)

        # 重置混淆矩阵
        trainer.reset_meters()
        # tqdm可以在长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)，
        # 是一个快速、扩展性强
        for ii, (img, sizes, bbox_, label_, scale, id_) in tqdm(enumerate(dataloader)):
            if len(bbox_) == 0:
                continue
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            # 训练的就这一步 下面的都是打印的信息
            # 转化成pytorch能够计算的格式，转tensor格式
            if opt.is_distillation == True:
                #读取软标签
                teacher_pred_labels = np.load('s_label/label/' + str(id_[0]) + '.npy')
                teacher_pred_bboxes = np.load('s_label/bbox/' + str(id_[0]) + '.npy')
                teacher_pred_features_ = np.load('s_label/feature/' + str(id_[0]) + '.npy')
                teacher_pred_scores = np.load('s_label/score/' + str(id_[0]) + '.npy')
                #格式转换
                teacher_pred_bboxes = teacher_pred_bboxes.astype(np.float32)
                teacher_pred_labels = teacher_pred_labels.astype(np.int32)
                teacher_pred_scores = teacher_pred_scores.astype(np.float32)
                #转成pytorch格式
                teacher_pred_bboxes_ = at.totensor(teacher_pred_bboxes)
                teacher_pred_labels_ = at.totensor(teacher_pred_labels)
                teacher_pred_scores_ = at.totensor(teacher_pred_scores)
                teacher_pred_features_ = at.totensor(teacher_pred_features_)
                #使用GPU，其实上一步已经把数据复制到gpu上面去了
                teacher_pred_bboxes_ = teacher_pred_bboxes_.cuda()
                teacher_pred_labels_ = teacher_pred_labels_.cuda()
                teacher_pred_scores_ = teacher_pred_scores_.cuda()
                teacher_pred_features_ = teacher_pred_features_.cuda()

                # 如果dataset.py 中的Transform 设置了图像翻转,就要使用这个判读软标签是否一起翻转
                if(teacher_pred_bboxes_[0][1] != bbox[0][0][1]):
                    _, o_C, o_H, o_W = img.shape
                    teacher_pred_bboxes_ = flip_bbox(
                        teacher_pred_bboxes_, (o_H, o_W), x_flip=True)

                losses = trainer.train_step(img, bbox, label, scale, epoch,
                                            teacher_pred_bboxes_, teacher_pred_labels_, teacher_pred_features_, teacher_pred_scores_)
            else:
                trainer.train_step(img, bbox, label, scale, epoch)

            if (ii + 1) % opt.plot_every == 0:
                # if os.path.exists(opt.debug_file):
                #     ipdb.set_trace() # 用于Ipython调试

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img)

                # plot predicti bboxes
                _bboxes, _labels, _scores, _ = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())

        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'epoch:{}\nlr:{}\nap:{}\nmap:{}\nloss:{}\n'.format(str(epoch),
                                                                      str(lr_),
                                                                      str(eval_result['ap']),
                                                                      str(eval_result['map']),
                                                                      str(trainer.get_meter_data())) # 返回的是平均损失
        print(log_info)
        trainer.vis.log(log_info)
        # write into txt
        result_path = 'train_eval_result/' + str(len(opt.VOC_BBOX_LABEL_NAMES_all)) + '-' + str(
            len(opt.VOC_BBOX_LABEL_NAMES_test)) + '/'
        save_dir = os.path.dirname(result_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(result_path + results_file, "a") as f:
            if epoch == 0:  # 第一次把配置信息也写入文件中
                f.write(str(opt._state_dict()) + '\n\n')
            f.write(log_info + "\n")

        # 保存最好结果并记住路径
        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(results_file_name=results_file, lr = lr_, best_map=best_map, epoch=epoch, isDistillation='isDistillation' + str(opt.is_distillation), onlyUseClsDistillation='onlyUseClsDistillation' + str(opt.only_use_cls_distillation), useHint='useHint' + str(opt.use_hint))

        # 每10轮加载前面最好权重，并且减少学习率
        if epoch % 10 == 0 and epoch != 0:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay


if __name__ == '__main__':
    import fire

    fire.Fire()
