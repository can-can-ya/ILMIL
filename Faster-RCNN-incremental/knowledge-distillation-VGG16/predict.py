'''
作者：苟加祥
功能：XXX
时间：2023/4/18 19:18
'''

import os
from utils.config import opt
from data.util import read_image
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from utils.vis_tool import visdom_bbox
from utils import array_tool as at
import xml.etree.ElementTree as ET
import numpy as np
# 更改gpu使用的核心
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def predict(**kwargs):
    opt._parse(kwargs)

    # Load a image
    img = read_image(opt.predict_picture_path, color=True)

    # 构建模型
    faster_rcnn = FasterRCNNVGG16()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()

    # 加载训练过的模型，在config配置路径就可以了
    trainer.load(opt.predict_model_path)

    # plot predicti bboxes
    _bboxes, _labels, _scores, _ = trainer.faster_rcnn.predict([img], visualize=True)
    pred_img = visdom_bbox(img,
                           at.tonumpy(_bboxes[0]),
                           at.tonumpy(_labels[0]).reshape(-1),
                           at.tonumpy(_scores[0]))
    trainer.vis.img('pred_img', pred_img)

    if opt.predict_GT: # 绘制GT
        anno_path = opt.predict_picture_path.split('/')
        anno_path[-2] = 'Annotations'
        b = anno_path[-1]
        b = b.split('.')
        b[-1] = 'xml'
        b = '.'.join(b)
        anno_path[-1] = b
        anno_path = '/'.join(anno_path)

        anno = ET.parse(anno_path)

        bbox = list()
        label = list()
        # 一张图像有多个bbox标签，循环读取bbox标签
        for obj in anno.findall('object'):
            name = obj.find('name').text.lower().strip()

            if name not in opt.VOC_BBOX_LABEL_NAMES_test:
                continue

            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])

            label.append(opt.VOC_BBOX_LABEL_NAMES_test.index(name))
        # np.stack在axis=i上增加一维。这边似乎只是转为np型数组
        # 由于这边要做类别增量，可能出现图像没有bbox，所有加一个判断
        # 使得没有bbox的图像不会报错 而是在后面的操作中会跳过
        if len(bbox) > 0:
            bbox = np.stack(bbox).astype(np.float32)
            label = np.stack(label).astype(np.int32)
        gt_img = visdom_bbox(img,
                             bbox,
                             label)
        trainer.vis.img('gt_img', gt_img)

if __name__ == '__main__':
    import fire

    fire.Fire()