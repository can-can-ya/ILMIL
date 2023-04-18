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

if __name__ == '__main__':
    import fire

    fire.Fire()