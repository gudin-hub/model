import os
import torch

import datetime
import numpy as np
from torch.utils.data import DataLoader
from nets.unet import Unet

from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.callbacks import EvalCallback

'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照JPG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
3、仅有按照VOC格式数据训练的模型可以利用这个文件进行miou的计算。
'''
if __name__ == "__main__":
    #---------------------------------------------------------------------------#
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #------------------------------#
    #   分类个数+1、如2+1
    #------------------------------#
    num_classes     = 2
    batch_size      = 5
    cuda =True
    # ------------------------------------------------------------------#
    #   是否给不同种类赋予不同的损失权值，默认是平衡的。
    #   设置的话，注意设置成numpy形式的，长度和num_classes一样。
    #   如：
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    # ------------------------------------------------------------------#
    cls_weights = np.ones([num_classes], np.float32)

    #--------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    #--------------------------------------------#
    name_classes    = ["background", "cnv"]
    # --------------------------------------------#
    #   input_shape     输入图片的大小，32的倍数
    # -----------------------------------------------------#
    input_shape = [512, 512]
    #-----------------------------------------------------#
    #   主干网络选择
    #   vgg
    #   resnet50
    #-----------------------------------------------------#
    backbone    = "resnet50"
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'

    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_val     = len(val_lines)
    epoch_step_val = num_val // batch_size
    val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=0, pin_memory=True,
                         drop_last=True, collate_fn=unet_dataset_collate, sampler=None)
    model_path = r".\logs\best_epoch_weights.pth"

    model = Unet(num_classes=num_classes, backbone=backbone)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("Load model done.")

    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join('eval_logs', "loss_" + str(time_str))
    os.makedirs(log_dir)
    eval_callback = EvalCallback(model, input_shape, num_classes, name_classes, val_lines, VOCdevkit_path, log_dir,
                                 cuda,
                                 eval_flag=True, period=1)
    eval_callback.eval_metrics(model)


