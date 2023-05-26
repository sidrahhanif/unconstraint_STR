import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.nn.init as init
import os
from models.torchutil import *
from models.yolo import Model
from models.craft import CRAFT
from models.basenet.vgg16_bn import vgg16_bn
from utils.general import (LOGGER, check_dataset, check_file, check_git_status, check_img_size, check_requirements,
                           check_suffix, check_yaml, colorstr, get_latest_run, increment_path, init_seeds,
                           intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods, one_cycle,
                           print_args, print_mutation, strip_optimizer)
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, select_device, torch_distributed_zero_first
from utils.downloads import attempt_download

class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CRAFT_yolov5(nn.Module):
    def __init__(self,opt, pretrained=True, freeze=False):
        super(CRAFT_yolov5, self).__init__()

        craft_model = CRAFT()
        craft_model.load_state_dict(copyStateDict(torch.load('/raid/tmp/Text_detection/CRAFT/data/pretrained_weights/craft_mlt_25k.pth')))
        self.craft = craft_model
        # Model
        LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
        if pretrained:
            with torch_distributed_zero_first(LOCAL_RANK):
                weights = attempt_download(opt.weights)  # download if not found locally
            ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
            model_yolo = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=opt.nc, anchors=opt.hyp.get('anchors')).to(opt.device)  # create
            exclude = ['anchor'] if (opt.cfg or opt.hyp.get('anchors')) and not opt.resume else []  # exclude keys
            csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            csd = intersect_dicts(csd, model_yolo.state_dict(), exclude=exclude)  # intersect
            model_yolo.load_state_dict(csd, strict=False)  # load
            LOGGER.info(f'Transferred {len(csd)}/{len(model_yolo.state_dict())} items from {weights}')  # report
        else:
            model_yolo = Model(opt.cfg, ch=3, nc=opt.nc, anchors=opt.hyp.get('anchors')).to(opt.device)  # create
        self.model_yolo = model_yolo
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def forward(self, x, char_mode):
        """ Base network """
        y, feature = self.craft(x)
        if char_mode == False:
            regionscores_yolo = y[:, :, :, 0] * 255
            regionscores_yolo.unsqueeze_(1)
            regionscores_yolo = regionscores_yolo.repeat(1, 3, 1, 1)
            #regionscores_yolo = list(image.to(self.device) for image in regionscores_yolo)

            y_yolo1, y_yolo2 = self.model_yolo(regionscores_yolo)
        else:
            return y,  feature

        return y, y_yolo1,y_yolo2, feature
