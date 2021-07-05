import os

import cv2
import numpy as np
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torchvision.transforms import transforms
from config import just_config
from models.swin_transformer import SwinTransformer
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import PIL.Image as Im
cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else  'cpu'
config = just_config()
model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                        patch_size=config.MODEL.SWIN.PATCH_SIZE,
                        in_chans=config.MODEL.SWIN.IN_CHANS,
                        num_classes=config.MODEL.NUM_CLASSES,
                        embed_dim=config.MODEL.SWIN.EMBED_DIM,
                        depths=config.MODEL.SWIN.DEPTHS,
                        num_heads=config.MODEL.SWIN.NUM_HEADS,
                        window_size=config.MODEL.SWIN.WINDOW_SIZE,
                        mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                        qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                        qk_scale=config.MODEL.SWIN.QK_SCALE,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        ape=config.MODEL.SWIN.APE,
                        patch_norm=config.MODEL.SWIN.PATCH_NORM,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT)


state_dicts = torch.load('../ckpt_epoch_81.pth')['model']
model.load_state_dict(state_dicts)
# print(model)
model.to(device)
process_img = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD),
])
classes_name = {0:'中国风', 1:'人物类', 2:'佩兹利', 3:'其他', 4:'几何', 5:'动物', 6:'动物纹理', 7:'叶子', 8:'大花朵', 9:'字符', 10:'小碎花', 11:'抽象', 12:'条纹',
                13:'格子', 14:'民族风', 15:'水果与花', 16:'波点', 17:'海洋风', 18:'热带风', 19:'组合类花卉', 20:'自然纹理', 21:'节日', 22:'装饰纹样', 23:'豹纹', 24:'迷彩'}


img_dir = 'E:/test/'

model.eval()
with torch.no_grad():
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir,img_name)
        img = cv2.imread(img_path)
        img = Im.fromarray(img)
        img = process_img(img)
        img = img.unsqueeze(0).to(device)
        out = model(img).cpu().numpy()
        # print(out.shape)
        index = np.argmax(out)
        print(img_name)
        print('predict the class of this pic is:{}'.format(classes_name[index]))