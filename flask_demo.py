#使用flask进行web部署
import os
import cv2
import numpy as np
import torch.nn as nn
import torch
from flask import request,Flask
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torchvision.transforms import transforms
from config import just_config
from models.swin_transformer import SwinTransformer
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import PIL.Image as Im
import json
cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else  'cpu'
###构造模型，可写成函数，不过麻烦

def create_model_and_data(weight_path,classes_name_dict=None):
    if classes_name_dict is None:
            classes_name = {0: '中国风', 1: '人物类', 2: '佩兹利', 3: '其他', 4: '几何', 5: '动物', 6: '动物纹理', 7: '叶子', 8: '大花朵', 9: '字符',
                            10: '小碎花', 11: '抽象', 12: '条纹',
                            13: '格子', 14: '民族风', 15: '水果与花', 16: '波点', 17: '海洋风', 18: '热带风', 19: '组合类花卉', 20: '自然纹理',
                            21: '节日', 22: '装饰纹样', 23: '豹纹', 24: '迷彩'}
    else:
        classes_name = classes_name_dict
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


    state_dicts = torch.load(weight_path)['model']
    model.load_state_dict(state_dicts)
    # print(model)
    model.to(device)
    process_img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD),
    ])
    model.eval()
    return model,process_img,classes_name          #model ,transform, name_dict

def inference(model,transform_process,classes_name,img_path):
    with torch.no_grad():
            img = cv2.imread(img_path)
            img = Im.fromarray(img)
            img = transform_process(img)
            img = img.unsqueeze(0).to(device)
            out = model(img).cpu().numpy()
            # print(out.shape)
            index = np.argmax(out)
            result = {'class_index':str(index),'pre_class':classes_name[index]}
            print('predict the class of this pic is:{}'.format(classes_name[index]))
            result = json.dumps(result,ensure_ascii=False)  ##第二个参数是显示中文，英文的话可以删除
            print(result)
            print(type(result))
    return result

app=  Flask(__name__)

@app.route('/',methods=['POST'])
def response():
    model,process_img,classes_name = create_model_and_data(weight_path='ckpt_epoch_81.pth',classes_name_dict=None)
    recept = json.loads(request.get_json())

    img_path = recept['img_path']
    result = inference(model,process_img,classes_name,img_path)
    return result


if __name__=="__main__":
   app.run('127.0.0.1',port=5000)