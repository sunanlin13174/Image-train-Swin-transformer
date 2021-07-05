import torch
import torch.nn as nn
import cv2
from models import swin_transformer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = swin_transformer.SwinTransformer()
# print(net)
pre_state_dict = torch.load('swin_tiny_patch4_window7_224.pth',map_location='cpu')

# for k,v in pre_state_dict.items():
#     print(k)
#     for k1,v1 in v.items():
#         print(k1)

# print(pre_state_dict)
net.load_state_dict(pre_state_dict['model'])
net.to(device)

