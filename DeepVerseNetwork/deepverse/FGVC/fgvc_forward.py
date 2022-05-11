# ============================================
# Evaluation of Attention-Driven FGVC network
# ============================================
import os
import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from deepverse.FGVC import fgvcnet_config
from deepverse.FGVC.models import WSDAN
from deepverse.FGVC.utils import TopKAccuracyMetric, batch_augment
from deepverse.FGVC import model_id_correspondences

# GPU settings
assert torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = fgvcnet_config.GPU
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

def FGVC_net_forward(img, furniture_class):
    #print('entering FGVC_net, furniture_class = ', furniture_class)
    if furniture_class == 1:
        # -- bed --
        model_ckpt = fgvcnet_config.furniture_model_dir + 'bed_model.ckpt'
        num_of_categories = 7
        CAD_class = '2818832'

    elif furniture_class == 2:
        # -- bin --
        model_ckpt = fgvcnet_config.furniture_model_dir + 'bin_model.ckpt'
        num_of_categories = 9
        CAD_class = '2747177'

    elif furniture_class == 3:
        # -- bookcase --
        model_ckpt = fgvcnet_config.furniture_model_dir + 'bookcase_model.ckpt'
        num_of_categories = 8
        CAD_class = '2871439'

    elif furniture_class == 4:
        # -- chair --
        model_ckpt = fgvcnet_config.furniture_model_dir + 'chair_model.ckpt'
        num_of_categories = 11
        CAD_class = '3001627'

    elif furniture_class == 5:
        # -- cabinet --
        model_ckpt = fgvcnet_config.furniture_model_dir + 'cabinet_model.ckpt'
        num_of_categories = 5
        CAD_class = '2933112'

    elif furniture_class == 6:
        # -- display --
        model_ckpt = fgvcnet_config.furniture_model_dir + 'display_model.ckpt'
        num_of_categories = 6
        CAD_class = '3211117'

    elif furniture_class == 7:
        # -- sofa --
        model_ckpt = fgvcnet_config.furniture_model_dir + 'sofa_model.ckpt'
        num_of_categories = 7
        CAD_class = '4256520'

    elif furniture_class == 8:
        # -- table --
        model_ckpt = fgvcnet_config.furniture_model_dir + 'table_model.ckpt'
        num_of_categories = 8
        CAD_class = '4379243'

    # -- instantiate the attention-driven FGVC network --
    attention_fgvc_net = WSDAN(num_classes=num_of_categories, M=fgvcnet_config.num_attentions, net=fgvcnet_config.net)

    # -- load ckpt and get state_dict --
    checkpoint = torch.load(model_ckpt)
    state_dict = checkpoint['state_dict']

    # -- load weights --
    attention_fgvc_net.load_state_dict(state_dict)
    
    # -- use cuda gpu --
    '''
    attention_fgvc_net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    '''
    
    # -- evaluate and make prediction --
    attention_fgvc_net.eval()
    with torch.no_grad():
        # -- make the object image a PyTorch tensor --
        transform = transforms.ToTensor()
        img_tensor = transform(img)

        #img = img.to(device)

        # -- feed to the network --
        y_pred_category, _, _ = attention_fgvc_net(img_tensor)
    
    # -- retrieve the corresponding 3D model ids --
    #print(y_pred_category)

    cad_id = model_ID_retrieval(y_pred_category)

    return CAD_class, cad_id