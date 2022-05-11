import os
import sys

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torch
import torchvision

"""
===================== process_obj.py ========================
Feed input RGB image and instances from the MaskRCNN network,
do processings on object instances to prepare data for inputs
of the coarse 3D reconstruction network.

Author: Chiang-Heng Chien
=============================================================
"""

class ProcessObj():
    def __init__(self, img_name, input_img, maskrcnn_instances):
        self.img_name = img_name
        self.img = input_img
        self.instances = maskrcnn_instances
        self.obj_classes = self.instances.pred_classes.numpy()
        self.obj_boxes = self.instances.pred_boxes.tensor.numpy()
        self.obj_masks = self.instances.pred_masks

    def mask_rcnn_output_processing(self, obj_idx, save_obj_img = False):
        #print("================================")
        save_idx = 0
        # -- loop over all objects in the image --
        #for obj_idx in range(len(self.instances)):
        
        # -- fetch data of each object --
        obj_class_label = self.obj_classes[obj_idx]
        obj_box = self.obj_boxes[obj_idx]
        '''
        obj_mask_bool = self.obj_masks[obj_idx].type(torch.uint8).numpy()
        extract_obj_img = self.img*obj_mask_bool[:,:,None]
        extract_obj_img = np.where(extract_obj_img.any(-1,keepdims=True), extract_obj_img, 255)

        pil_img = Image.fromarray(extract_obj_img, 'RGB')
        cropped_img = ProcessObj.crop_object(pil_img, obj_box)
        cropped_img = cropped_img.resize((256, 256), Image.ANTIALIAS)
        '''
        # -- fetch data of each object --
        pil_img = Image.fromarray(self.img, 'RGB')
        cropped_img = ProcessObj.crop_object(pil_img, obj_box)
        cropped_img = cropped_img.resize((256, 256), Image.ANTIALIAS)
        

        #print(obj_class_label)
        #print(obj_box)
        if save_obj_img:
            #cropped_img.show()
            filename = self.img_name + '_' + str(obj_class_label) + '_' + str(save_idx)
            ProcessObj.save_obj_crop_img(cropped_img, filename)
            save_idx += 1
        
        return cropped_img

    def crop_object(image, box):
        # -- crop an object in an image using detectron2 pred_boxes --
        x_top_left = box[0]
        y_top_left = box[1]
        x_bottom_right = box[2]
        y_bottom_right = box[3]
        x_center = (x_top_left + x_bottom_right) / 2
        y_center = (y_top_left + y_bottom_right) / 2

        crop_img = image.crop((int(x_top_left), int(y_top_left), int(x_bottom_right), int(y_bottom_right)))
        return crop_img
    
    
    def save_obj_crop_img(img, filename, dir_folder = '/home/chchien/BrownU/courses/Deep-Learning-2022/Final-Project/ROCA-decompose/obj_extract_imgs/', file_ext = '.png'):
        save_dir = dir_folder + filename + file_ext
        img.save(save_dir)

