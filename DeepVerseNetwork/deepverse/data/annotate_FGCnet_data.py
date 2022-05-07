import os
import sys

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torch
import torchvision
from deepverse.data.constants import CAD_TAXONOMY

"""
===================== annotate_FGCnet_data.py ========================


Author: Chiang-Heng Chien
======================================================================
"""

class mask_obj_annotater():
    def __init__(self, input_img, maskrcnn_instances, cad_ids):
        self.img = input_img
        self.instances = maskrcnn_instances
        self.cad_ids = cad_ids
        self.obj_classes = self.instances.pred_classes.numpy()
        self.obj_boxes = self.instances.pred_boxes.tensor.numpy()
        self.obj_masks = self.instances.pred_masks
    
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

    def masking_objs(self, obj_idx):        
        # -- fetch data of each object --
        obj_class_label = self.obj_classes[obj_idx]
        obj_box = self.obj_boxes[obj_idx]
        #obj_mask_bool = self.obj_masks[obj_idx].type(torch.uint8).numpy()
        #extract_obj_img = self.img*obj_mask_bool[:,:,None]
        #extract_obj_img = np.where(extract_obj_img.any(-1,keepdims=True), extract_obj_img, 255)
        pil_img = Image.fromarray(self.img, 'RGB')
        cropped_img = mask_obj_annotater.crop_object(pil_img, obj_box)
        cropped_img = cropped_img.resize((256, 256), Image.ANTIALIAS)

        return cropped_img
    
    def save_obj_crop_img(img, class_folder_name, id_folder_name, dir, file_ext = '.png'):
        #save_dir = os.path.join(dir, class_folder_name, id_folder_name, img_filename + file_ext)
        save_dir = os.path.join(dir, class_folder_name, id_folder_name)
        #print(save_dir)
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
                print("created path %s", save_dir)
            except OSError:
                print("Creation of the directory %s failed" % save_dir)
        
        filenum = mask_obj_annotater.count_num_files(save_dir)
        filenum += 1
        save_dir = os.path.join(dir, class_folder_name, id_folder_name, str(filenum) + file_ext)
        img.save(save_dir)

        # -- also make the image flipped horizontally --
        img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_flipped.resize((256, 256), Image.ANTIALIAS)
        #white = (255,255,255)
        #im1_rot_cw = img.rotate(20, Image.NEAREST, fillcolor = white)
        #im1_rot_cw.resize((256, 256), Image.ANTIALIAS)
        #im1_rot_ccw = img.rotate(-20, Image.NEAREST, fillcolor = white)
        #im1_rot_ccw.resize((256, 256), Image.ANTIALIAS)
        #im_flp_rot_cw = img_flipped.rotate(20, Image.NEAREST, fillcolor = white)
        #im_flp_rot_cw.resize((256, 256), Image.ANTIALIAS)
        #im_flp_rot_ccw = img_flipped.rotate(-20, Image.NEAREST, fillcolor = white)
        #im_flp_rot_ccw.resize((256, 256), Image.ANTIALIAS)
        filenum += 1
        save_dir = os.path.join(dir, class_folder_name, id_folder_name, str(filenum) + file_ext)
        img_flipped.save(save_dir)
        #filenum += 1
        #save_dir = os.path.join(dir, class_folder_name, id_folder_name, str(filenum) + file_ext)
        #im1_rot_cw.save(save_dir)
        #filenum += 1
        #save_dir = os.path.join(dir, class_folder_name, id_folder_name, str(filenum) + file_ext)
        #im1_rot_ccw.save(save_dir)
        #filenum += 1
        #save_dir = os.path.join(dir, class_folder_name, id_folder_name, str(filenum) + file_ext)
        #im_flp_rot_cw.save(save_dir)
        #filenum += 1
        #save_dir = os.path.join(dir, class_folder_name, id_folder_name, str(filenum) + file_ext)
        #im_flp_rot_ccw.save(save_dir)
    
    def count_num_files(dir):
        initial_count = 0
        for path in os.listdir(dir):
            if os.path.isfile(os.path.join(dir, path)):
                initial_count += 1
        
        return initial_count

    def annotate_objs(self, save_obj_img = True):
        #print("=========== Annotating Masked Objects ============")
        save_idx = 0
        # -- loop over all objects in the image --
        for obj_idx in range(len(self.instances)):
            cropped_img = mask_obj_annotater.masking_objs(self, obj_idx)

            if self.cad_ids[obj_idx] is not None:
                associate_cad_class = self.cad_ids[obj_idx][0]
                associate_cad_id = self.cad_ids[obj_idx][1]

                if int(associate_cad_class) > 0:
                    class_name = CAD_TAXONOMY[int(associate_cad_class)]

                    if save_obj_img:
                        dir = '/home/chchien/BrownU/courses/Deep-Learning-2022/Final-Project/box_obj_imgs/'
                        class_folder_name = str(class_name)
                        img_filename = str(save_idx)
                        id_folder_name = str(associate_cad_id)
                        #store_dir = os.path.join(dir, class_folder_name, id_folder_name)

                        mask_obj_annotater.save_obj_crop_img(cropped_img, class_folder_name, id_folder_name, dir)
                        save_idx += 1


