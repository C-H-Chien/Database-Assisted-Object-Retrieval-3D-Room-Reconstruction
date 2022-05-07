import os
import sys

import argparse
import numpy as np
import open3d as o3d
from PIL import Image
from trimesh.exchange.export import export_mesh
from trimesh.util import concatenate as stack_meshes
from matplotlib import pyplot as plt
import torch
import torchvision

from deepverse.engine import Predictor
from deepverse.structures import ProcessObj
from deepverse.data.annotate_FGCnet_data import mask_obj_annotater
from skimage.transform import resize

import json



def read_json_file(args):
    name = 'scan2cad_instances_val'

    json_file_dir = args.data_dir
    f = open(os.path.join(json_file_dir, '{}.json'.format(name)))
    data = json.load(f)
    
    return data
    #for category_dict in data["categories"]:
        #print(category_dict["name"])
        #print(type(category_dict["id"]))
    
    #img_files_dir = args.imgs_dir

    """
    test_counter = 0
    for imgs_dict in data["images"]:
        test_counter += 1
        if test_counter == 1:
            print(imgs_dict["file_name"])
            fetch_img = Image.open(os.path.join(img_files_dir, 'Images', imgs_dict["file_name"]))
            np_img = np.asarray(fetch_img)
            plt.imshow(np_img, interpolation='nearest')
            plt.show()
        else:
            break
    """


    



def main(args):
    json_data = read_json_file(args)

    # -- initiate the predictor model --
    predictor = Predictor(
        data_dir=args.data_dir,
        model_path=args.model_path,
        config_path=args.config_path
    )

    # -- read images --
    test_counter = 0
    for imgs_dict in json_data["images"]:
        test_counter += 1
        if test_counter > 890: 
            fetch_img = Image.open(os.path.join(args.imgs_dir, 'Images', imgs_dict["file_name"]))
            fetch_scene = imgs_dict["scene_id"]

            img = np.asarray(fetch_img)
            instances, cad_ids = predictor(img, scene=fetch_scene)
            #print(instances, cad_ids)

            if (instances is not None) and (cad_ids is not None):
                proccessor = mask_obj_annotater(img, instances, cad_ids)
                proccessor.annotate_objs(save_obj_img = True)
                print(fetch_scene)
            #np_img = np.asarray(fetch_img)
            #plt.imshow(np_img, interpolation='nearest')
            #plt.show()
    
    print("Done!")

    """
    predictor = Predictor(
        data_dir=args.data_dir,
        model_path=args.model_path,
        config_path=args.config_path
    )    

    # -- loop over multiple images, if needed --
    for name, scene in zip(
        #('3m', 'sofa', 'lab', 'desk'),
        #('scene0474_02', 'scene0207_00', 'scene0378_02', 'scene0474_02')
        ('3m', 'lab'),
        ('scene0474_02', 'scene0207_00')
    ):
        
        # -- 1) read image and scene, feed to MaskRCNN, and get instances and annotated cad_ids back --
        print(name, scene)
        
        img = Image.open(os.path.join('assets', '{}.jpg'.format(name)))
        img = np.asarray(img)
        instances, cad_ids = predictor(img, scene=scene)

        # -- 1-1) process output of MaskRCNN for the preparation of 3d coarse reconstruction block --
        print(instances, cad_ids)
        proccessor = ProcessObj(name, img, instances)
        proccessor.mask_rcnn_output_processing()
    """


if __name__ == '__main__':
    #torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--imgs_dir', required=True)
    args = parser.parse_args(sys.argv[1:])
    main(args)
    
