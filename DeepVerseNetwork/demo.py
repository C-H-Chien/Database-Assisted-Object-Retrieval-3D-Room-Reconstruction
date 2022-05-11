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
from skimage.transform import resize
from deepverse.FGVC import FGVC_net_forward


def main(args):
    predictor = Predictor(
        data_dir=args.data_dir,
        model_path=args.model_path,
        config_path=args.config_path
    )    

    # -- loop over multiple images, if needed --
    for name, scene in zip(
        #('3m', 'sofa', 'lab', 'desk'),
        #('scene0474_02', 'scene0207_00', 'scene0378_02', 'scene0474_02')
        #('sofa2', 'sofa'),
        #('scene0423_01', 'scene0207_00')
        ('3m', 'desk'),
        ('scene0474_02', 'scene0474_02')
    ):
        
        # -- 1) read image and scene, feed to MaskRCNN, and get instances and annotated cad_ids back --
        #print(name, scene)
        
        img = Image.open(os.path.join('assets', '{}.jpg'.format(name)))
        img = np.asarray(img)
        #instances, cad_ids = predictor(img, scene=scene)
        instances, _ = predictor(img, scene=scene)

        # -- 1-1) process output of MaskRCNN for the preparation of 3d coarse reconstruction block --
        #print(instances, cad_ids)        
        proccessor = ProcessObj(name, img, instances)
        # -- loop over all 2d objects --
        CAD_objects = []
        for obj_idx in range(len(instances)):
            cropped_obj_img = proccessor.mask_rcnn_output_processing(obj_idx)

            #if name == 'books':
            #    cad_ids_list = list(cad_ids[0])
            #    cad_ids_list[1] = '9368cd9028151e1e9d51a07a5989d077'
            #    cad_ids[0] = tuple(cad_ids_list)

            # -- ignore the dummy furniture --
            if int(instances.pred_classes[obj_idx]) == 0:
                continue

            # -- 2) do fine-grained classification to predict the cad_ids --
            cad_class, cad_id = FGVC_net_forward(cropped_obj_img, int(instances.pred_classes[obj_idx]))
            CAD_objects.append((cad_class, cad_id))
            
        
        # -- 3) retrieve corresponding 3D object mesh from database based on CAD IDs --
        meshes = predictor.retrieve_obj_mesh(
            instances,
            #cad_ids,
            CAD_objects,
            excluded_classes= ()
        )

        # -- 4) visualize the result --
        o3d.visualization.draw_geometries(meshes)
        
    

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--config_path', required=True)
    args = parser.parse_args(sys.argv[1:])
    main(args)
    
