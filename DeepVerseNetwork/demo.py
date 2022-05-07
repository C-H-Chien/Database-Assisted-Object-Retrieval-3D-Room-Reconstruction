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
        ('3m', 'sofa'),
        ('scene0474_02', 'scene0207_00')
    ):
        
        # -- 1) read image and scene, feed to MaskRCNN, and get instances and annotated cad_ids back --
        #print(name, scene)
        
        img = Image.open(os.path.join('assets', '{}.jpg'.format(name)))
        img = np.asarray(img)
        instances, cad_ids = predictor(img, scene=scene)

        # -- 1-1) process output of MaskRCNN for the preparation of 3d coarse reconstruction block --
        #print(instances, cad_ids)
        print(cad_ids)
        proccessor = ProcessObj(name, img, instances)
        proccessor.mask_rcnn_output_processing(save_obj_img = True)

        """
        print(cad_ids[0][1])
        cad_ids_list = list(cad_ids[0])
        cad_ids_list[1] = 'a97bd9998d12467d7275456565149f4f'
        cad_ids[0] = tuple(cad_ids_list)
        """

        #print("=========================================")
        #print(instances.pred_rotations[0].tolist())

        # -- 2) build coarse 3D reconstruction --

        # -- 3) embed the 3D object reconstuction --

        """
        # -- 4) retrieve corresponding 3D object mesh from database --
        meshes = predictor.retrieve_obj_mesh(
            instances,
            cad_ids,
            excluded_classes= ()
        )

        # -- 5) visualize the result --
        o3d.visualization.draw_geometries(meshes)
        """
        


if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--config_path', required=True)
    args = parser.parse_args(sys.argv[1:])
    main(args)
    
