from typing import Any, List, Iterable, Tuple, Union

import numpy as np
import torch
import trimesh

from detectron2.modeling import build_model
from detectron2.structures import Instances

from deepverse.config import deepverse_config
from deepverse.data import CADCatalog
from deepverse.data.constants import CAD_TAXONOMY, COLOR_BY_CLASS
from deepverse.data.datasets import register_scan2cad
from deepverse.structures import Intrinsics
from deepverse.utils.alignment_errors import translation_diff
from deepverse.utils.linalg import make_M_from_tqs
from deepverse.utils.linalg import mesh_transform

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

class Predictor:
    def __init__(
        self,
        data_dir: str,
        model_path: str,
        config_path: str,
        thresh: float = 0.5
    ):
        # -- fetch system configurations --
        cfg = deepverse_config('Scan2CAD', 'Scan2CAD')
        cfg.merge_from_file(config_path)
        cfg.MODEL.INSTANCES_CONFIDENCE_THRESH = thresh

        # -- fetch the best model weights from the MaskRCNN model --
        model = build_model(cfg)
        backup = torch.load(model_path)
        model.load_state_dict(backup['model'])

        # -- do model inference --
        model.eval()
        model.requires_grad_(False)

        print('\nDone building Mask-RCNN predictor\n')

        data_name = 'Scan2CADVal'
        # -- appears in [datasets.py] --
        register_scan2cad(data_name, {}, '', data_dir, '', '', 'val')

        # -- fetch data from database and preprocess it --
        cad_manager = CADCatalog.get(data_name)
        
        # -- get point cloud of 3D CAD objects and the corresponding labels (ids) --
        # -- Notes: 1) points: point cloud of all 3d objects, len(points)=9, sampled from volumes of 3D objects --
        # --        2) ids: a tuple (object class, object id) --
        points, ids = cad_manager.batched_points_and_ids(volumes=True)

        #print(len(points), len(ids))
        #print(points[5][125], np.shape(points[5][125]))
        #for i in range(5):
        #    print(len(points[5][i+120]))

        #pt_cloud = points[5][125]
        #x = [item[0] for item in pt_cloud]
        #y = [item[1] for item in pt_cloud]
        #z = [item[2] for item in pt_cloud]
        #fig = plt.figure(figsize=(8, 8))
        #ax = fig.add_subplot(111, projection='3d')

        #ax.scatter(x, y, z)
        #plt.show()

        model.set_cad_models(points, ids, cad_manager.scene_alignments)
        model.embed_cad_models()


        self.model = model
        self.cad_manager = cad_manager

        with open('./assets/camera.obj') as f:
            cam = trimesh.load(f, file_type='obj', force='mesh')
        cam.apply_scale(0.25)
        cam.visual.face_colors = [100, 100, 100, 255]
        self._camera_mesh = cam

        self.scene_rot = np.array([
            [1, 0, 0, 0],
            [0, np.cos(np.pi), -np.sin(np.pi), 0],
            [0, np.sin(np.pi), np.cos(np.pi), 0],
            [0, 0, 0, 1]
        ])

    """
    @property
    def can_render(self):
        return Rasterizer is not None
    """

    @torch.no_grad()
    def __call__(
        self,
        image_rgb: np.ndarray,
        f: Union[np.ndarray, float] = 435.,
        scene: str = 'scene0474_02'
    ) -> Tuple[Instances, List[Tuple[str, str]]]:

        # -- grab the scene --
        inputs = {'scene': scene}
        # -- grab the image as a numpy array --
        inputs['image'] = torch.as_tensor(
            np.ascontiguousarray(image_rgb[:, :, ::-1].transpose(2, 0, 1))
        )
        # -- grab camera intrinsic parameters --
        if isinstance(f, np.ndarray):
            inputs['intrinsics'] = f[:3, :3]
        else:
            inputs['intrinsics'] = Intrinsics(torch.tensor([
                [f, 0., image_rgb.shape[1] / 2],
                [0., f, image_rgb.shape[0] / 2],
                [0., 0., 1.]
            ]))
        outputs = self.model([inputs])[0]
        cad_ids = outputs['cad_ids']
        return outputs['instances'].to('cpu'), cad_ids


    # -- Edited by Chiang-Heng --
    # -- retrieve mesh 3D objects from database according to the cad_ids --
    def retrieve_obj_mesh(
        self,
        instances: Instances,
        cad_ids: List[Tuple[str, str]],
        min_dist_3d: float = 0.4,
        excluded_classes: Iterable[str] = (),
        as_open3d: bool = True
    ) -> Union[List[trimesh.Trimesh], List[Any]]:

        meshes = []
        trans_cls_scores = []

        # -- loop over all objects and instances, each of which retrieves CAD models --
        for i in range(len(instances)):
            cad_id = cad_ids[i]
            if cad_id is None:
                continue
            
            # -- retrieve the first hierarchy --
            if CAD_TAXONOMY[int(cad_id[0])] in excluded_classes:
                continue

            # -- class scores obtained from Mask-RCNN --
            trans_cls_scores.append((
                instances.pred_translations[i],
                instances.pred_classes[i].item(),
                instances.scores[i].item(),
            ))

            # -- retrieve meshes from the database --
            mesh = self.cad_manager.model_by_id(*cad_ids[i])
            
            # -- create mesh represented objects by python3d supported trimesh --
            mesh = trimesh.Trimesh(
                vertices=mesh.verts_list()[0].numpy(),
                faces=mesh.faces_list()[0].numpy()
            )

            # -- [DEFINE OBJECT POSE 1] user-defined object pose: translation, rotation, scale --
            '''
            obj_eul_angles = [0.1, 0.2, 0.3]
            trs = mesh_transform(
                instances.pred_translations[i].tolist(),
                obj_eul_angles,
                #instances.pred_rotations[i].tolist(),
                instances.pred_scales[i].tolist()
            )
            '''

            # -- [DEFINE OBJECT POSE 2] get pose estimated by ROCA: translation, rotation, scale --
            translation = instances.pred_translations[i].tolist()
            
            trs = make_M_from_tqs(
                #instances.pred_translations[i].tolist(),
                translation,
                instances.pred_rotations[i].tolist(),
                instances.pred_scales[i].tolist()
            )

            # -- apply the object pose to the 3D CAD models in the scene --
            mesh.apply_transform(self.scene_rot @ trs)
            

            color = COLOR_BY_CLASS[int(cad_ids[i][0])]
            if as_open3d:
                mesh = mesh.as_open3d
                mesh.paint_uniform_color(color)
                mesh.compute_vertex_normals()
            else:
                mesh.visual.face_colors = [*(255 * color), 255]
            meshes.append(mesh)

        return meshes

    def output_to_mesh(
        self,
        instances: Instances,
        cad_ids: List[Tuple[str, str]],
        min_dist_3d: float = 0.4,
        excluded_classes: Iterable[str] = (),
        nms_3d: bool = True,
        as_open3d: bool = False
    ) -> Union[List[trimesh.Trimesh], List[Any]]:

        meshes = []
        trans_cls_scores = []
        for i in range(len(instances)):
            cad_id = cad_ids[i]
            if cad_id is None:
                continue
            
            # -- retrieve the first hierarchy --
            if CAD_TAXONOMY[int(cad_id[0])] in excluded_classes:
                continue

            trans_cls_scores.append((
                instances.pred_translations[i],
                instances.pred_classes[i].item(),
                instances.scores[i].item(),
            ))

            mesh = self.cad_manager.model_by_id(*cad_ids[i])
            mesh = trimesh.Trimesh(
                vertices=mesh.verts_list()[0].numpy(),
                faces=mesh.faces_list()[0].numpy()
            )

            trs = make_M_from_tqs(
                instances.pred_translations[i].tolist(),
                instances.pred_rotations[i].tolist(),
                instances.pred_scales[i].tolist()
            )
            mesh.apply_transform(self.scene_rot @ trs)

            color = COLOR_BY_CLASS[int(cad_ids[i][0])]
            if as_open3d:
                mesh = mesh.as_open3d
                mesh.paint_uniform_color(color)
                mesh.compute_vertex_normals()
            else:
                mesh.visual.face_colors = [*(255 * color), 255]
            meshes.append(mesh)

        if nms_3d:
            keeps = self._3d_nms(trans_cls_scores, min_dist_3d)
            meshes = [m for m, b in zip(meshes, keeps) if b]

        if as_open3d:
            cam = self._camera_mesh
            cam = cam.as_open3d
            cam.compute_vertex_normals()
            meshes.append(cam)
        else:
            meshes.append(self._camera_mesh)

        return meshes

    @staticmethod
    def _3d_nms(tcs, min_dist):
        keeps = [True for _ in tcs]
        if min_dist <= 0:
            return keeps
        for i, (t, c, s) in enumerate(tcs):
            if any(
                c_ == c and s_ > s and translation_diff(t_, t) < min_dist
                for t_, c_, s_ in tcs
            ):
                keeps[i] = False
        return keeps
