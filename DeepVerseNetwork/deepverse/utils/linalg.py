import numpy as np
import quaternion
import torch
import math

from pytorch3d.structures import Meshes

# -- edited by Chiang-Heng --
# -- takes in translation matrix, rotation matrix, scale, and the center of the 3D mesh objects --
def mesh_transform(t: list, r: list, s: list, center=None) -> np.ndarray:
    # -- translation matrix --
    T = np.eye(4)
    T[0:3, 3] = t
    # -- rotation matrix --
    R = np.eye(4)
    rotation_matrix = eul_angles_to_rotation(r)
    R[0:3, 0:3] = rotation_matrix
    # -- scale --
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    # -- object center coordinate --
    C = np.eye(4)
    if center is not None:
        C[0:3, 3] = center

    M = T.dot(R).dot(S).dot(C)
    return M

def eul_angles_to_rotation(eul_angles: list):
    eulx = eul_angles[0]
    euly = eul_angles[1]
    eulz = eul_angles[2]
    # -- yaw = rotz, pitch = roty, roll = rotx --
    yaw = [[math.cos(eulz), -math.sin(eulz), 0],[math.sin(eulz), math.cos(eulz), 0], [0, 0, 1]]
    pitch = [[math.cos(euly), 0, math.sin(euly)], [0, 1, 0], [-math.sin(euly), 0, math.cos(euly)]]
    roll = [[1, 0, 0],[0, math.cos(eulx), -math.sin(eulx)], [0, math.sin(eulx), math.cos(eulx)]]
    return np.matmul(np.matmul(yaw, pitch), roll)


def make_M_from_tqs(t: list, q: list, s: list, center=None) -> np.ndarray:
    if not isinstance(q, np.quaternion):
        q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    C = np.eye(4)
    if center is not None:
        C[0:3, 3] = center

    M = T.dot(R).dot(S).dot(C)
    return M


def decompose_mat4(M: np.ndarray) -> tuple:
    R = M[0:3, 0:3].copy()
    sx = np.linalg.norm(R[0:3, 0])
    sy = np.linalg.norm(R[0:3, 1])
    sz = np.linalg.norm(R[0:3, 2])

    s = np.array([sx, sy, sz])

    R[:, 0] /= sx
    R[:, 1] /= sy
    R[:, 2] /= sz

    q = quaternion.as_float_array(quaternion.from_rotation_matrix(R[0:3, 0:3]))
    # q = quaternion.from_float_array(quaternion_from_matrix(M, False))

    t = M[0:3, 3]
    return t, q, s


def transform_mesh(
    meshes: Meshes,
    t: list,
    q: np.quaternion,
    s: list
) -> Meshes:
    assert len(meshes) == 1, 'Batched evaluation is not supported yet'
    mat = make_M_from_tqs(t, q, s)
    pts = meshes.verts_list()[0].numpy()
    pts = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts = pts @ mat.T
    return Meshes([torch.from_numpy(pts[:, :3]).float()], meshes.faces_list())
