import numpy as np
from scipy.spatial.transform import Rotation


def pos_to_hom(pos):
    mat = np.eye(4)
    mat[:3, 3] = pos[:3]
    return mat


def quat_to_hom(quat):
    rot = Rotation.from_quat(quat)
    mat = np.eye(4)
    mat[:3, :3] = rot.as_matrix()
    return mat


def hom_to_pos(mat):
    return mat[:3, 3].copy()


def hom_to_quat(mat):
    rot = Rotation.from_matrix(mat[:3, :3])
    quat = rot.as_quat()
    return quat


def pos_quat_to_hom(pos, quat):
    trans_mat = pos_to_hom(pos)
    quat_mat = quat_to_hom(quat)
    return np.matmul(trans_mat, quat_mat)


def project(world_pos_quat, view_mat, proj_mat):
    world_T_p = pos_quat_to_hom(*world_pos_quat)
    cam_T_world = view_mat
    cam_T_p = cam_T_world @ world_T_p
    cam_p = cam_T_p[:3, 3]
    cam_p_hom = cam_p / cam_p[2]
    pix_p = proj_mat @ cam_p_hom.T
    pix_p = pix_p[:2].astype(int)
    return pix_p
