import cv2
import numpy as np
from numba import njit, prange
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import ConvexHull

voxel_origin = np.array([0, -25.6, -2])
voxel_size = 0.2
image_width = 1220
image_height = 370
scene_size = (51.2, 51.2, 6.4)


def set_meta_info(_voxel_origin, _voxel_size, _image_width, _image_height, _scene_size):
    global voxel_origin, voxel_size, image_width, image_height, scene_size
    voxel_origin = _voxel_origin
    voxel_size = _voxel_size
    image_width = _image_width
    image_height = _image_height
    scene_size = _scene_size


@njit(parallel=True)
def vox2world(vol_origin, vox_coords, vox_size, offsets=(0.5, 0.5, 0.5)):
    """Convert voxel grid coordinates to world coordinates."""
    vol_origin = vol_origin.astype(np.float32)
    vox_coords = vox_coords.astype(np.float32)
    cam_pts = np.empty_like(vox_coords, dtype=np.float32)

    for i in prange(vox_coords.shape[0]):
        for j in range(3):
            cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j]) + vox_size * offsets[j]
    return cam_pts


@njit(parallel=True)
def cam2pix(cam_pts, intr):
    intr = intr.astype(np.float32)
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
    for i in prange(cam_pts.shape[0]):
        pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
        pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
    return pix


@njit(parallel=True)
def pix2cam(pix, depth, intr):
    intr = intr.astype(np.float32)
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    cam_pts = np.empty((pix.shape[0], 3), dtype=np.float32)
    for i in prange(pix.shape[0]):
        cam_pts[i, 0] = (pix[i, 0] - cx) * depth[i] / fx
        cam_pts[i, 1] = (pix[i, 1] - cy) * depth[i] / fy
        cam_pts[i, 2] = depth[i]
    return cam_pts


def rigid_transform(xyz, transform):
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]


def vox2pix(calib, vox_coords=None):
    cam_E = calib["Tr"]
    cam_k = calib["P2"]

    if vox_coords is None:
        # Compute the x, y, z bounding of the scene in meter
        vol_bnds = np.zeros((3, 2))
        vol_bnds[:, 0] = voxel_origin
        vol_bnds[:, 1] = voxel_origin + np.array(scene_size)
        # Compute the voxels centroids in lidar cooridnates
        vol_dim = np.ceil((vol_bnds[:, 1] - vol_bnds[:, 0]) / voxel_size).copy(order="C").astype(int)
        xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing="ij")
        vox_coords = np.concatenate([xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)], axis=0).astype(int).T

    # Project voxels'centroid from lidar coordinates to camera coordinates
    cam_pts = vox2world(voxel_origin, vox_coords, voxel_size)
    cam_pts = rigid_transform(cam_pts, cam_E)

    # Project camera coordinates to pixel positions
    projected_pix = cam2pix(cam_pts, cam_k)
    pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]

    # Eliminate pixels outside view frustum
    pix_z = cam_pts[:, 2]
    fov_mask = np.logical_and(pix_x >= 0, np.logical_and(pix_x < image_width, np.logical_and(pix_y >= 0, np.logical_and(pix_y < image_height, pix_z > 0))))

    return projected_pix, fov_mask, pix_z, vox_coords


def vox2pix2(calib, vox_coords):
    cam_E = calib["Tr"]
    cam_k = calib["P2"]

    # Project voxels'centroid from lidar coordinates to camera coordinates
    cam_pts = []
    cam_pts.extend(vox2world(voxel_origin, vox_coords, voxel_size, (0, 0, 0)))
    cam_pts.extend(vox2world(voxel_origin, vox_coords, voxel_size, (0, 0, 1)))
    cam_pts.extend(vox2world(voxel_origin, vox_coords, voxel_size, (0, 1, 0)))
    cam_pts.extend(vox2world(voxel_origin, vox_coords, voxel_size, (0, 1, 1)))
    cam_pts.extend(vox2world(voxel_origin, vox_coords, voxel_size, (1, 0, 0)))
    cam_pts.extend(vox2world(voxel_origin, vox_coords, voxel_size, (1, 0, 1)))
    cam_pts.extend(vox2world(voxel_origin, vox_coords, voxel_size, (1, 1, 0)))
    cam_pts.extend(vox2world(voxel_origin, vox_coords, voxel_size, (1, 1, 1)))
    cam_pts = rigid_transform(cam_pts, cam_E)

    # Project camera coordinates to pixel positions
    projected_pix = cam2pix(cam_pts, cam_k)
    pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]

    # Eliminate pixels outside view frustum
    pix_z = cam_pts[:, 2]
    fov_mask = np.logical_and(pix_x >= 0, np.logical_and(pix_x < image_width, np.logical_and(pix_y >= 0, np.logical_and(pix_y < image_height, pix_z > 0))))

    return projected_pix, fov_mask, pix_z


def pix2vol(calib, p2d, z):
    cam_E = calib["Tr"]
    cam_k = calib["P2"]
    p3d_cam = pix2cam(p2d, z, cam_k)
    p3d_world = rigid_transform(p3d_cam, cam_E.T)
    p3d_vox = (p3d_world - voxel_origin) / voxel_size
    return p3d_vox


def griddata(points, values, xi, rescale=False):
    ip = NearestNDInterpolator(points, values, rescale=rescale)
    return ip(xi)


def get_can_see(label, calib):
    projected_pix, fov_mask, pix_z, vox_coords = vox2pix(calib)
    vox_coords = vox_coords[fov_mask]
    fov_mask_label = label[vox_coords[:, 0], vox_coords[:, 1], vox_coords[:, 2]]
    label_mask = (fov_mask_label != 0) & (fov_mask_label != 255)
    vox_coords = vox_coords[label_mask]
    pix_z = pix_z[fov_mask]
    pix_z = pix_z[label_mask]
    projected_pix, _, _ = vox2pix2(calib, vox_coords)
    projected_vox = projected_pix.reshape((8, -1, 2))
    projected_vox = projected_vox.transpose((1, 0, 2))

    near2far = np.argsort(pix_z)
    depth = np.zeros([image_height, image_width], dtype=np.float32)
    can_see = []
    for i in near2far:
        z = pix_z[i]
        points = projected_vox[i].astype(np.int32)
        hull = ConvexHull(points)
        vertex = points[hull.vertices]
        new_image = np.zeros_like(depth)
        cv2.fillPoly(new_image, [vertex], color=z)
        new_image = new_image * (depth == 0)
        if new_image.sum() > 0:
            depth = depth + new_image
            can_see.append(i)

    can_see_coords = vox_coords[can_see]
    can_see_label = np.zeros_like(label)
    can_see_label[can_see_coords[:, 0], can_see_coords[:, 1], can_see_coords[:, 2]] = 1
    return can_see_label.astype(bool)
