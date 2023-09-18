from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import *
from scipy.spatial import cKDTree
from voxel import vox2pix


def filter_by_density(voxel_np, radius, threshold):
    voxel = torch.from_numpy(voxel_np).reshape([1, 1, 256, 256, 32])
    nn_avgpool = nn.AvgPool3d(kernel_size=radius * 2 + 1, stride=1, padding=radius)
    avgpool = nn_avgpool(voxel.float())
    sumpool = avgpool * (radius * 2 + 1) ** 3
    sumpool[sumpool < threshold] = 0
    sumpool[(sumpool > threshold) | (sumpool == threshold)] = 1
    return (sumpool * voxel).bool().numpy().reshape(voxel_np.shape)


cls2threshold = {
    0: 5,  # "unlabeled", and others ignored
    1: 50,  # "car"
    2: 5,  # "bicycle"
    3: 5,  # "motorcycle"
    4: 50,  # "truck"
    5: 50,  # "other-vehicle"
    6: 5,  # "person"
    7: 5,  # "bicyclist"
    8: 5,  # "motorcyclist"
    9: 100,  # "road"
    10: 50,  # "parking"
    11: 50,  # "sidewalk"
    12: 50,  # "other-ground"
    13: 50,  # "building"
    14: 5,  # "fence"
    15: 50,  # "vegetation"
    16: 20,  # "trunk"
    17: 100,  # "terrain"
    18: 5,  # "pole"
    19: 5,  # "traffic-sign"
}


def cluster_by_distance(label, points: np.ndarray, threshold=20):
    n = points.shape[0]
    kdtree = cKDTree(points)
    near_is = kdtree.query_ball_tree(kdtree, threshold)
    remainder = set(list(range(n)))
    clusters = []
    for i in range(n):
        if i not in remainder:
            continue
        cur_set = set()
        cur_queue = [i]
        while len(cur_queue) > 0:
            q = cur_queue.pop(0)
            cur_set.add(q)
            for j in near_is[q]:
                if j not in cur_set and j in remainder:
                    cur_queue.append(j)
                    cur_set.add(j)
        remainder -= cur_set
        if len(cur_set) > cls2threshold[label]:
            clusters.append(points[list(cur_set)])
    return clusters


def voxel2obj(voxel: np.ndarray, can_see: np.ndarray, calib):
    # filter by density
    unique_labels = set(np.unique(voxel))
    unique_labels -= set([0, 255])
    mask = np.zeros_like(voxel, dtype=bool)
    for label in unique_labels:
        label_voxel = deepcopy(voxel)
        label_voxel[~(label_voxel == label)] = 0
        label_voxel[(label_voxel == label)] = 1
        label_mask = filter_by_density(label_voxel, 2, 10)
        mask = mask | label_mask
    voxel[mask == 0] = 0

    obj_labels = []
    obj_voxels = []
    point_idxes = np.argwhere(~(voxel == 255) & ~(voxel == 0))
    labels = voxel[point_idxes[:, 0], point_idxes[:, 1], point_idxes[:, 2]].astype(np.int32)
    for label in unique_labels:
        label_clusters = point_idxes[np.isclose(labels, label)].astype(np.float32)
        clusters = cluster_by_distance(label, label_clusters, 5)
        for c_ps in clusters:
            c_ps = c_ps.astype(np.int32)
            c_ps[:, 0] = np.clip(c_ps[:, 0], 0, 255)
            c_ps[:, 1] = np.clip(c_ps[:, 1], 0, 255)
            c_ps[:, 2] = np.clip(c_ps[:, 2], 0, 255)
            c_can_see = can_see[c_ps[:, 0], c_ps[:, 1], c_ps[:, 2]]
            c_pix, _, _, _ = vox2pix(calib, c_ps[c_can_see])
            if c_can_see.sum() < 5:
                continue

            minx2d = np.min(c_pix[:, 0])
            maxx2d = np.max(c_pix[:, 0])
            miny2d = np.min(c_pix[:, 1])
            maxy2d = np.max(c_pix[:, 1])
            minx3d = np.min(c_ps[:, 0])
            maxx3d = np.max(c_ps[:, 0])
            miny3d = np.min(c_ps[:, 1])
            maxy3d = np.max(c_ps[:, 1])
            minz3d = np.min(c_ps[:, 2])
            maxz3d = np.max(c_ps[:, 2])
            obj_labels.append(
                {
                    "label": int(label),
                    "box2d": [float(minx2d), float(miny2d), float(maxx2d), float(maxy2d)],
                    "box3d": [
                        float(minx3d),
                        float(miny3d),
                        float(minz3d),
                        float(maxx3d),
                        float(maxy3d),
                        float(maxz3d),
                    ],
                    "id": len(obj_labels),
                }
            )

            voxel256 = np.zeros((256, 256, 32), dtype=np.int8)
            voxel256[c_ps[:, 0], c_ps[:, 1], c_ps[:, 2]] = 1
            box3d = torch.from_numpy(voxel256[minx3d : maxx3d + 1, miny3d : maxy3d + 1, minz3d : maxz3d + 1]).float()
            box3d_norm = F.interpolate(box3d[None, None, :, :, :], size=(32, 32, 32), mode="area")[0, 0]
            obj_voxels.append(box3d_norm.bool().numpy())

    return obj_labels, np.stack(obj_voxels)
