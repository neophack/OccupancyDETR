import cv2
import numpy as np
import pyvista as pv
import torch
import torch.nn.functional as F
from common import *
from tqdm import tqdm
from voxel import *

from data.semantic_kitti import *


def viz_raw_voxel(label_path, occluded_path=None, invalid_path=None):
    remap_lut = get_remap_lut()
    label = read_label(label_path)
    label = read_label(label_path)
    label = remap_lut[label.astype(np.uint16)]
    if occluded_path is not None:
        occluded = read_occluded(occluded_path)
        label[occluded == 0] = 255
    if invalid_path is not None:
        invalid = read_invalid(invalid_path)
        label[invalid == 0] = 255
    label = label.reshape([256, 256, 32])

    dims = [256, 256, 32]
    grid = pv.UniformGrid(dims)
    grid["Occupancy"] = label.flatten(order="F")
    cmap = get_cmap()
    plotter = pv.Plotter()
    for i in range(20):
        color = cmap[i]
        threshed_occupied = grid.threshold([i, i + 0.1])
        if threshed_occupied.n_points > 0:
            opacity = 0.5 if i == 0 else 1
            plotter.add_mesh(threshed_occupied, color=color, show_edges=False, opacity=opacity)
    plotter.show()


def viz_vox2pix(calib_path, invalid_path, label_path):
    calib = read_calib(calib_path)

    remap_lut = get_remap_lut()
    label = read_label(label_path)
    label = remap_lut[label.astype(np.uint16)]

    invalid = read_invalid(invalid_path)
    label[invalid == 1] = 0
    label = label.reshape([256, 256, 32])

    projected_pix, fov_mask, pix_z, vox_coords = vox2pix(calib)
    vox_coords = vox_coords[fov_mask]
    labe_mask = label[vox_coords[:, 0], vox_coords[:, 1], vox_coords[:, 2]] != 0
    vox_coords = vox_coords[labe_mask]
    pix_z = pix_z[fov_mask]
    pix_z = pix_z[labe_mask]
    projected_pix, _, _ = vox2pix2(calib, vox_coords)
    projected_vox = projected_pix.reshape((8, -1, 2))
    projected_vox = projected_vox.transpose((1, 0, 2))

    near2far = np.argsort(pix_z)
    depth = np.zeros([image_height, image_width], dtype=np.float32)
    can_see = []
    for i in tqdm(near2far):
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
    depth[depth == 0] = 100
    depth = (1 - depth / 100) * 255
    cv2.imwrite("viz.png", depth)

    can_see_coords = vox_coords[can_see]
    can_see_label = np.zeros_like(label)
    can_see_label[can_see_coords[:, 0], can_see_coords[:, 1], can_see_coords[:, 2]] = label[can_see_coords[:, 0], can_see_coords[:, 1], can_see_coords[:, 2]]
    np.save("viz.npy", can_see_label)


def viz_bbox2d(viz_path, label_path):
    img = cv2.imread(viz_path)
    labels = read_json(label_path)
    cmap = get_cmap()
    for data in labels["labels"]:
        label = data["label"]
        minx, miny, maxx, maxy = np.array(data["bbox"], dtype=np.int32)
        color = (int(cmap[label][0]), int(cmap[label][1]), int(cmap[label][2]))
        img = cv2.rectangle(img, (minx, miny), (maxx, maxy), color, 2)
    cv2.imwrite("viz2.png", img)


def viz_label(image_path, label_path, voxel_path):
    labels = read_json(label_path)["labels"]
    image = cv2.imread(image_path)
    cmap = get_cmap().tolist()
    h, w = image.shape[:2]
    for obj in labels:
        cls = obj["label"]
        minx, miny, maxx, maxy = obj["box2d"]
        minx = int(minx)
        miny = int(miny)
        maxx = int(maxx)
        maxy = int(maxy)
        color = cmap[cls]
        image = cv2.rectangle(image, (minx, miny), (maxx, maxy), color, 2)
        image = cv2.putText(image, f"{id2label[cls]}", (minx, miny), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite("viz.png", image)

    voxel = torch.zeros((256, 256, 32))
    obj3ds = np.load(voxel_path)
    for i in range(len(obj3ds)):
        box3d = np.array(labels[i]["box3d"], dtype=np.int32)
        cls = labels[i]["label"]
        occ_in_box = torch.from_numpy(obj3ds[i]).float()
        minx, miny, minz, maxx, maxy, maxz = box3d
        d, w, h = box3d[3:] - box3d[:3]
        d = max(1, min(d, 256 - minx))
        w = max(1, min(w, 256 - miny))
        h = max(1, min(h, 32 - minz))
        occ_in_world = torch.zeros((256, 256, 32))
        occ_in_box = F.interpolate(occ_in_box[None, None, :, :, :], size=(d, w, h))[0, 0]
        occ_in_world[minx : minx + d, miny : miny + w, minz : minz + h] = occ_in_box
        voxel[occ_in_world > 0.5] = cls
    voxel = voxel.numpy()

    dims = [256, 256, 32]
    grid = pv.UniformGrid(dims)
    grid["Occupancy"] = voxel.flatten(order="F")
    cmap = get_cmap()
    plotter = pv.Plotter()
    for i in range(1, 20):
        color = cmap[i]
        threshed_occupied = grid.threshold([i, i + 0.1])
        if threshed_occupied.n_points > 0:
            plotter.add_mesh(threshed_occupied, color=color, show_edges=False)

    for obj in labels:
        cls = obj["label"]
        minx, miny, minz, maxx, maxy, maxz = obj["box3d"]
        color = cmap[cls]
        box = pv.Box([minx, maxx, miny, maxy, minz, maxz])
        edges = box.extract_feature_edges()
        plotter.add_mesh(edges, color=color)
        plotter.add_point_labels([[minx, miny, minz]], [id2label[cls]], font_size=8)
    plotter.show()


def viz_result(image_path, result_path, voxel_path):
    result = read_json(result_path)
    image = cv2.imread(image_path)
    cmap = get_cmap().tolist()
    h, w = image.shape[:2]
    for cls, s, bbox in zip(result["classes"], result["scores"], result["boxes"]):
        minx, miny, maxx, maxy = bbox
        minx = int(minx * w)
        miny = int(miny * h)
        maxx = int(maxx * w)
        maxy = int(maxy * h)
        color = cmap[cls]
        image = cv2.rectangle(image, (minx, miny), (maxx, maxy), color, 2)
        image = cv2.putText(image, f"{id2label[cls]} {s:.2f}", (minx, miny), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite("viz.png", image)

    label = np.load(voxel_path)
    dims = [256, 256, 32]
    grid = pv.UniformGrid(dims)
    grid["Occupancy"] = label.flatten(order="F")
    cmap = get_cmap()
    plotter = pv.Plotter()
    for i in range(1, 20):
        color = cmap[i]
        threshed_occupied = grid.threshold([i, i + 0.1])
        if threshed_occupied.n_points > 0:
            plotter.add_mesh(threshed_occupied, color=color, show_edges=False)

    for cls, s, bbox3d in zip(result["classes"], result["scores"], result["boxes3d"]):
        minx, miny, minz, maxx, maxy, maxz = bbox3d
        minx = int(minx * 256)
        miny = int(miny * 256)
        minz = int(minz * 32)
        maxx = int(maxx * 256)
        maxy = int(maxy * 256)
        maxz = int(maxz * 32)
        color = cmap[cls]
        box = pv.Box([minx, maxx, miny, maxy, minz, maxz])
        edges = box.extract_feature_edges()
        plotter.add_mesh(edges, color=color)
        plotter.add_point_labels([[minx, miny, minz]], [id2label[cls]], font_size=8)
    plotter.show()


def viz_intersection(voxel_path, restore_path):
    label = np.load(restore_path).astype(np.float32)
    gt = np.load(voxel_path).astype(np.float32)
    inter = np.zeros_like(label)
    inter[gt != label] = 0.5
    label = label + inter
    dims = [256, 256, 32]
    grid = pv.UniformGrid(dims)
    grid["Occupancy"] = label.flatten(order="F")
    cmap = get_cmap()
    plotter = pv.Plotter()
    for i in range(1, 20):
        color = cmap[i]
        threshed_occupied = grid.threshold([i, i + 0.1])
        if threshed_occupied.n_points > 0:
            plotter.add_mesh(threshed_occupied, color=color, show_edges=False)
        threshed_occupied2 = grid.threshold([i+0.5, i + 0.6])
        if threshed_occupied2.n_points > 0:
            plotter.add_mesh(threshed_occupied2, color=color, show_edges=False, opacity=0.2)
    plotter.show()


def record_results(output_dir):
    dims = [256, 256, 32]
    cmap = get_cmap()
    plotter = pv.Plotter()
    plotter.camera_position = [
        (-170.04, 123.01, 276.53),
        (112.41, 126.32, -2.1660),
        (0.7023, -0.002, 0.71180),
    ]
    plotter.open_movie("viz.mp4", framerate=4)
    actors = []
    for f in tqdm(sorted(os.listdir(output_dir))):
        if not f.endswith(".npy"):
            continue
        for a in actors:
            plotter.remove_actor(a)
        actors.clear()
        voxel_path = os.path.join(output_dir, f)
        label = np.load(voxel_path)
        grid = pv.UniformGrid(dims)
        grid["Occupancy"] = label.flatten(order="F")
        for i in range(1, 20):
            color = cmap[i]
            threshed_occupied = grid.threshold([i, i + 0.1])
            if threshed_occupied.n_points > 0:
                actors.append(plotter.add_mesh(threshed_occupied, color=color, show_edges=False))
        plotter.write_frame()
    plotter.close()


if __name__ == "__main__":
    # restore_dir = "/home/jyp/Cloud/data2/Datasets/SemanticKITTI/dataset/sequences/08/restore"
    # output_dir = "/home/jyp/Cloud/data2/Datasets/SemanticKITTI/dataset/sequences/08/outputs"
    # for file in os.listdir(restore_dir):
    #     voxel_path = os.path.join(output_dir, file)
    #     restore_path = os.path.join(restore_dir, file)
    #     viz_intersection(voxel_path, restore_path)

    # viz_result(
    #     "/home/jyp/Cloud/data2/Datasets/SemanticKITTI/dataset/sequences/08/image_2/000755.png",
    #     "/home/jyp/Cloud/data2/Datasets/SemanticKITTI/dataset/sequences/08/outputs/000755.json",
    #     "/home/jyp/Cloud/data2/Datasets/SemanticKITTI/dataset/sequences/08/outputs/000755.npy",
    # )

    record_results("/mnt/data2/Datasets/SemanticKITTI/dataset/sequences/08/outputs")