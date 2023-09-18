import numpy as np
import torch
import torch.nn.functional as F
from common import *
from tqdm import tqdm

from tools.data.semantic_kitti import *


def restore_label(label_path, voxel_path, output_path):
    labels = read_json(label_path)["labels"]
    volumes = []
    for i in range(len(labels)):
        box3d = np.array(labels[i]["box3d"], dtype=np.int32)
        d, w, h = box3d[:3] - box3d[3:]
        volumes.append(d * w * h)
    sorted_idx = np.argsort(volumes)[::-1]

    voxel = torch.zeros((256, 256, 32), dtype=torch.int8)
    obj3ds = np.load(voxel_path)
    for i in sorted_idx:
        box3d = np.array(labels[i]["box3d"], dtype=np.int32)
        cls = labels[i]["label"]
        occ_in_box = torch.from_numpy(obj3ds[i]).float()
        minx, miny, minz, maxx, maxy, maxz = box3d
        d, w, h = (box3d[3:] - box3d[:3]).tolist()
        d = max(1, min(d, 256 - minx))
        w = max(1, min(w, 256 - miny))
        h = max(1, min(h, 32 - minz))
        occ_in_world = torch.zeros((256, 256, 32))
        occ_in_box = F.interpolate(occ_in_box[None, None, :, :, :], size=(d, w, h))[0, 0]
        occ_in_world[minx : minx + d, miny : miny + w, minz : minz + h] = occ_in_box
        voxel[occ_in_world > 0.5] = cls
    voxel = voxel.numpy()
    np.save(output_path, voxel)


if __name__ == "__main__":
    seq_dir = "/mnt/data2/Datasets/SemanticKITTI/dataset/sequences/08"
    obj3d_dir = f"{seq_dir}/obj3d"
    preprocessed_dir = f"{seq_dir}/preprocessed"
    restore_dir = f"{seq_dir}/restore"
    for file in tqdm(os.listdir(preprocessed_dir)):
        label_path = os.path.join(preprocessed_dir, file)
        obj3d_path = os.path.join(obj3d_dir, file.replace(".json", ".npy"))
        output_path = os.path.join(restore_dir, file.replace(".json", ".npy"))
        restore_label(label_path, obj3d_path, output_path)
