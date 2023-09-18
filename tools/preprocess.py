import glob
import multiprocessing
import shutil
from multiprocessing.pool import AsyncResult
from typing import List

from common import *
from tqdm import tqdm
from voxel import get_can_see
from voxel2obj import voxel2obj

from tools.data.semantic_kitti import *


def preprocess_frame(label_path, occluded_path, invalid_path, calib_path, output_path, obj3ds_path):
    calib = read_calib(calib_path)
    remap_lut = get_remap_lut()
    label = read_label(label_path)
    occluded = read_occluded(occluded_path)
    invalid = read_invalid(invalid_path)
    label = remap_lut[label.astype(np.uint16)]
    label[occluded == 0] = 255
    label[invalid == 1] = 255
    label = label.reshape([256, 256, 32])
    can_see = get_can_see(label, calib)
    label_dict, obj_voxels = voxel2obj(label, can_see, calib)
    np.save(obj3ds_path, obj_voxels)
    write_json(
        output_path,
        {
            "labels": label_dict,
            "calib_path": calib_path,
            "label_path": label_path,
            "occluded_path": occluded_path,
            "invalid_path": invalid_path,
            "obj3ds_path": obj3ds_path,
        },
    )
    print(len(label_dict))


def preprocess_sequence(seq_dir):
    pool = multiprocessing.Pool(processes=10)
    results: List[AsyncResult] = []

    calib_path = os.path.join(seq_dir, "calib.txt")
    voxel_dir = os.path.join(seq_dir, "voxels")
    image_dir = os.path.join(seq_dir, "image_2")
    preprocessed_dir = os.path.join(seq_dir, "preprocessed")
    obj3d_dir = os.path.join(seq_dir, "obj3d")
    os.makedirs(preprocessed_dir, exist_ok=True)
    os.makedirs(obj3d_dir, exist_ok=True)
    params = []
    glob_path = os.path.join(voxel_dir, "*.label")
    for label_path in glob.glob(glob_path):
        file_name = os.path.basename(label_path)
        frame_id = file_name.split(".")[0]
        occluded_path = os.path.join(voxel_dir, f"{frame_id}.occluded")
        invalid_path = os.path.join(voxel_dir, f"{frame_id}.invalid")
        preprocessed_path = os.path.join(preprocessed_dir, f"{frame_id}.json")
        obj3ds_path = os.path.join(obj3d_dir, f"{frame_id}.npy")
        if os.path.exists(preprocessed_path):
            continue
        params.append((label_path, occluded_path, invalid_path, calib_path, preprocessed_path, obj3ds_path))
    for label_path, occluded_path, invalid_path, calib_path, preprocessed_path, obj3ds_path in params:
        # preprocess_frame(label_path, occluded_path, invalid_path, calib_path, preprocessed_path, obj3ds_path)
        results.append(pool.apply_async(preprocess_frame, args=(label_path, occluded_path, invalid_path, calib_path, preprocessed_path, obj3ds_path)))
    for result in tqdm(results):
        result.wait()
    pool.close()
    pool.join()

    lines = []
    for file in os.listdir(preprocessed_dir):
        frame_id = file.split(".")[0]
        image_path = os.path.join(image_dir, f"{frame_id}.png")
        labels = read_json(os.path.join(preprocessed_dir, file))
        labels["image_path"] = image_path
        lines.append(json.dumps(labels))
    if len(lines) > 0:
        with open(os.path.join(seq_dir, "labels.json"), "w") as f:
            f.write("\n".join(lines))


def proprocess_all(dataset_dir):
    for seq_name in os.listdir(dataset_dir):
        seq_dir = os.path.join(dataset_dir, seq_name)
        if os.path.isdir(seq_dir):
            preprocess_sequence(seq_dir)


def clear(dataset_dir):
    for seq_name in os.listdir(dataset_dir):
        seq_dir = os.path.join(dataset_dir, seq_name, "preprocessed")
        if os.path.isdir(seq_dir):
            shutil.rmtree(seq_dir)


if __name__ == "__main__":
    # clear("/mnt/data2/Datasets/SemanticKITTI/dataset/sequences")
    preprocess_sequence("/mnt/data2/Datasets/SemanticKITTI/dataset/sequences/00")
    preprocess_sequence("/mnt/data2/Datasets/SemanticKITTI/dataset/sequences/01")
    preprocess_sequence("/mnt/data2/Datasets/SemanticKITTI/dataset/sequences/02")
    preprocess_sequence("/mnt/data2/Datasets/SemanticKITTI/dataset/sequences/03")
    preprocess_sequence("/mnt/data2/Datasets/SemanticKITTI/dataset/sequences/04")
    preprocess_sequence("/mnt/data2/Datasets/SemanticKITTI/dataset/sequences/05")
    preprocess_sequence("/mnt/data2/Datasets/SemanticKITTI/dataset/sequences/06")
    preprocess_sequence("/mnt/data2/Datasets/SemanticKITTI/dataset/sequences/07")
    preprocess_sequence("/mnt/data2/Datasets/SemanticKITTI/dataset/sequences/08")
    preprocess_sequence("/mnt/data2/Datasets/SemanticKITTI/dataset/sequences/09")
    preprocess_sequence("/mnt/data2/Datasets/SemanticKITTI/dataset/sequences/10")
