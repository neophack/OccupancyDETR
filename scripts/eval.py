import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import torch
from datasets import load_dataset
from PIL import Image

from src import OccupancyDetr, OccupancyDetrOutput, OccupancyDetrProcessor
from tools.common import *
from tools.data.semantic_kitti import read_calib
from tools.eval import eval_2d_objs, eval_voxels

seq_path = "/mnt/data2/Datasets/SemanticKITTI/dataset/sequences/08/labels.json"
output_dir = "/mnt/data2/Datasets/SemanticKITTI/dataset/sequences/08/outputs"
gt_dir = "/mnt/data2/Datasets/SemanticKITTI/dataset/sequences/08/restore"
preprocessed_dir = "/mnt/data2/Datasets/SemanticKITTI/dataset/sequences/08/preprocessed"
model_path = "saved_models/base_all_08_19_occ2d_4"
device = torch.device("cuda")
confidence_threshold = 0.5
occupancy_threshold = 0.5
mode = "occupancy"


def prepare_input(example):
    image = Image.open(example["image_path"])
    batch = image_processor([image], return_tensors="pt")
    calib = read_calib(example["calib_path"])
    batch["cam_Es"] = torch.tensor([calib["Tr"].tolist()]).to(device)
    batch["cam_Ks"] = torch.tensor([calib["P2"].tolist()]).to(device)
    batch["pixel_values"] = batch["pixel_values"].to(device)
    batch["pixel_mask"] = batch["pixel_mask"].to(device)
    return {
        "input": batch,
        "target": [],
        "info": example["image_path"],
    }


raw_datasets = load_dataset("json", data_files={"test": [seq_path]})
image_processor = OccupancyDetrProcessor.from_pretrained("SenseTime/deformable-detr")
model = OccupancyDetr.from_pretrained(model_path).to(device)
model.mode = mode
model.eval()
model.config.confidence_threshold = confidence_threshold
model.config.occupancy_threshold = occupancy_threshold
print(model.num_parameters())
image_processor.config = model.config
os.makedirs(output_dir, exist_ok=True)
dts = []
for example in raw_datasets["test"]:
    example = prepare_input(example)
    t1 = time.time()
    with torch.no_grad():
        output: OccupancyDetrOutput = model(**example["input"])
    t2 = time.time()
    print(f"{example['info']}: {t2 - t1}s")
    dts.append(t2 - t1)
    results, voxel = image_processor.post_process_for_OccupancyDetr(output)
    id = example["info"].split("/")[-1].split(".")[0]
    result_path = os.path.join(output_dir, f"{id}.json")
    results["image_path"] = example["info"]
    write_json(result_path, results)
    voxel_path = os.path.join(output_dir, f"{id}.npy")
    np.save(voxel_path, voxel.astype(np.uint16))

print(f"Average time: {np.mean(dts)}s")

AP50, _ = eval_2d_objs(output_dir, preprocessed_dir)
print(f"AP50: {AP50}")

miou = eval_voxels(output_dir, gt_dir)
print(f"IOU: {miou}")
print(f"MIOU: {np.mean(miou)}")
