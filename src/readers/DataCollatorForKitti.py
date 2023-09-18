from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from transformers.data.data_collator import DataCollatorMixin
from transformers.models.deformable_detr.image_processing_deformable_detr import DeformableDetrImageProcessor

from tools.data.semantic_kitti import image_height, image_width, read_calib

from ..models.OccupancyDetrConfig import OccupancyDetrConfig


@dataclass
class DataCollatorForKitti(DataCollatorMixin):
    return_tensors: str = "pt"

    def __init__(self, processor: DeformableDetrImageProcessor):
        self.processor = processor
        self.all_calibs = {}

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        if return_tensors is None:
            return_tensors = self.return_tensors

        # loading
        batch_images = []
        batch_labels = []
        batch_cam_Es = []
        batch_cam_Ks = []
        ids = []
        max_length = 0
        for feature in features:
            image = Image.open(feature["image_path"])
            batch_images.append(image)
            ids.append(feature["image_path"])

            if feature["calib_path"] in self.all_calibs:
                calib = self.all_calibs[feature["calib_path"]]
            else:
                calib = read_calib(feature["calib_path"])
                self.all_calibs[feature["calib_path"]] = calib
            batch_cam_Es.append(calib["Tr"].tolist())
            batch_cam_Ks.append(calib["P2"].tolist())

            labels = {
                "class_labels": [],
                "boxes": [],
                "boxes3d": [],
            }
            for obj in feature["labels"][:300]:
                labels["class_labels"].append(obj["label"])
                minx, miny, maxx, maxy = obj["box2d"]
                x = ((minx + maxx) / 2) / image_width
                y = ((miny + maxy) / 2) / image_height
                w = (maxx - minx + 1) / image_width
                h = (maxy - miny + 1) / image_height
                labels["boxes"].append([x, y, w, h])
                minx3d, miny3d, minz3d, maxx3d, maxy3d, maxz3d = obj["box3d"]
                x3d = ((minx3d + maxx3d) / 2) / 256
                y3d = ((miny3d + maxy3d) / 2) / 256
                z3d = ((minz3d + maxz3d) / 2) / 32
                d3d = (maxx3d - minx3d + 1) / 256
                w3d = (maxy3d - miny3d + 1) / 256
                h3d = (maxz3d - minz3d + 1) / 32
                labels["boxes3d"].append([x3d, y3d, z3d, d3d, w3d, h3d])
            labels["class_labels"] = torch.tensor(labels["class_labels"])
            labels["boxes"] = torch.tensor(labels["boxes"])
            labels["boxes3d"] = torch.tensor(labels["boxes3d"])
            labels["obj3d"] = torch.from_numpy(np.load(feature["obj3ds_path"])).float()
            batch_labels.append(labels)
            max_length = max(max_length, len(labels["class_labels"]))

        # For multi-gpu training, convert to tensors
        batch_class_labels = []
        batch_boxes = []
        batch_boxes3d = []
        batch_obj3d = []
        for labels in batch_labels:
            padding_class_labels = torch.full((max_length,), -1, dtype=torch.long)
            padding_boxes = torch.zeros((max_length, 4), dtype=torch.float)
            padding_boxes3d = torch.zeros((max_length, 6), dtype=torch.float)
            padding_obj3d = torch.zeros((max_length, *labels["obj3d"].shape[1:]), dtype=torch.float)
            padding_class_labels[: len(labels["class_labels"])] = labels["class_labels"]
            padding_boxes[: len(labels["boxes"])] = labels["boxes"]
            padding_boxes3d[: len(labels["boxes3d"])] = labels["boxes3d"]
            padding_obj3d[: len(labels["obj3d"])] = labels["obj3d"]
            batch_class_labels.append(padding_class_labels)
            batch_boxes.append(padding_boxes)
            batch_boxes3d.append(padding_boxes3d)
            batch_obj3d.append(padding_obj3d)

        batch = self.processor(batch_images, return_tensors=return_tensors)
        batch["labels"] = torch.stack(batch_class_labels)
        batch["label_boxes"] = torch.stack(batch_boxes)
        batch["label_boxes3d"] = torch.stack(batch_boxes3d)
        batch["label_obj3d"] = torch.stack(batch_obj3d)
        batch["cam_Es"] = torch.tensor(batch_cam_Es)
        batch["cam_Ks"] = torch.tensor(batch_cam_Ks)
        batch["ids"] = ids
        return batch
