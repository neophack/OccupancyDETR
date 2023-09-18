from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.deformable_detr.image_processing_deformable_detr import DeformableDetrImageProcessor

from ..models.OccupancyDetr import OccupancyDetrOutput
from ..models.OccupancyDetrConfig import OccupancyDetrConfig


class OccupancyDetrProcessor(DeformableDetrImageProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config: OccupancyDetrConfig = None

    def post_process_for_OccupancyDetr(
        self,
        output: OccupancyDetrOutput,
    ) -> Dict:
        n = output.num_preds[0]  # only support batch size 1
        pred_occ = output.pred_occ[:n].detach().cpu() if output.pred_occ is not None else None
        pred_logits = output.pred_logits[:n].detach().sigmoid().cpu().numpy()
        pred_boxes = output.pred_boxes[:n].detach().cpu().numpy()
        pred_boxes3d = output.pred_boxes3d[:n].detach().cpu().numpy()
        classes, scores = pred_logits.argmax(-1), pred_logits.max(-1)

        volumes = []
        for i, box3d in enumerate(pred_boxes3d):
            d, w, h = box3d[3:]
            volumes.append(d * w * h)
        idx = np.argsort(volumes)[::-1]

        voxel = torch.zeros((256, 256, 32), dtype=torch.int8)
        results = {
            "classes": [],
            "scores": [],
            "boxes": [],
            "boxes3d": [],
        }
        for i in idx:
            results["classes"].append(int(classes[i]))
            results["scores"].append(float(scores[i]))
            box = np.zeros((4))
            box[:2] = pred_boxes[i][:2] - pred_boxes[i][2:] / 2
            box[2:] = pred_boxes[i][:2] + pred_boxes[i][2:] / 2
            results["boxes"].append(box.tolist())
            box3d = np.zeros((6))
            box3d[:3] = pred_boxes3d[i][:3] - pred_boxes3d[i][3:6] / 2
            box3d[3:] = pred_boxes3d[i][:3] + pred_boxes3d[i][3:6] / 2
            results["boxes3d"].append(box3d.tolist())
            if pred_occ is not None:
                x, y, z, d, w, h = pred_boxes3d[i]
                x, y, z = int(x * 256), int(y * 256), int(z * 32)
                d, w, h = int(d * 256), int(w * 256), int(h * 32)
                minx = max(0, x - d // 2)
                miny = max(0, y - w // 2)
                minz = max(0, z - h // 2)
                d = max(1, min(d, 256 - minx))
                w = max(1, min(w, 256 - miny))
                h = max(1, min(h, 32 - minz))
                occ_in_world = torch.zeros((256, 256, 32))
                occ_in_box = F.interpolate(pred_occ[i][None, None, :, :, :], size=(d, w, h))[0, 0]
                occ_in_world[minx : minx + d, miny : miny + w, minz : minz + h] = occ_in_box
                voxel[occ_in_world > self.config.occupancy_threshold] = classes[i]
        return results, voxel.numpy()
