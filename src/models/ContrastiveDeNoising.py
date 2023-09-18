import torch
import torch.nn as nn
from transformers.models.deformable_detr.modeling_deformable_detr import *

from .OccupancyDetrConfig import OccupancyDetrConfig


class ContrastiveDeNoising(nn.Module):
    def __init__(self, config: OccupancyDetrConfig) -> None:
        super().__init__()
        self.dn_number = config.dn_number
        self.label_noise_ratio = config.dn_label_noise_ratio
        self.box_noise_scale = config.dn_box_noise_scale
        self.num_queries = config.num_queries
        self.num_labels = config.num_labels
        self.d_model = config.d_model
        self.label_enc = nn.Embedding(config.num_labels, config.d_model)

    def forward(self, targets):
        if not self.training or self.dn_number == 0:
            return None, None, None, None

        # positive and negative dn queries
        device = targets[0]["class_labels"].device
        dn_number = self.dn_number * 2
        known = [torch.ones_like(t["class_labels"]) for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            return None, None, None, None

        dn_number = dn_number // (int(max(known_num) * 2))
        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t["class_labels"] for t in targets])
        boxes = torch.cat([t["boxes"] for t in targets])
        batch_idx = torch.cat([torch.full_like(t["class_labels"].long(), i) for i, t in enumerate(targets)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)
        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        # noise label
        if self.label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            # half of bbox prob
            chosen_indice = torch.nonzero(p < (self.label_noise_ratio * 0.5)).view(-1)
            # randomly put a new one here
            new_label = torch.randint_like(chosen_indice, 0, self.num_labels)
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_pad = int(max(known_num))

        # noise boxes
        pad_size = int(single_pad * 2 * dn_number)
        positive_idx = torch.arange(0, len(boxes), device=device)
        positive_idx = positive_idx.unsqueeze(0).repeat(dn_number, 1)
        torch.arange(0, dn_number, device=device)
        positive_idx += torch.arange(0, dn_number, device=device).unsqueeze(1) * len(boxes) * 2
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        if self.box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            rand_sign = torch.randint_like(known_bboxs, low=0, high=2) * 2.0 - 1.0
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part, diff) * self.box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        input_label_embed = self.label_enc(known_labels_expaned)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, self.d_model).to(device)
        padding_bbox = torch.zeros(pad_size, 4).to(device)

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([], device=device)
        if len(known_num):
            map_known_indice = torch.cat([torch.arange(0, num) for num in known_num]).to(device)
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
        if len(known_bid):
            input_query_label[(known_bid, map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid, map_known_indice)] = input_bbox_embed

        # build dn mask
        tgt_size = pad_size + self.num_queries
        dn_mask = torch.ones(tgt_size, tgt_size, device=device) < 0
        # 1. match query cannot see the reconstruct
        dn_mask[pad_size:, :pad_size] = True
        dn_mask[:pad_size, pad_size:] = True
        # 2. reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                dn_mask[single_pad * 2 * i : single_pad * 2 * (i + 1), single_pad * 2 * (i + 1) : pad_size] = True
            if i == dn_number - 1:
                dn_mask[single_pad * 2 * i : single_pad * 2 * (i + 1), : single_pad * i * 2] = True
            else:
                dn_mask[single_pad * 2 * i : single_pad * 2 * (i + 1), single_pad * 2 * (i + 1) : pad_size] = True
                dn_mask[single_pad * 2 * i : single_pad * 2 * (i + 1), : single_pad * 2 * i] = True
        dn_mask = dn_mask.unsqueeze(0).repeat(batch_size, 1, 1)
        # for huggingface attention
        dn_mask2 = torch.zeros(dn_mask.shape, device=device)
        dn_mask2[dn_mask] = float("-inf")

        dn_meta = {
            "pad_size": pad_size,
            "num_dn_group": dn_number,
        }
        return input_query_label, input_query_bbox, dn_mask2, dn_meta

    @staticmethod
    def dn_post_process(outputs_class, outputs_boxes, outputs_boxes3d, dn_meta, aux_loss, _set_aux_loss):
        if dn_meta and dn_meta["pad_size"] > 0:
            output_known_class = outputs_class[:, :, : dn_meta["pad_size"], :]
            output_known_boxes = outputs_boxes[:, :, : dn_meta["pad_size"], :]
            output_known_boxes3d = outputs_boxes3d[:, : dn_meta["pad_size"], :]
            outputs_class = outputs_class[:, :, dn_meta["pad_size"] :, :]
            outputs_boxes = outputs_boxes[:, :, dn_meta["pad_size"] :, :]
            outputs_boxes3d = outputs_boxes3d[:, dn_meta["pad_size"] :, :]
            out = {"logits": output_known_class[-1], "pred_boxes": output_known_boxes[-1], "pred_boxes3d": output_known_boxes3d}
            if aux_loss:
                out["aux_outputs"] = _set_aux_loss(output_known_class, output_known_boxes)
            dn_meta["output_known_lbs_bboxes"] = out
        return outputs_class, outputs_boxes, outputs_boxes3d
