from transformers.models.deformable_detr.modeling_deformable_detr import *

from .OccupancyDetrConfig import OccupancyDetrConfig


def center_to_corners_format3d(box):
    boxes = torch.cat([box[..., :3] - box[..., 3:] / 2, box[..., :3] + box[..., 3:] / 2], -1)
    boxes = torch.clamp(boxes, 0, 1)
    return boxes


def corners_to_center_format3d(box):
    boxes = torch.cat([(box[..., :3] + box[..., 3:]) / 2, box[..., 3:] - box[..., :3]], -1)
    boxes = torch.clamp(boxes, 0, 1)
    return boxes


def generalized_box_iou3d(bbox_preds, bbox_targets):
    bbox_preds = bbox_preds.unsqueeze(1)
    bbox_targets = bbox_targets.unsqueeze(0)
    intersect_mins = torch.max(bbox_preds[..., :3], bbox_targets[..., :3])
    intersect_maxs = torch.min(bbox_preds[..., 3:], bbox_targets[..., 3:])
    intersect_whd = torch.clamp(intersect_maxs - intersect_mins, min=0)
    intersect_vol = intersect_whd[..., 0] * intersect_whd[..., 1] * intersect_whd[..., 2]
    bbox_preds_vol = (bbox_preds[..., 3] - bbox_preds[..., 0]) * (bbox_preds[..., 4] - bbox_preds[..., 1]) * (bbox_preds[..., 5] - bbox_preds[..., 2])
    bbox_targets_vol = (bbox_targets[..., 3] - bbox_targets[..., 0]) * (bbox_targets[..., 4] - bbox_targets[..., 1]) * (bbox_targets[..., 5] - bbox_targets[..., 2])
    union_vol = bbox_preds_vol + bbox_targets_vol - intersect_vol
    iou = intersect_vol / torch.clamp(union_vol, min=1e-6)
    enclose_mins = torch.min(bbox_preds[..., :3], bbox_targets[..., :3])
    enclose_maxs = torch.max(bbox_preds[..., 3:], bbox_targets[..., 3:])
    enclose_whd = torch.clamp(enclose_maxs - enclose_mins, min=0)
    enclose_vol = enclose_whd[..., 0] * enclose_whd[..., 1] * enclose_whd[..., 2]
    return iou - (enclose_vol - union_vol) / torch.clamp(enclose_vol, min=1e-6)


def dice_loss3d(pred, target):
    pred = pred.contiguous().flatten(1)
    target = target.contiguous().flatten(1)
    intersection = 2 * (pred * target).sum(1)
    union = pred.sum(1) + target.sum(1)
    loss = 1 - ((intersection + 1) / (union + 1))
    return loss


class OccupancyDetrLoss(DeformableDetrLoss):
    def __init__(self, matcher, losses, config: OccupancyDetrConfig):
        super().__init__(matcher, config.num_labels, config.focal_alpha, losses)
        self.config = config

    def prep_for_dn(self, dn_meta):
        output_known_lbs_bboxes = dn_meta["output_known_lbs_bboxes"]
        num_dn_groups, pad_size = dn_meta["num_dn_group"], dn_meta["pad_size"]
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size // num_dn_groups
        return output_known_lbs_bboxes, single_pad, num_dn_groups

    def loss_boxes3d(self, outputs, targets, indices, num_boxes):
        if "pred_boxes3d" not in outputs:
            return {}
        idx = self._get_source_permutation_idx(indices)
        source_boxes = outputs["pred_boxes3d"][idx]
        target_boxes = torch.cat([t["boxes3d"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction="none")
        losses["loss_bbox3d"] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(generalized_box_iou3d(center_to_corners_format3d(source_boxes), center_to_corners_format3d(target_boxes)))
        losses["loss_giou3d"] = loss_giou.sum() / num_boxes
        return losses

    def loss_occ(self, outputs, targets, indices, num_boxes):
        if "pred_occ" not in outputs:
            return {}
        tb, ti = outputs["occ_idx2target_idx"]
        source_occs_in_box = outputs["pred_occ"]
        source_boxes = outputs["enlarged_boxes3d"]
        target_occs_in_box = torch.stack([targets[b]["obj3d"][i] for b, i in zip(tb, ti)]).float()
        target_boxes = torch.stack([targets[b]["boxes3d"][i] for b, i in zip(tb, ti)])

        target_occs_in_pred_box = []
        for occ, box, tocc, tbox in zip(source_occs_in_box, source_boxes, target_occs_in_box, target_boxes):
            x, y, z, d, w, h = tbox
            x, y, z = int(x * 256), int(y * 256), int(z * 32)
            d, w, h = int(d * 256), int(w * 256), int(h * 32)
            minx = max(0, x - d // 2)
            miny = max(0, y - w // 2)
            minz = max(0, z - h // 2)
            w = max(1, min(w, 256 - minx))
            d = max(1, min(d, 256 - miny))
            h = max(1, min(h, 32 - minz))
            tworld = torch.zeros(self.config.voxel_shape, device=tocc.device)
            tocc_in_box = F.interpolate(tocc[None, None, :, :, :], size=(d, w, h))[0, 0]
            tworld[minx : minx + d, miny : miny + w, minz : minz + h] = tocc_in_box

            x, y, z, d, w, h = box
            x, y, z = int(x * 256), int(y * 256), int(z * 32)
            d, w, h = int(d * 256), int(w * 256), int(h * 32)
            minx = max(0, x - d // 2)
            miny = max(0, y - w // 2)
            minz = max(0, z - h // 2)
            w = max(1, min(w, 256 - minx))
            d = max(1, min(d, 256 - miny))
            h = max(1, min(h, 32 - minz))
            tocc_in_box = tworld[minx : minx + d, miny : miny + w, minz : minz + h]
            tocc_in_box = F.interpolate(tocc_in_box[None, None, :, :, :], size=occ.shape)[0, 0]
            target_occs_in_pred_box.append(tocc_in_box)
        target_occs_in_pred_box = torch.stack(target_occs_in_pred_box)

        losses = {}
        loss_dice3d = dice_loss3d(source_occs_in_box, target_occs_in_pred_box)
        losses["loss_dice3d"] = loss_dice3d.sum() / num_boxes
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "boxes3d": self.loss_boxes3d,
            "occupancy": self.loss_occ,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets, indices):
        device = outputs["logits"].device

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        losses = {}
        # Compute DN losses
        dn_meta = outputs["dn_meta"]
        if self.training and dn_meta and "output_known_lbs_bboxes" in dn_meta:
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]["class_labels"]) > 0:
                    t = torch.arange(0, len(targets[i]["class_labels"]) - 1, device=device)
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = torch.arange(0, scalar, device=device).unsqueeze(1) * single_pad + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([], dtype=torch.long, device=device)
                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            l_dict = {}
            for loss in self.losses:
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes * scalar))

            l_dict = {k + f"_dn": v for k, v in l_dict.items()}
            losses.update(l_dict)

        # For two stage, compute losses for proposals
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            indices = enc_outputs["proposal_indices"]
            if indices is None:
                indices = self.matcher(enc_outputs, targets)
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["class_labels"] = torch.zeros_like(bt["class_labels"])
            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)
            if self.config.only_encoder_loss:
                return losses

        # Compute all the requested losses
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                for loss in self.losses:
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and "output_known_lbs_bboxes" in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes["aux_outputs"][i]
                    l_dict = {}
                    for loss in self.losses:
                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes * scalar))
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
