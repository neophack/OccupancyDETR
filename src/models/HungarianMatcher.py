from transformers.models.deformable_detr.modeling_deformable_detr import *


def center_to_corners_format2d(box):
    boxes = torch.cat([box[..., :2] - box[..., 2:] / 2, box[..., :2] + box[..., 2:] / 2], -1)
    boxes = torch.clamp(boxes, 0, 1)
    return boxes


class HungarianMatcher(DeformableDetrHungarianMatcher):
    @torch.no_grad()
    def forward(self, outputs, targets, mask=None):
        batch_size, num_queries = outputs["pred_boxes"].shape[:2]

        # Compute the classification cost.
        class_cost = 0
        if "logits" in outputs:
            alpha = 0.25
            gamma = 2.0
            # [batch_size * num_queries, num_classes]
            out_prob = outputs["logits"].flatten(0, 1).sigmoid()
            target_ids = torch.cat([v["class_labels"] for v in targets])
            neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            class_cost = pos_cost_class[:, target_ids] - neg_cost_class[:, target_ids]

        # Compute the L1 cost between boxes
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        target_bbox = torch.cat([v["boxes"] for v in targets])
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)

        # Compute the giou cost between boxes
        giou_cost = -generalized_box_iou(center_to_corners_format2d(out_bbox), center_to_corners_format2d(target_bbox))

        # Final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()
        if mask is not None:
            cost_matrix[mask.expand_as(cost_matrix).cpu()] = float("inf")

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
