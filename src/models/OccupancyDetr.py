import torch
import torch.nn as nn
from transformers.models.deformable_detr.modeling_deformable_detr import *

from . import MLP
from .ContrastiveDeNoising import ContrastiveDeNoising
from .HungarianMatcher import HungarianMatcher
from .OccupancyDetrConfig import OccupancyDetrConfig
from .OccupancyDetrLoss import *
from .OccupancyDetrModel import OccupancyDetrModel, OccupancyDetrModelOutput
from .OccupancyDetrPretrainedModel import OccupancyDetrPretrainedModel
from .SpatialTransformer3D import SpatialTransformer3D
from .SpatialTransformerBEV import SpatialTransformerBEV


@dataclass
class OccupancyDetrOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None  # all queries logits
    pred_logits: torch.FloatTensor = None  # filtered by confidence_threshold
    pred_boxes: torch.FloatTensor = None  # filtered by confidence_threshold
    pred_boxes3d: torch.FloatTensor = None  # filtered by confidence_threshold
    pred_occ: torch.FloatTensor = None  # filtered by confidence_threshold
    num_preds: torch.LongTensor = None  # number of predictions for each image


class OccupancyDetr(OccupancyDetrPretrainedModel):
    def __init__(self, config: OccupancyDetrConfig, mode: str = "detection"):
        """mode: detection or occupancy"""
        super().__init__(config)
        self.config = config
        self.mode = mode
        self.model = OccupancyDetrModel(config)
        if config.query_mode == "bev":
            self.occ_transformer = SpatialTransformerBEV(config)
        elif config.query_mode == "3d":
            self.occ_transformer = SpatialTransformer3D(config)
        else:
            assert mode == "detection"
            self.occ_transformer = None
        self.cdn = ContrastiveDeNoising(config)
        self.matcher = HungarianMatcher(class_cost=config.class_cost, bbox_cost=config.bbox_cost, giou_cost=config.giou_cost)
        self.criterion = OccupancyDetrLoss(
            matcher=self.matcher,
            losses=["labels", "boxes", "boxes3d", "occupancy"],
            config=config,
        )

        # Detection heads on top
        self.class_embed = nn.Linear(config.d_model, config.num_labels)
        self.bbox_embed = MLP(config.d_model, config.d_model, 4, 3)
        self.bbox3d_head = MLP(config.d_model, config.d_model, 6, 3)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(config.num_labels) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = config.decoder_layers + 1
        self.class_embed = nn.ModuleList([copy.deepcopy(self.class_embed) for i in range(num_pred)])
        self.bbox_embed = nn.ModuleList([copy.deepcopy(self.bbox_embed) for i in range(num_pred)])
        self.model.decoder.bbox_embed = self.bbox_embed
        self.model.decoder.class_embed = self.class_embed
        # hack implementation for two-stage
        for bbox_embed_layer in self.bbox_embed[:-1]:
            nn.init.constant_(bbox_embed_layer.layers[-1].bias.data[2:], 0.0)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_detection(self):
        for name, param in self.named_parameters():
            param.requires_grad_(False)
        for name, param in self.occ_transformer.named_parameters():
            param.requires_grad_(True)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def forward(
        self,
        cam_Es,
        cam_Ks,
        pixel_values,
        pixel_mask=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        label_boxes=None,
        label_boxes3d=None,
        label_obj3d=None,
        output_attentions=None,
        output_hidden_states=None,
        ids=None,
    ):
        labels_dict = None
        if labels is not None:
            labels_dict = []
            for cls, bbox, bbox3d, obj3d in zip(labels, label_boxes, label_boxes3d, label_obj3d):
                cls_mask = cls != -1
                labels_dict.append(
                    {
                        "class_labels": cls[cls_mask],
                        "boxes": bbox[cls_mask],
                        "boxes3d": bbox3d[cls_mask],
                        "obj3d": obj3d[cls_mask],
                    }
                )

            dn_query_label, dn_query_bbox, dn_mask, dn_meta = self.cdn(labels_dict)
        else:
            dn_query_label, dn_query_bbox, dn_mask, dn_meta = None, None, None, None

        # First, sent images through DETR base model to obtain encoder + decoder outputs
        outputs: OccupancyDetrModelOutput = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            dn_query_label=dn_query_label,
            dn_query_bbox=dn_query_bbox,
            dn_mask=dn_mask,
            matcher=self.matcher if self.config.early_matching else None,
            labels_dict=labels_dict if self.config.early_matching else None,
        )

        hidden_states = outputs.intermediate_hidden_states
        init_reference = outputs.init_reference_points
        inter_references = outputs.intermediate_reference_points
        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        spatial_shapes = outputs.spatial_shapes
        level_start_index = outputs.level_start_index
        valid_ratios = outputs.valid_ratios
        indices = outputs.query_indices

        # class logits + predicted bounding boxes
        outputs_classes = []
        outputs_boxes = []
        for level in range(hidden_states.shape[1]):
            if level == 0:
                reference = init_reference
            else:
                reference = inter_references[:, level - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[level](hidden_states[:, level])
            delta_bbox = self.bbox_embed[level](hidden_states[:, level])
            if reference.shape[-1] == 4:
                outputs_bbox_logits = delta_bbox + reference
            elif reference.shape[-1] == 2:
                delta_bbox[..., :2] += reference
                outputs_bbox_logits = delta_bbox
            else:
                raise ValueError(f"reference.shape[-1] should be 4 or 2, but got {reference.shape[-1]}")
            outputs_bbox = outputs_bbox_logits.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_boxes.append(outputs_bbox)
        outputs_class = torch.stack(outputs_classes)
        outputs_boxes = torch.stack(outputs_boxes)
        outputs_boxes3d = self.bbox3d_head(hidden_states[:, -1]).sigmoid()
        if self.config.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_boxes, outputs_boxes3d = self.cdn.dn_post_process(
                outputs_class,
                outputs_boxes,
                outputs_boxes3d,
                dn_meta,
                self.config.auxiliary_loss,
                self._set_aux_loss,
            )
        logits = outputs_class[-1]
        pred_boxes = outputs_boxes[-1]
        pred_boxes3d = outputs_boxes3d
        pred_occ = None
        pred_logits = None

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            # First: extract outputs
            outputs_loss = {"ids": ids}
            outputs_loss["logits"] = logits
            outputs_loss["pred_boxes"] = pred_boxes
            outputs_loss["pred_boxes3d"] = pred_boxes3d
            outputs_loss["dn_meta"] = dn_meta
            outputs_loss["enc_outputs"] = {
                "logits": outputs.enc_outputs_class,
                "pred_boxes": outputs.enc_outputs_coord_logits.sigmoid(),
                "proposal_indices": outputs.proposal_indices,
            }
            if self.config.auxiliary_loss:
                auxiliary_outputs = self._set_aux_loss(outputs_class, outputs_boxes)
                outputs_loss["auxiliary_outputs"] = auxiliary_outputs

            # Second: match
            if indices is None:
                indices = self.matcher(outputs_loss, labels_dict)

            # Third: compute the losses, based on outputs and labels
            self.criterion.to(self.device)

            if self.mode == "occupancy":
                b_idx, idx = self.criterion._get_source_permutation_idx(indices)
                target_b_idx, target_idx = self.criterion._get_target_permutation_idx(indices)
                sample = torch.randperm(idx.nelement())[: self.config.occ_decoder_batch_size]
                b_idx, target_b_idx = b_idx[sample], target_b_idx[sample]
                idx, target_idx = idx[sample], target_idx[sample]
                obj_hidden_states = hidden_states[b_idx, -1, idx]
                obj_boxes3d = pred_boxes3d[b_idx, idx].clone()
                obj_boxes3d[:, 3:] += self.config.larger_boxes
                obj_boxes3d = corners_to_center_format3d(center_to_corners_format3d(obj_boxes3d))
                features2d = encoder_last_hidden_state[b_idx]
                cam_Es = cam_Es[b_idx]
                cam_Ks = cam_Ks[b_idx]
                pred_occ = self.occ_transformer(
                    cam_Es,
                    cam_Ks,
                    obj_hidden_states,
                    obj_boxes3d,
                    features2d,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios[b_idx],
                )
                outputs_loss["pred_occ"] = pred_occ
                outputs_loss["enlarged_boxes3d"] = obj_boxes3d
                outputs_loss["occ_idx2target_idx"] = (target_b_idx, target_idx)

            loss_dict = self.criterion(outputs_loss, labels_dict, indices)

            # Fourth: compute total loss, as a weighted sum of the various losses
            weight_dict = {
                "ce": 1,
                "bbox": self.config.bbox_loss_coefficient,
                "bbox3d": self.config.bbox_loss_coefficient,
                "giou": self.config.giou_loss_coefficient,
                "giou3d": self.config.giou_loss_coefficient,
                "dice3d": self.config.mask_loss_coefficient,
            }
            w_losses = []
            for k, v in loss_dict.items():
                tag, loss_k = k.split("_")[:2]
                if tag == "loss" and loss_k in weight_dict:
                    w_losses.append(weight_dict[loss_k] * v)

            loss = sum(w_losses)

            confidences = logits.sigmoid().max(dim=-1).values
            indices = (confidences > self.config.confidence_threshold).nonzero()
            b_idx, idx = indices[:, 0], indices[:, 1]
            obj_hidden_states = hidden_states[:, -1][b_idx, idx]
            pred_logits = logits[b_idx, idx]
            pred_boxes = pred_boxes[b_idx, idx]
            pred_boxes3d = pred_boxes3d[b_idx, idx]
        else:
            confidences = logits.sigmoid().max(dim=-1).values
            indices = (confidences > self.config.confidence_threshold).nonzero()
            b_idx, idx = indices[:, 0], indices[:, 1]
            obj_hidden_states = hidden_states[:, -1][b_idx, idx]
            pred_logits = logits[b_idx, idx]
            pred_boxes = pred_boxes[b_idx, idx]
            pred_boxes3d = pred_boxes3d[b_idx, idx]
            obj_boxes3d = pred_boxes3d.clone()
            obj_boxes3d[:, 3:] += self.config.larger_boxes
            obj_boxes3d = corners_to_center_format3d(center_to_corners_format3d(obj_boxes3d))
            if self.mode == "occupancy":
                occs = []
                batch_size = self.config.occ_decoder_batch_size
                for i in range(0, len(b_idx), batch_size):
                    b_idx_batch = b_idx[i : i + batch_size]
                    obj_hidden_states_batch = obj_hidden_states[i : i + batch_size]
                    obj_boxes3d_batch = obj_boxes3d[i : i + batch_size]
                    features2d_batch = encoder_last_hidden_state[b_idx_batch]
                    cam_Es_batch = cam_Es[b_idx_batch]
                    cam_Ks_batch = cam_Ks[b_idx_batch]
                    occupancy_batch = self.occ_transformer(
                        cam_Es_batch,
                        cam_Ks_batch,
                        obj_hidden_states_batch,
                        obj_boxes3d_batch,
                        features2d_batch,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index,
                        valid_ratios=valid_ratios[b_idx_batch],
                    )
                    occs_batch = occupancy_batch.detach().cpu()
                    del occupancy_batch
                    occs.append(occs_batch)
                pred_occ = torch.cat(occs, dim=0)

        num_preds = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        for i in range(len(b_idx)):
            num_preds[b_idx[i]] += 1
        return OccupancyDetrOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_logits=pred_logits,
            pred_boxes=pred_boxes,
            pred_boxes3d=pred_boxes3d,
            pred_occ=pred_occ,
            num_preds=num_preds,
        )
