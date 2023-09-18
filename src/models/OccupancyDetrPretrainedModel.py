from transformers.models.deformable_detr.modeling_deformable_detr import *

from .OccupancyDetrConfig import OccupancyDetrConfig


class OccupancyDetrPretrainedModel(DeformableDetrPreTrainedModel):
    config_class = OccupancyDetrConfig
