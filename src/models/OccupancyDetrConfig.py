from transformers.models.deformable_detr.configuration_deformable_detr import DeformableDetrConfig


class OccupancyDetrConfig(DeformableDetrConfig):
    def __init__(
        self,
        # Dataset parameters
        voxel_shape=(256, 256, 32),
        voxel_size=0.2,
        voxel_origin=(0, -25.6, -2),
        image_width=1220,
        image_height=370,
        # OccupancyDetr parameters
        query_mode="bev",  # "bev" or "3d"
        query_shape=(8, 8, 2),
        larger_boxes=0.05,
        confidence_threshold=0.8,
        occupancy_threshold=0.5,
        occ_decoder_layers=6,
        occ_decoder_batch_size=32,
        # Training parameters
        early_matching=True,
        only_encoder_loss=False,
        dn_number=0,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        # original DeformableDetrConfig parameters
        use_timm_backbone=True,
        backbone_config=None,
        num_channels=3,
        num_queries=100,
        encoder_layers=6,
        encoder_ffn_dim=1024,
        encoder_attention_heads=8,
        decoder_layers=6,
        decoder_ffn_dim=1024,
        decoder_attention_heads=8,
        is_encoder_decoder=True,
        activation_function="relu",
        d_model=256,
        position_embedding_type="sine",
        backbone="resnet50",
        use_pretrained_backbone=True,
        num_feature_levels=4,
        encoder_n_points=4,
        decoder_n_points=4,
        **kwargs,
    ):
        super().__init__(
            use_timm_backbone=use_timm_backbone,
            backbone_config=backbone_config,
            num_channels=num_channels,
            num_queries=num_queries,
            encoder_layers=encoder_layers,
            encoder_ffn_dim=encoder_ffn_dim,
            encoder_attention_heads=encoder_attention_heads,
            decoder_layers=decoder_layers,
            decoder_ffn_dim=decoder_ffn_dim,
            decoder_attention_heads=decoder_attention_heads,
            is_encoder_decoder=is_encoder_decoder,
            activation_function=activation_function,
            d_model=d_model,
            position_embedding_type=position_embedding_type,
            backbone=backbone,
            use_pretrained_backbone=use_pretrained_backbone,
            num_feature_levels=num_feature_levels,
            encoder_n_points=encoder_n_points,
            decoder_n_points=decoder_n_points,
            **kwargs,
        )

        self.voxel_shape = voxel_shape
        self.voxel_size = voxel_size
        self.voxel_origin = voxel_origin
        self.image_width = image_width
        self.image_height = image_height
        self.query_mode = query_mode
        self.query_shape = query_shape
        self.larger_boxes = larger_boxes
        self.confidence_threshold = confidence_threshold
        self.occupancy_threshold = occupancy_threshold
        self.occ_decoder_layers = occ_decoder_layers
        self.occ_decoder_batch_size = occ_decoder_batch_size
        self.early_matching = early_matching
        self.only_encoder_loss = only_encoder_loss
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.pad_token_id = 0
        self.dropout = 0
        self.encoder_layerdrop = 0
        self.attention_dropout = 0
        self.activation_dropout = 0
        self.two_stage = True
        self.with_box_refine = True
        self.auxiliary_loss = True
