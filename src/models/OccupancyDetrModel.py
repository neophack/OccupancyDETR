from transformers.models.deformable_detr.modeling_deformable_detr import *

from .DinoDecoder import DinoDecoder
from .OccupancyDetrConfig import OccupancyDetrConfig
from .OccupancyDetrPretrainedModel import OccupancyDetrPretrainedModel


@dataclass
class OccupancyDetrModelOutput(DeformableDetrModelOutput):
    spatial_shapes: Optional[torch.LongTensor] = None
    level_start_index: Optional[torch.LongTensor] = None
    valid_ratios: Optional[torch.FloatTensor] = None
    output_proposals: Optional[torch.FloatTensor] = None
    query_indices: Optional[List] = None
    proposal_indices: Optional[List] = None


class OccupancyDetrModel(OccupancyDetrPretrainedModel):
    def __init__(self, config: OccupancyDetrConfig):
        super().__init__(config)
        self.config = config

        # Create backbone + positional encoding
        backbone = DeformableDetrConvEncoder(config)
        position_embeddings = build_position_encoding(config)
        self.backbone = DeformableDetrConvModel(backbone, position_embeddings)

        # Create input projection layers
        if config.num_feature_levels > 1:
            num_backbone_outs = len(backbone.intermediate_channel_sizes)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.intermediate_channel_sizes[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, config.d_model, kernel_size=1),
                        nn.GroupNorm(32, config.d_model),
                    )
                )
            for _ in range(config.num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, config.d_model, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, config.d_model),
                    )
                )
                in_channels = config.d_model
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.intermediate_channel_sizes[-1], config.d_model, kernel_size=1),
                        nn.GroupNorm(32, config.d_model),
                    )
                ]
            )

        self.encoder = DeformableDetrEncoder(config)
        self.decoder = DinoDecoder(config)

        self.level_embed = nn.Parameter(torch.Tensor(config.num_feature_levels, config.d_model))

        self.enc_output = nn.Linear(config.d_model, config.d_model)
        self.enc_output_norm = nn.LayerNorm(config.d_model)
        self.query_embeddings = nn.Embedding(config.num_queries, config.d_model)
        self.reference_points = nn.Linear(config.d_model, 4)

        self.post_init()

    def freeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(False)

    def unfreeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(True)

    def get_valid_ratio(self, mask):
        """Get the valid ratio of all feature maps."""

        _, height, width = mask.shape
        valid_height = torch.sum(mask[:, :, 0], 1)
        valid_width = torch.sum(mask[:, 0, :], 1)
        valid_ratio_heigth = valid_height.float() / height
        valid_ratio_width = valid_width.float() / width
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_heigth], -1)
        return valid_ratio

    def gen_encoder_output_proposals(self, enc_output, padding_mask, spatial_shapes):
        """Generate the encoder output proposals from encoded enc_output."""

        batch_size = enc_output.shape[0]
        proposals = []
        _cur = 0
        for level, (height, width) in enumerate(spatial_shapes):
            mask_flatten_ = padding_mask[:, _cur : (_cur + height * width)].view(batch_size, height, width, 1)
            valid_height = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_width = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = meshgrid(
                torch.linspace(0, height - 1, height, dtype=torch.float32, device=enc_output.device),
                torch.linspace(0, width - 1, width, dtype=torch.float32, device=enc_output.device),
                indexing="ij",
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_width.unsqueeze(-1), valid_height.unsqueeze(-1)], 1).view(batch_size, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(batch_size, -1, -1, -1) + 0.5) / scale
            width_heigth = torch.ones_like(grid) * 0.05 * (2.0**level)
            proposal = torch.cat((grid, width_heigth), -1).view(batch_size, -1, 4)
            proposals.append(proposal)
            _cur += height * width
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))  # inverse sigmoid
        proposals_mask = padding_mask.unsqueeze(-1) | ~output_proposals_valid
        output_proposals = output_proposals.masked_fill(proposals_mask, float("-inf"))

        # assign each pixel as an object query
        object_query = enc_output
        object_query = object_query.masked_fill(proposals_mask, float(0))
        object_query = self.enc_output_norm(self.enc_output(object_query))
        return object_query, output_proposals, proposals_mask

    def forward(
        self,
        pixel_values,
        pixel_mask=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        dn_query_label=None,
        dn_query_bbox=None,
        dn_mask=None,
        matcher=None,
        labels_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), dtype=torch.long, device=device)

        # Extract multi-scale feature maps of same resolution `config.d_model` (cf Figure 4 in paper)
        # First, sent pixel_values + pixel_mask through Backbone to obtain the features
        features, position_embeddings_list = self.backbone(pixel_values, pixel_mask)

        # Then, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        sources = []
        masks = []
        for level, (source, mask) in enumerate(features):
            sources.append(self.input_proj[level](source))
            masks.append(mask)
            if mask is None:
                raise ValueError("No attention mask was provided")

        # Lowest resolution feature maps are obtained via 3x3 stride 2 convolutions on the final stage
        if self.config.num_feature_levels > len(sources):
            _len_sources = len(sources)
            for level in range(_len_sources, self.config.num_feature_levels):
                if level == _len_sources:
                    source = self.input_proj[level](features[-1][0])
                else:
                    source = self.input_proj[level](sources[-1])
                mask = nn.functional.interpolate(pixel_mask[None].float(), size=source.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone.position_embedding(source, mask).to(source.dtype)
                sources.append(source)
                masks.append(mask)
                position_embeddings_list.append(pos_l)

        # Prepare encoder inputs (by flattening)
        source_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for level, (source, mask, pos_embed) in enumerate(zip(sources, masks, position_embeddings_list)):
            batch_size, num_channels, height, width = source.shape
            spatial_shape = (height, width)
            spatial_shapes.append(spatial_shape)
            source = source.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[level].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            source_flatten.append(source)
            mask_flatten.append(mask)
        source_flatten = torch.cat(source_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=source_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        valid_ratios = valid_ratios.float()

        # Transformer encoder
        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=source_flatten,
            attention_mask=mask_flatten,
            position_embeddings=lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # Prepare decoder inputs
        output_memory, output_proposals, proposals_mask = self.gen_encoder_output_proposals(encoder_outputs.last_hidden_state, ~mask_flatten, spatial_shapes)
        enc_outputs_class = self.decoder.class_embed[-1](output_memory)
        enc_outputs_coord_logits = self.decoder.bbox_embed[-1](output_memory) + output_proposals

        indices = None
        query_indices = None
        if labels_dict is not None and matcher is not None:
            outputs = {"pred_boxes": output_proposals.sigmoid()}
            indices = matcher(outputs, labels_dict, proposals_mask)
            query_indices = []
            topk_proposals = []
            enc_outputs_class_detach = enc_outputs_class.detach().clone()
            for b, (b_source_idx, b_target_idx) in enumerate(indices):
                left_topk = self.config.num_queries - b_source_idx.shape[0]
                enc_outputs_class_detach[b, b_source_idx, :] = float("-inf")
                left_topk_proposals = torch.topk(enc_outputs_class_detach[b, :, 0], left_topk, dim=0)[1]
                topk_proposals.append(torch.cat([b_source_idx.to(device), left_topk_proposals], dim=0))
                new_source_idx = torch.arange(b_source_idx.shape[0], device=device)
                query_indices.append((new_source_idx, b_target_idx))
            topk_proposals = torch.stack(topk_proposals, dim=0)
        else:
            topk = self.config.num_queries
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]

        topk_coords_unact = torch.gather(enc_outputs_coord_logits, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).detach()
        reference_points = topk_coords_unact.sigmoid()
        query = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])).detach()
        if dn_query_bbox is not None and dn_query_label is not None:
            query = self.query_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            reference_points = torch.cat([dn_query_bbox.sigmoid(), reference_points], 1)
            query = torch.cat([dn_query_label, query], dim=1)
        init_reference_points = reference_points

        # Transformer decoder
        decoder_outputs: DeformableDetrDecoderOutput = self.decoder(
            inputs_embeds=query,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            dn_mask=dn_mask,
        )

        return OccupancyDetrModelOutput(
            init_reference_points=init_reference_points,
            last_hidden_state=decoder_outputs.last_hidden_state,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord_logits=enc_outputs_coord_logits,
            output_proposals=output_proposals,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            query_indices=query_indices,
            proposal_indices=indices,
        )
