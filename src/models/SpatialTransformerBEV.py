from transformers.models.deformable_detr.modeling_deformable_detr import *
from transformers.models.detr.modeling_detr import *

from .OccupancyDetrConfig import OccupancyDetrConfig
from .OccupancyDetrPretrainedModel import OccupancyDetrPretrainedModel


class SpatialDecoderLayer(nn.Module):
    def __init__(self, config: OccupancyDetrConfig):
        super().__init__()

        d_model = config.d_model
        self_config = copy.deepcopy(config)
        self_config.num_feature_levels = 1
        self.self_attn = DeformableDetrMultiscaleDeformableAttention(self_config, num_heads=config.encoder_attention_heads, n_points=config.encoder_n_points)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.encoder_attn = DeformableDetrMultiscaleDeformableAttention(config, num_heads=config.encoder_attention_heads, n_points=config.encoder_n_points)
        self.encoder_attn_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_position_embeddings: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        reference_points_bev: torch.Tensor,
        spatial_shapes_bev: torch.Tensor,
        level_start_index_bev: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ):
        # Self Attention
        residual = hidden_states

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=query_position_embeddings,
            encoder_hidden_states=hidden_states,
            reference_points=reference_points_bev,
            spatial_shapes=spatial_shapes_bev,
            level_start_index=level_start_index_bev,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross Attention
        residual = hidden_states

        hidden_states, _ = self.encoder_attn(
            hidden_states=hidden_states,
            position_embeddings=query_position_embeddings,
            encoder_hidden_states=encoder_hidden_states,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class VoxelLearnedPositionEmbedding(nn.Module):
    def __init__(self, embedding_dim, voxel_shape, query_shape):
        super().__init__()
        self.voxel_shape = voxel_shape
        self.query_shape = query_shape
        each_dim = embedding_dim // 3
        self.x_embeddings = nn.Embedding(voxel_shape[0], each_dim)
        self.y_embeddings = nn.Embedding(voxel_shape[1], each_dim)
        self.z_embeddings = nn.Embedding(voxel_shape[2], embedding_dim - 2 * each_dim)

    def forward(self, boxes3d):
        coords = torch.ones((*self.query_shape[:2], 1), device=boxes3d.device).nonzero()
        coords = coords.unsqueeze(0).repeat(boxes3d.shape[0], 1, 1)
        xs = boxes3d[:, 0] * self.voxel_shape[0]
        ys = boxes3d[:, 1] * self.voxel_shape[1]
        zs = boxes3d[:, 2] * self.voxel_shape[2]
        ds = boxes3d[:, 3] * self.voxel_shape[0]
        ws = boxes3d[:, 4] * self.voxel_shape[1]
        minxs = torch.max(torch.zeros_like(xs), xs - ds / 2)
        minys = torch.max(torch.zeros_like(ys), ys - ws / 2)
        ds = torch.max(torch.ones_like(ds), torch.min(ds, self.voxel_shape[0] - minxs))
        ws = torch.max(torch.ones_like(ws), torch.min(ws, self.voxel_shape[1] - minys))
        coords_x = coords[:, :, 0] * ds[:, None] / self.query_shape[0] + minxs[:, None]
        coords_y = coords[:, :, 1] * ws[:, None] / self.query_shape[1] + minys[:, None]
        coords_z = zs.unsqueeze(1).repeat(1, self.query_shape[0] * self.query_shape[1])
        coords = torch.stack([coords_x, coords_y, coords_z], dim=-1).floor()
        x_emb = self.x_embeddings(coords_x.long())
        y_emb = self.y_embeddings(coords_y.long())
        z_emb = self.z_embeddings(coords_z.long())
        emb = torch.cat([x_emb, y_emb, z_emb], dim=-1)
        return emb, coords


def voxel2world(coords, origin, scale, offsets=(0.5, 0.5, 0.5)):
    offsets = torch.tensor(offsets, device=coords.device)
    world_pts = origin + (scale * coords) + scale * offsets
    return world_pts


def world2cam(xyz, cam_Es):
    ones = torch.ones((*xyz.shape[:2], 1), device=xyz.device)
    xyz_h = torch.cat((xyz, ones), dim=2)
    xyz_t_h = torch.bmm(cam_Es, xyz_h.transpose(1, 2)).transpose(1, 2)
    return xyz_t_h[:, :, :3]


def cam2pix(cam_pts, cam_Ks):
    cam_pts = cam_pts.clone()
    fx, fy = cam_Ks[:, 0, 0], cam_Ks[:, 1, 1]
    cx, cy = cam_Ks[:, 0, 2], cam_Ks[:, 1, 2]
    cam_pts[:, :, :2] = cam_pts[:, :, :2] / cam_pts[:, :, 2, None]
    cam_pts[:, :, 0] = cam_pts[:, :, 0] * fx.unsqueeze(1) + cx.unsqueeze(1)
    cam_pts[:, :, 1] = cam_pts[:, :, 1] * fy.unsqueeze(1) + cy.unsqueeze(1)
    pix = torch.round(cam_pts).long()
    return pix[:, :, :2]


class SpatialTransformerBEV(OccupancyDetrPretrainedModel):
    def __init__(self, config: OccupancyDetrConfig):
        super().__init__(config)
        self.voxel_shape = config.voxel_shape
        self.voxel_origin = config.voxel_origin
        self.voxel_size = config.voxel_size
        self.image_size = (config.image_width, config.image_height)
        self.query_shape = config.query_shape
        self.H = config.query_shape[2]

        d_model = config.d_model
        self.query_position_embeddings = VoxelLearnedPositionEmbedding(d_model, self.voxel_shape, self.query_shape)
        self.decoder_layers = []
        for _ in range(config.occ_decoder_layers):
            self.decoder_layers.append(SpatialDecoderLayer(config))
        self.decoder_layers = nn.ModuleList(self.decoder_layers)
        self.occupancy_head = nn.Linear(d_model, self.H)

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def get_reference_points(spatial_shapes, device):
        reference_points_list = []
        for level, (depth, width) in enumerate(spatial_shapes):
            ref_x, ref_y = meshgrid(
                torch.linspace(0.5, depth - 0.5, depth, dtype=torch.float32, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_x = ref_x.reshape(-1)[None] / depth
            ref_y = ref_y.reshape(-1)[None] / width
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
        return reference_points

    def forward(
        self,
        cam_Es,
        cam_Ks,
        context,
        boxes3d,
        features2d,
        spatial_shapes,
        level_start_index,
        valid_ratios,
    ):
        shape = (context.shape[0], *self.query_shape[:2], context.shape[1])

        position_embeddings, coords = self.query_position_embeddings(boxes3d)
        hidden_states = context[:, None, None, :].expand(shape).detach()

        device = hidden_states.device
        spatial_shapes_bev = torch.tensor([self.query_shape[:2]], device=device)
        level_start_index_bev = torch.tensor([0], device=device)
        reference_points_bev = self.get_reference_points(spatial_shapes_bev, device)
        spatial_shapes_bev = spatial_shapes_bev[:, [1, 0]]  # adjust for image deformable attn

        voxel_origin = torch.tensor(self.voxel_origin, device=device)
        world_pts = voxel2world(coords, voxel_origin, self.voxel_size)
        cam_pts = world2cam(world_pts, cam_Es)
        pixs = cam2pix(cam_pts, cam_Ks)
        reference_points = pixs / torch.tensor(self.image_size, device=device)[None, None, :]
        reference_points = torch.clamp(reference_points, 0.05, 0.95)
        reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]
        for decoder_layer in self.decoder_layers:
            hidden_states = decoder_layer(
                hidden_states.flatten(1, 2),
                query_position_embeddings=position_embeddings,
                encoder_hidden_states=features2d,
                reference_points_bev=reference_points_bev,
                spatial_shapes_bev=spatial_shapes_bev,
                level_start_index_bev=level_start_index_bev,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )
            hidden_states = hidden_states.view(shape)

        occupancy = self.occupancy_head(hidden_states.flatten(1, 2)).view(*shape[:3], self.H).sigmoid()

        return occupancy
