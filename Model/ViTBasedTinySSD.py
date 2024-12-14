import torch
from torch import nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
from d2l import torch as d2l
# 클래스 예측 레이어
def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)

# 바운딩 박스 예측 레이어
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

# 멀티스케일 예측 결과 병합
def concat_preds(preds):
    return torch.cat([torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1) for pred in preds], dim=1)

# 멀티스케일 ViT 블록
class MultiScaleViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_heads=12, num_layers=12):
        super(MultiScaleViT, self).__init__()
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=num_layers,
            num_heads=num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            num_classes=0  # 분류용이 아님
        )
        self.conv_proj = nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1)

    def forward(self, X):
        B, C, H, W = X.shape
        patch_tokens = self.vit.patch_embed(X)  # (B, num_patches, embed_dim)
        if patch_tokens.shape[1] != self.vit.pos_embed.shape[1]:
            # Positional Embedding 크기 맞추기
            pos_embed = F.interpolate(
                self.vit.pos_embed.unsqueeze(0),
                size=(patch_tokens.shape[1], self.vit.pos_embed.shape[2]),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        else:
            pos_embed = self.vit.pos_embed
        patch_tokens = self.vit.pos_drop(patch_tokens + pos_embed)

        for blk in self.vit.blocks:
            patch_tokens = blk(patch_tokens)

        B, N, E = patch_tokens.shape
        H_p, W_p = H // self.vit.patch_embed.patch_size[0], W // self.vit.patch_embed.patch_size[1]
        feature_map = patch_tokens.transpose(1, 2).reshape(B, E, H_p, W_p)

        return self.conv_proj(feature_map)

# ViT 기반 TinySSD 모델 정의
class ViTBasedTinySSD(nn.Module):
    def __init__(self, num_classes, img_size=224, patch_size=16, embed_dim=768, num_heads=12, num_layers=12):
        super(ViTBasedTinySSD, self).__init__()
        self.num_classes = num_classes
        self.backbone = MultiScaleViT(img_size, patch_size, embed_dim, num_heads, num_layers)
        idx_to_in_channels = [embed_dim // 2, embed_dim // 4, embed_dim // 8, embed_dim // 16, embed_dim // 32]
        self.cls_layers = nn.ModuleList()
        self.bbox_layers = nn.ModuleList()

        self.sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
        self.ratios = [[1, 2, 0.5]] * 5
        self.num_anchors = len(self.sizes[0]) + len(self.ratios[0]) - 1

        for i in range(5):
            self.cls_layers.append(cls_predictor(idx_to_in_channels[i], self.num_anchors, num_classes))
            self.bbox_layers.append(bbox_predictor(idx_to_in_channels[i], self.num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5

        feature_map = self.backbone(X)

        for i in range(5):
            downsampled = F.adaptive_avg_pool2d(feature_map, (feature_map.shape[2] // (2 ** i), feature_map.shape[3] // (2 ** i)))
            anchors[i] = d2l.multibox_prior(downsampled, self.sizes[i], self.ratios[i])
            cls_preds[i] = self.cls_layers[i](downsampled)
            bbox_preds[i] = self.bbox_layers[i](downsampled)

        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds).reshape(cls_preds[0].shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
