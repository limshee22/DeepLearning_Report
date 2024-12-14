import torch
from torch import nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
from d2l import torch as d2l

def concat_preds(preds):
    if not isinstance(preds, list):
        raise ValueError("concat_preds 함수는 리스트 입력만 허용합니다.")
    return torch.cat([torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1) for pred in preds], dim=1)

def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)

def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

class MultiScaleViT(nn.Module):
    def __init__(self, img_size=256, patch_size=16, embed_dim=768, num_heads=12, num_layers=12):
        super(MultiScaleViT, self).__init__()
        self.vit = VisionTransformer(
            img_size=img_size, 
            patch_size=patch_size, 
            embed_dim=embed_dim, 
            depth=num_layers, 
            num_heads=num_heads, 
            mlp_ratio=4, 
            qkv_bias=True, 
            num_classes=0,
            global_pool=''
        )
        self.conv_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        if isinstance(img_size, tuple):
            self.num_patches = (img_size[0] // self.patch_size[0]) * (img_size[1] // self.patch_size[1])
        else:
            self.num_patches = (img_size // self.patch_size[0]) ** 2

    def forward(self, X):
        B, C, H, W = X.shape
        patch_tokens = self.vit.patch_embed(X)
        
        if self.vit.pos_embed.shape[1] > self.num_patches:
            pos_embed = self.vit.pos_embed[:, 1:, :]
        else:
            pos_embed = self.vit.pos_embed
            
        patch_tokens = self.vit.pos_drop(patch_tokens + pos_embed)
        
        for blk in self.vit.blocks:
            patch_tokens = blk(patch_tokens)
        
        patch_h = self.patch_size[0]
        patch_w = self.patch_size[1]
        
        B, N, E = patch_tokens.shape
        H_p, W_p = H // patch_h, W // patch_w
        feature_map = patch_tokens.transpose(1, 2).reshape(B, E, H_p, W_p)
        
        return self.conv_proj(feature_map)

class ViTBasedTinySSD(nn.Module):
    def __init__(self, num_classes, img_size=256, patch_size=16, embed_dim=768, num_heads=12, num_layers=12):
        super(ViTBasedTinySSD, self).__init__()
        self.num_classes = num_classes
        self.backbone = MultiScaleViT(img_size, patch_size, embed_dim, num_heads, num_layers)
        
        self.reduce_layers = nn.ModuleList()
        idx_to_in_channels = [embed_dim, embed_dim // 2, embed_dim // 4, embed_dim // 8, embed_dim // 16]
        reduced_channels = 384
        
        for in_channels in idx_to_in_channels:
            self.reduce_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
                    nn.BatchNorm2d(reduced_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        self.cls_layers = nn.ModuleList()
        self.bbox_layers = nn.ModuleList()
        
        for _ in range(5):
            self.cls_layers.append(cls_predictor(reduced_channels, num_anchors, num_classes))
            self.bbox_layers.append(bbox_predictor(reduced_channels, num_anchors))
            
        self.downsample = nn.ModuleList([
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(embed_dim // 4, embed_dim // 8, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(embed_dim // 8, embed_dim // 16, kernel_size=3, stride=2, padding=1)
        ])

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        
        feature_map = self.backbone(X)
        
        reduced_feat = self.reduce_layers[0](feature_map)
        anchors[0] = d2l.multibox_prior(reduced_feat, sizes[0], ratios[0])
        cls_preds[0] = self.cls_layers[0](reduced_feat)
        bbox_preds[0] = self.bbox_layers[0](reduced_feat)
        
        current_feat = feature_map
        for i in range(4):
            current_feat = self.downsample[i](current_feat)
            reduced_feat = self.reduce_layers[i+1](current_feat)
            anchors[i+1] = d2l.multibox_prior(reduced_feat, sizes[i+1], ratios[i+1])
            cls_preds[i+1] = self.cls_layers[i+1](reduced_feat)
            bbox_preds[i+1] = self.bbox_layers[i+1](reduced_feat)
        
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds

sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1
