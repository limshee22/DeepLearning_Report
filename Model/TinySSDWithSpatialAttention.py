import torch
from torch import nn
import torch.nn.functional as F
from d2l import torch as d2l


# Spatial Attention Block 정의
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        avg_out = torch.mean(X, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(X, dim=1, keepdim=True)  # (B, 1, H, W)
        combined = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        attention = self.sigmoid(self.conv(combined))  # (B, 1, H, W)
        return X * attention


# 클래스 예측 레이어
def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)


# 바운딩 박스 예측 레이어
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)


# 다운샘플링 블록 + Spatial Attention
def down_sample_blk_with_spatial_attention(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(SpatialAttention(kernel_size=7))  # Spatial Attention 추가
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)


# 베이스 네트워크 정의
def base_net_with_spatial_attention():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk_with_spatial_attention(num_filters[i], num_filters[i + 1]))
    return nn.Sequential(*blk)


# 블록 전방 전달
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return Y, anchors, cls_preds, bbox_preds


# 멀티스케일 예측 결과 병합
def concat_preds(preds):
    if not isinstance(preds, list):
        raise ValueError("concat_preds 함수는 리스트 입력만 허용합니다.")
    return torch.cat([torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1) for pred in preds], dim=1)


# TinySSD 모델 정의
class TinySSDWithSpatialAttention(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSDWithSpatialAttention, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        self.blocks = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        self.bbox_layers = nn.ModuleList()

        for i in range(5):
            self.blocks.append(self.get_blk(i))
            self.cls_layers.append(cls_predictor(idx_to_in_channels[i], num_anchors, num_classes))
            self.bbox_layers.append(bbox_predictor(idx_to_in_channels[i], num_anchors))

    def get_blk(self, i):
        if i == 0:
            return base_net_with_spatial_attention()  # Spatial Attention 블록이 포함된 베이스 네트워크
        elif i == 4:
            return nn.AdaptiveMaxPool2d((1, 1))
        else:
            return down_sample_blk_with_spatial_attention(128 if i > 1 else 64, 128)

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i, blk in enumerate(self.blocks):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, blk, sizes[i], ratios[i], self.cls_layers[i], self.bbox_layers[i]
            )
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds


# 멀티스케일 앵커 설정
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1
